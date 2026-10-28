#!/usr/bin/env python3
"""
Invisibility Portal
===================
Real-time, hand-tracked "portal" effects with MediaPipe + OpenCV.

Each visible hand opens a portal anchored to its index fingertip. Inside the
portal you see a live filter. The default filter reveals a background plate
captured at startup, so anything in front of it (you, your hand, a mug)
looks invisible.

Gestures (per hand, no keyboard needed)
  * Spread / close thumb and index finger   -> grow / shrink the portal
  * Pinch thumb + MIDDLE fingertip           -> next effect
  * Pinch thumb + RING fingertip             -> next shape (circle / hexagon / star)

Keyboard (optional extras)
  b      re-capture background (step out of frame first)
  e / s  next effect / next shape          1-7  pick effect directly
  l      toggle hand landmark overlay      f    toggle fullscreen
  q/Esc  quit

Run:  python invisibility_portal.py [--camera 0] [--width 1280] [--height 720]
"""

import argparse
import math
import os
import ssl
import sys
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
EFFECTS = ["Invisibility", "Swirl", "Thermal", "Pixelate", "Neon Edges", "Mirror", "Grayscale"]
EFFECT_COLORS = {  # BGR outline colour per effect
    "Invisibility": (255, 220, 120),
    "Swirl": (255, 120, 220),
    "Thermal": (40, 140, 255),
    "Pixelate": (120, 255, 120),
    "Neon Edges": (255, 255, 0),
    "Mirror": (235, 235, 235),
    "Grayscale": (185, 185, 185),
}
SHAPES = ["Circle", "Hexagon", "Star"]

MIN_R = 45                  # smallest portal radius (px)
MAX_R_FRAC = 0.45           # largest radius as a fraction of the shorter frame side
SIZE_GAIN = 1.0             # radius = thumb-index spread * gain
POS_RATE, SIZE_RATE = 18.0, 9.0   # smoothing speeds (higher = snappier)
PINCH_ON, PINCH_OFF = 0.28, 0.42  # pinch thresholds, relative to palm length
PINCH_MARGIN = 0.08         # pinch must be clearly closer than the neighbouring finger
GESTURE_COOLDOWN = 0.45     # seconds between gesture triggers per hand
LOST_GRACE = 0.25           # keep a portal alive this long after a hand drops out
COUNTDOWN_S = 3.0           # seconds before the background plate is captured
BG_FRAMES = 15              # frames median-averaged into the background plate

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, RING_TIP = 0, 4, 8, 9, 12, 16

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")


# --------------------------------------------------------------------------
# Hand tracking (works with both the classic and the newer MediaPipe APIs)
# --------------------------------------------------------------------------
def ensure_model():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
    if not os.path.exists(path):
        print("Downloading MediaPipe hand model (one time, ~8 MB)...")
        # python.org builds on macOS ship without root certificates, so use certifi's bundle.
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()
        with urllib.request.urlopen(MODEL_URL, context=ctx) as resp, open(path + ".part", "wb") as f:
            f.write(resp.read())
        os.replace(path + ".part", path)
    return path


class HandTracker:
    """Returns a list of (label, landmarks[21x2] in pixels) per detected hand."""

    def __init__(self, max_hands=2):
        solutions = getattr(mp, "solutions", None)
        if solutions is not None and hasattr(solutions, "hands"):
            self.mode = "solutions"
            self._hands = solutions.hands.Hands(
                max_num_hands=max_hands, model_complexity=1,
                min_detection_confidence=0.6, min_tracking_confidence=0.5)
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            self.mode = "tasks"
            opts = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
                running_mode=vision.RunningMode.VIDEO, num_hands=max_hands,
                min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5)
            self._hands = vision.HandLandmarker.create_from_options(opts)
            self._t0 = time.monotonic()
            self._last_ts = -1

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        found = []
        if self.mode == "solutions":
            rgb.flags.writeable = False
            res = self._hands.process(rgb)
            if res.multi_hand_landmarks:
                for i, lm in enumerate(res.multi_hand_landmarks):
                    label = (res.multi_handedness[i].classification[0].label
                             if res.multi_handedness else str(i))
                    pts = np.array([(p.x * w, p.y * h) for p in lm.landmark], np.float32)
                    found.append((label, pts))
        else:
            ts = int((time.monotonic() - self._t0) * 1000)
            ts = max(ts, self._last_ts + 1)
            self._last_ts = ts
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self._hands.detect_for_video(img, ts)
            for i, lm in enumerate(res.hand_landmarks):
                label = res.handedness[i][0].category_name if res.handedness else str(i)
                pts = np.array([(p.x * w, p.y * h) for p in lm], np.float32)
                found.append((label, pts))

        # Two hands with the same label (rare misclassification) -> make keys unique
        out, seen = [], set()
        for label, pts in found:
            if label in seen:
                label = f"{label}-{len(out)}"
            seen.add(label)
            out.append((label, pts))
        return out


# --------------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------------
def shape_polygon(shape, cx, cy, r, rot):
    if shape == "Hexagon":
        a = rot + np.arange(6) * (np.pi / 3)
        rr = np.full(6, r, np.float64)
    else:  # Star
        a = rot - np.pi / 2 + np.arange(10) * (np.pi / 5)
        rr = np.where(np.arange(10) % 2 == 0, r, r * 0.45)
    pts = np.stack([cx + rr * np.cos(a), cy + rr * np.sin(a)], axis=1)
    return np.round(pts).astype(np.int32).reshape(-1, 1, 2)


def draw_shape(img, shape, cx, cy, r, rot, color, thickness):
    if shape == "Circle":
        cv2.circle(img, (int(round(cx)), int(round(cy))), int(round(r)),
                   color, thickness, cv2.LINE_AA)
        return
    pts = shape_polygon(shape, cx, cy, r, rot)
    if thickness < 0:
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
    else:
        cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Effects: each takes the source crop, background crop, portal centre/radius
# in crop coordinates, and time; returns a crop of the same size.
# --------------------------------------------------------------------------
def fx_invisibility(src, bg, cx, cy, r, t):
    return bg


def fx_swirl(src, bg, cx, cy, r, t):
    h, w = src.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float32)
    dx, dy = xx - cx, yy - cy
    d = np.sqrt(dx * dx + dy * dy)
    falloff = np.clip(1.0 - d / max(r, 1.0), 0.0, 1.0) ** 2
    ang = np.arctan2(dy, dx) + falloff * (3.0 + 1.5 * math.sin(t * 1.7))
    mx = (cx + d * np.cos(ang)).astype(np.float32)
    my = (cy + d * np.sin(ang)).astype(np.float32)
    return cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def fx_thermal(src, bg, cx, cy, r, t):
    g = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(cv2.equalizeHist(g), (5, 5), 0)
    return cv2.applyColorMap(g, cv2.COLORMAP_JET)


def fx_pixelate(src, bg, cx, cy, r, t):
    h, w = src.shape[:2]
    block = max(6, int(r // 9))
    small = cv2.resize(src, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def fx_neon(src, bg, cx, cy, r, t):
    g = cv2.GaussianBlur(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    edges = cv2.dilate(cv2.Canny(g, 60, 140), np.ones((2, 2), np.uint8))
    hue = int((t * 60) % 180)
    color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
    lines = np.zeros_like(src)
    lines[edges > 0] = color
    glow = cv2.convertScaleAbs(cv2.GaussianBlur(lines, (0, 0), 4), alpha=2.0)
    base = cv2.convertScaleAbs(src, alpha=0.2)
    return cv2.add(cv2.add(base, glow), lines)


def fx_mirror(src, bg, cx, cy, r, t):
    h, w = src.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float32)
    return cv2.remap(src, (2 * cx - xx).astype(np.float32), yy,
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def fx_grayscale(src, bg, cx, cy, r, t):
    return cv2.cvtColor(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


EFFECT_FUNCS = {
    "Invisibility": fx_invisibility, "Swirl": fx_swirl, "Thermal": fx_thermal,
    "Pixelate": fx_pixelate, "Neon Edges": fx_neon, "Mirror": fx_mirror,
    "Grayscale": fx_grayscale,
}


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def render_portal(frame, out, bg, cx, cy, r, shape, effect, t):
    """Composite one portal into `out` (in place)."""
    H, W = frame.shape[:2]
    pad = int(r * 0.15) + 14
    x0, y0 = max(0, int(cx - r - pad)), max(0, int(cy - r - pad))
    x1, y1 = min(W, int(cx + r + pad)), min(H, int(cy + r + pad))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    lcx, lcy = cx - x0, cy - y0
    rot = t * 0.6

    # Feathered mask
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    draw_shape(mask, shape, lcx, lcy, r, rot, 255, -1)
    k = max(3, int(r * 0.08)) | 1
    alpha = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32)[..., None] / 255.0

    src = frame[y0:y1, x0:x1]
    fx = EFFECT_FUNCS[effect](src, bg[y0:y1, x0:x1], lcx, lcy, r, t)
    region = out[y0:y1, x0:x1]
    blended = fx.astype(np.float32) * alpha + region.astype(np.float32) * (1.0 - alpha)
    region[:] = blended.astype(np.uint8)

    # Glowing rim
    color = EFFECT_COLORS[effect]
    pulse = 0.75 + 0.25 * math.sin(t * 4)
    ring = np.zeros_like(region)
    draw_shape(ring, shape, lcx, lcy, r, rot, color, 7)
    ring = cv2.convertScaleAbs(cv2.GaussianBlur(ring, (0, 0), 5), alpha=1.6 * pulse)
    region[:] = cv2.add(region, ring)
    draw_shape(region, shape, lcx, lcy, r, rot, (255, 255, 255), 2)


def put_text(img, text, org, scale=0.6, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def put_centered(img, text, y, scale, color=(255, 255, 255), thick=2):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    put_text(img, text, ((img.shape[1] - tw) // 2, y), scale, color, thick)


def draw_hud(img, effect, shape, fps, toast, toast_age):
    H, W = img.shape[:2]
    panel = img[8:44, 8:470]
    panel[:] = cv2.addWeighted(panel, 0.4, np.zeros_like(panel), 0.6, 0)
    put_text(img, f"Effect: {effect}   Shape: {shape}   {fps:4.1f} FPS", (18, 32), 0.6,
             EFFECT_COLORS[effect])
    put_text(img, "Thumb-index spread: size | Thumb+middle pinch: effect | "
                  "Thumb+ring pinch: shape | B: new background | Q: quit",
             (12, H - 14), 0.45, (220, 220, 220))
    if toast and toast_age < 1.2:
        fade = 1.0 - max(0.0, toast_age - 0.7) / 0.5
        c = tuple(int(v * fade) for v in (255, 255, 255))
        put_centered(img, toast, 90, 1.1, c, 2)


def draw_landmarks(img, pts):
    for i in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP):
        cv2.circle(img, tuple(int(v) for v in pts[i]), 6, (0, 255, 255), -1, cv2.LINE_AA)
    for a, b in ((THUMB_TIP, INDEX_TIP), (THUMB_TIP, MIDDLE_TIP), (THUMB_TIP, RING_TIP)):
        cv2.line(img, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]),
                 (0, 200, 255), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Gestures
# --------------------------------------------------------------------------
def dist(a, b):
    return float(np.linalg.norm(a - b))


def new_portal(pts, max_r):
    r = float(np.clip(dist(pts[THUMB_TIP], pts[INDEX_TIP]) * SIZE_GAIN, MIN_R, max_r))
    return {"pos": pts[INDEX_TIP].astype(np.float64).copy(), "r": r, "r_target": r,
            "mid": False, "ring": False, "cool": 0.0, "seen": 0.0, "pts": pts}


def update_portal(p, pts, now, dt, max_r):
    """Smooth position/size and detect pinch gestures. Returns list of events."""
    palm = dist(pts[WRIST], pts[MIDDLE_MCP]) + 1e-6
    d_idx = dist(pts[THUMB_TIP], pts[INDEX_TIP]) / palm
    d_mid = dist(pts[THUMB_TIP], pts[MIDDLE_TIP]) / palm
    d_ring = dist(pts[THUMB_TIP], pts[RING_TIP]) / palm
    events = []

    mid_now = d_mid < PINCH_ON and d_idx > d_mid + PINCH_MARGIN
    ring_now = d_ring < PINCH_ON and d_mid > d_ring + PINCH_MARGIN

    if not p["mid"] and mid_now:
        p["mid"] = True
        if now >= p["cool"]:
            events.append("effect")
            p["cool"] = now + GESTURE_COOLDOWN
    elif p["mid"] and d_mid > PINCH_OFF:
        p["mid"] = False

    if not p["ring"] and ring_now:
        p["ring"] = True
        if now >= p["cool"]:
            events.append("shape")
            p["cool"] = now + GESTURE_COOLDOWN
    elif p["ring"] and d_ring > PINCH_OFF:
        p["ring"] = False

    # Freeze the size while a cycle-pinch is happening so it doesn't collapse
    if not (p["mid"] or p["ring"] or now < p["cool"]):
        p["r_target"] = float(np.clip(dist(pts[THUMB_TIP], pts[INDEX_TIP]) * SIZE_GAIN,
                                      MIN_R, max_r))

    a = 1.0 - math.exp(-POS_RATE * dt)
    p["pos"] += (pts[INDEX_TIP] - p["pos"]) * a
    b = 1.0 - math.exp(-SIZE_RATE * dt)
    p["r"] += (p["r_target"] - p["r"]) * b
    p["seen"] = now
    p["pts"] = pts
    return events


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Hand-tracked invisibility portal")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--no-mirror", action="store_true", help="don't mirror the camera image")
    return ap.parse_args()


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}. Try --camera 1.")

    tracker = HandTracker(max_hands=2)
    win = "Invisibility Portal"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    fullscreen = False

    effect_i, shape_i = 0, 0
    portals = {}
    bg, bg_buf = None, []
    phase, phase_start = "countdown", time.monotonic()
    toast, toast_t = "", 0.0
    show_landmarks = False
    fps, last, fails = 0.0, time.monotonic(), 0
    t0 = time.monotonic()

    while True:
        ok, frame = cap.read()
        if not ok:
            fails += 1
            if fails > 30:
                sys.exit("Camera stopped delivering frames.")
            continue
        fails = 0
        if not args.no_mirror:
            frame = cv2.flip(frame, 1)

        now = time.monotonic()
        dt = min(0.1, now - last)
        last = now
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-3))
        t = now - t0
        H, W = frame.shape[:2]
        max_r = MAX_R_FRAC * min(H, W)

        if phase == "countdown":
            disp = cv2.convertScaleAbs(frame, alpha=0.45)
            remaining = COUNTDOWN_S - (now - phase_start)
            put_centered(disp, "Step out of the frame", H // 2 - 60, 1.2)
            put_centered(disp, "Capturing background in", H // 2 - 10, 0.8)
            put_centered(disp, str(max(1, math.ceil(remaining))), H // 2 + 80, 3.0, (120, 220, 255), 5)
            if remaining <= 0:
                phase, bg_buf = "capture", []

        elif phase == "capture":
            bg_buf.append(frame.copy())
            disp = cv2.convertScaleAbs(frame, alpha=0.45)
            put_centered(disp, "Capturing background... hold still", H // 2, 1.0)
            if len(bg_buf) >= BG_FRAMES:
                bg = np.median(np.stack(bg_buf), axis=0).astype(np.uint8)
                bg_buf = []
                phase = "live"
                toast, toast_t = "Background captured - raise your hand!", now

        else:  # live
            if bg.shape != frame.shape:
                bg = cv2.resize(bg, (W, H))
            for label, pts in tracker.process(frame):
                if label not in portals:
                    portals[label] = new_portal(pts, max_r)
                for ev in update_portal(portals[label], pts, now, dt, max_r):
                    if ev == "effect":
                        effect_i = (effect_i + 1) % len(EFFECTS)
                        toast, toast_t = EFFECTS[effect_i], now
                    else:
                        shape_i = (shape_i + 1) % len(SHAPES)
                        toast, toast_t = SHAPES[shape_i], now
            portals = {k: p for k, p in portals.items() if now - p["seen"] <= LOST_GRACE}

            disp = frame.copy()
            effect, shape = EFFECTS[effect_i], SHAPES[shape_i]
            for p in portals.values():
                render_portal(frame, disp, bg, float(p["pos"][0]), float(p["pos"][1]),
                              p["r"], shape, effect, t)
                if show_landmarks:
                    draw_landmarks(disp, p["pts"])
            draw_hud(disp, effect, shape, fps, toast, now - toast_t)

        cv2.imshow(win, disp)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("b"):
            phase, phase_start, portals = "countdown", now, {}
        elif key == ord("e"):
            effect_i = (effect_i + 1) % len(EFFECTS)
            toast, toast_t = EFFECTS[effect_i], now
        elif key == ord("s"):
            shape_i = (shape_i + 1) % len(SHAPES)
            toast, toast_t = SHAPES[shape_i], now
        elif ord("1") <= key <= ord(str(len(EFFECTS))):
            effect_i = key - ord("1")
            toast, toast_t = EFFECTS[effect_i], now
        elif key == ord("l"):
            show_landmarks = not show_landmarks
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
