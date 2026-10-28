# Invisibility Portal

Real-time hand-tracked portal with live filters, built on MediaPipe + OpenCV.
Each hand opens a portal on its index fingertip; inside it you see one of seven
effects: Invisibility (background reveal), Swirl, Thermal, Pixelate, Neon Edges,
Mirror, Grayscale. Shapes: circle, hexagon, star.

## Demo

<video src="https://github.com/anisha274/invisibility-portal/raw/main/assets/demo.mp4" controls muted width="100%"></video>

If the player doesn't load, [watch the demo video](assets/demo.mp4).

## Setup

Needs Python 3.9–3.12 and a webcam.

```bash
python -m venv venv
# Windows: venv\Scripts\activate     macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python invisibility_portal.py
```

On first run with newer MediaPipe versions, the script downloads the hand model
(`hand_landmarker.task`, ~8 MB) next to itself. That needs internet once.

## Using it

1. When the app starts, **step out of the frame** for 3 seconds. It captures the
   empty background — this is what the Invisibility effect reveals.
2. Raise a hand. A portal appears on your index fingertip.

| Gesture | Action |
|---|---|
| Spread / close thumb and index | Grow / shrink the portal |
| Pinch thumb + middle fingertip | Next effect |
| Pinch thumb + ring fingertip | Next shape |

Keyboard extras: `b` re-capture background · `e`/`s` next effect/shape ·
`1`–`7` pick effect · `l` show landmarks · `f` fullscreen · `q`/`Esc` quit.

Options: `--camera 1` (other webcam), `--width 1920 --height 1080`, `--no-mirror`.

## Tips for a convincing invisibility effect

- Keep the camera perfectly still (a tripod or propped-up laptop).
- Even, steady lighting. Auto-exposure shifts make the revealed background look
  brighter/darker than the live frame — press `b` to re-capture if that happens.
- A background with some texture (bookshelf, poster) sells the effect better than
  a plain wall.

## Tuning

All thresholds are at the top of the script: `PINCH_ON`/`PINCH_OFF` (pinch
sensitivity), `POS_RATE`/`SIZE_RATE` (smoothing), `MIN_R`/`MAX_R_FRAC` (portal
size range), `GESTURE_COOLDOWN`.
