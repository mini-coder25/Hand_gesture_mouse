# Gesture Cam – Touchless Mouse Control with Hand Gestures

Control your mouse cursor and clicks using hand gestures via a standard webcam. Powered by OpenCV, MediaPipe Hands, and PyAutoGUI.

## Features
- Move cursor with index fingertip
- Left-click with thumb–index pinch
- Right-click with thumb–middle pinch
- Toggle control on/off with an open palm
- Smoothing to reduce cursor jitter
- On-screen overlays (FPS, status, landmarks)

## Requirements
- Windows with a built-in or USB webcam
- Python 3.11 recommended (3.10 also works)

## Quick Start (PowerShell)

1) Create a virtual environment and install dependencies

```powershell
# From the repo root
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) Run the app

```powershell
python main.py --camera 0 --width 1280 --height 720 --fps 60
```

- Press Q or ESC to quit.
- Show an open palm for ~0.5s to toggle control.

## Options

```powershell
python main.py --camera 0 --width 1280 --height 720 --fps 60 --no-flip --alpha 0.6 --speed-gain 1.8 --show 1 --left-scale 0.33 --right-scale 0.33 --drag-hold 1
```

- `--camera`: Webcam index (default 0)
- `--width`/`--height`: Camera capture resolution (default 1280x720)
- `--fps`: Requested camera FPS (default 60; depends on webcam support)
- `--no-flip`: Disable horizontal mirroring (default is flipped for natural feel)
- `--alpha`: Smoothing factor [0–1], higher = faster (default 0.6)
- `--speed-gain`: Extra cursor speed multiplier (default 1.8)
- `--show`: 1 to show video window; 0 to hide (default 1)
- `--left-scale`: Left-click pinch threshold as fraction of palm width (default 0.33)
- `--right-scale`: Right-click pinch threshold as fraction of palm width (default 0.33)
- `--drag-hold`: Enable click-and-drag on thumb–index hold (default 0)

## Gestures
- Move: index fingertip guides the cursor
- Left-click: thumb tip + index tip pinch (distance < threshold)
- Right-click: thumb tip + middle tip pinch (distance < threshold)
- Toggle ON/OFF: open palm (4 fingers up) held briefly

Tip: If clicks don’t trigger, try increasing thresholds (e.g., `--left-scale 0.4 --right-scale 0.4`) or improve lighting.

## Tips
- Good lighting improves detection
- Keep your hand within the camera’s view
- Adjust thresholds (`--left-scale`, `--right-scale`) and smoothing (`--alpha`) to taste

## Troubleshooting
- Camera not opening: Close other apps using the webcam and try `--camera 1` or another index.
- Import errors (mediapipe/opencv): Ensure the venv is active (`.\.venv\Scripts\Activate.ps1`) and reinstall `pip install -r requirements.txt`.
- Cursor moves opposite: Use `--no-flip` to disable mirroring.
- Laggy preview: Lower capture size (`--width 640 --height 360`).

## Known Limitations
- Multi-monitor setups use the primary display bounds from PyAutoGUI
- Mediapipe performance depends on CPU/GPU; older hardware may need smaller camera resolution

## License
This project is provided for educational purposes.
