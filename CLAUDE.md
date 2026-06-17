# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Real-time image recognition (YOLOv8 / deep learning) on frames captured from an IMX219 CSI camera on a **NVIDIA Jetson Orin Nano** (JetPack 6.2, L4T 36.5, aarch64, Ubuntu 22.04). This project combines the CSI camera pipeline from `~/csi-test` with the YOLO inference from `~/python-camara`.

## Prerequisites

```bash
sudo systemctl start nvargus-daemon   # required before any CSI camera access
sudo nvpmodel -m 0                    # max power mode (optional, for best inference speed)
sudo jetson_clocks                    # unlock max clocks (optional)
```

## Running

```bash
source .venv/bin/activate

# Default (720p60, top-3 predictions)
python3 csi_classifier.py

# 1080p, show top-5
python3 csi_classifier.py --mode 1080p --top-k 5

# Reduce GPU load by inferring every 2nd frame
python3 csi_classifier.py --infer-every 2

# 180-degree flip (common for upside-down mount)
python3 csi_classifier.py --flip 2

# Unattended experiment over SSH: no display, fixed duration, CSV metrics log
python3 csi_classifier.py --headless --duration 3600 --log-file logs/run.csv
```

`--log-file` writes one CSV row per inferred frame (`timestamp_iso, epoch_s, frame_n, latency_ms, fps_instant, n_detections`), flushed immediately after each write. The on-screen HUD also shows live latency (`Inferencia: NN.N ms`) next to FPS and detection count, independent of `--log-file`.

To get this CSV into the Jetson's observability stack (Telegraf/InfluxDB/Grafana, see `~/observability/README.md`), point `--log-file` at a path inside `~/observability/logs/` — the `jetson-infer-metrics` systemd service watches that directory and pushes new rows straight to InfluxDB's HTTP API. (Telegraf's `inputs.tail` was tried first and abandoned: it never picked up live appends in the Telegraf 1.21.4 packaged for this Jetson.)

Monitor GPU load:
```bash
sudo tegrastats
```

## Virtual Environment

Use **`.venv/`** inside the project root (`csi-image-recognition/.venv/`):

| Package | Version | Note |
|---|---|---|
| torch | 2.8.0 | from Jetson PyPI |
| torchvision | 0.23.0 | from Jetson PyPI |
| opencv-python | 4.8.1.78 | |
| numpy | 1.26.4 | must be 1.x — see constraint below |
| ultralytics | 8.4.21 | YOLOv8 |

**PyTorch must be installed from the Jetson-specific index** (standard PyPI builds have no CUDA support on aarch64):

```bash
pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

**NumPy must stay at 1.x.** The in-venv `opencv-python` (4.8.1.78) was built against NumPy 1.x; upgrading to NumPy 2.x breaks it at runtime.

## GStreamer Pipeline Architecture

CSI cameras require the GStreamer backend — `cv2.VideoCapture(device, cv2.CAP_GSTREAMER)` — not the V4L2 backend.

```
nvarguscamerasrc (sensor ISP, NVMM memory, AE/AWB)
    → nvvidconv (zero-copy resize, format convert, flip)
    → video/x-raw, format=BGRx
    → videoconvert
    → video/x-raw, format=BGR
    → appsink (OpenCV frame)
```

`flip-method` values: 0=none, 1=ccw90, 2=180°, 3=cw90, 4=horizontal, 6=vertical.

## IMX219 Sensor Modes

| Mode | Resolution | FPS | sensor-id arg |
|---|---|---|---|
| full | 3280×2464 | 21 | 0 |
| wide | 3280×1848 | 28 | 0 |
| 1080p | 1920×1080 | 30 | 0 |
| 1232p | 1640×1232 | 30 | 0 |
| 720p60 | 1280×720 | 60 | 0 |

## YOLOv8 Inference Pattern

```python
from ultralytics import YOLO
model = YOLO("yolov8s.pt")           # small model — good balance on Jetson
results = model(frame, device=0)     # device=0 → CUDA GPU
```

Model weights `yolov8s.pt` are stored inside the project root.

## Common Failure Modes

- **"Failed to create capture"**: `nvargus-daemon` is not running. Start it and retry.
- **"CANCELLED" in argus logs**: Normal exit message from the ISP service; not an error.
- **NumPy ABI error at import**: NumPy was upgraded to 2.x. Downgrade: `pip install numpy==1.26.4`.
- **`torch.cuda.is_available()` returns False**: PyTorch was installed from standard PyPI instead of the Jetson index. Reinstall from `pypi.jetson-ai-lab.io`.

## Predecessor Projects

| Project | Path | What it demonstrates |
|---|---|---|
| csi-test | `~/csi-test/` | CSI camera pipeline, sensor modes, GStreamer patterns |
