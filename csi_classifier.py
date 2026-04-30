#!/usr/bin/env python3
"""
CSI Camera + YOLOv8 real-time object detection — Jetson Orin Nano.

Detects objects in each frame, draws bounding boxes with labels and confidence
scores, and overlays an exhibition HUD.

Usage:
    python3 csi_classifier.py
    python3 csi_classifier.py --mode 1080p --conf 0.4
    python3 csi_classifier.py --flip 2 --infer-every 2

Controls:
    q / ESC  — quit
    s        — save snapshot to snapshot_<timestamp>.jpg
"""

import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO

SENSOR_MODES = {
    "full":   (3280, 2464, 21),
    "wide":   (3280, 1848, 28),
    "1080p":  (1920, 1080, 30),
    "1232p":  (1640, 1232, 30),
    "720p60": (1280,  720, 60),
}

DISPLAY_WIDTH  = 1280
DISPLAY_HEIGHT = 720
MODEL_PATH     = "/home/opinzon/python-camara/yolov8s.pt"


# ---------------------------------------------------------------------------
# GStreamer pipeline
# ---------------------------------------------------------------------------

def gstreamer_pipeline(sensor_id=0, width=1280, height=720, fps=60,
                       display_width=1280, display_height=720, flip_method=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=NV12, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={display_width}, height={display_height}, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink max-buffers=1 drop=True"
    )


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _class_color(class_id: int) -> tuple:
    """Distinct BGR color per class using golden-angle HSV spacing."""
    hue = int((class_id * 137) % 180)
    hsv = np.array([[[hue, 200, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def _corner_box(frame, x1, y1, x2, y2, color, thickness=2):
    """L-shaped corner brackets — tech/AI aesthetic for exhibition."""
    L = max(10, min(24, (x2 - x1) // 5, (y2 - y1) // 5))
    corners = [
        ((x1, y1 + L), (x1, y1), (x1 + L, y1)),   # top-left
        ((x2 - L, y1), (x2, y1), (x2, y1 + L)),   # top-right
        ((x1, y2 - L), (x1, y2), (x1 + L, y2)),   # bottom-left
        ((x2 - L, y2), (x2, y2), (x2, y2 - L)),   # bottom-right
    ]
    for a, b, c in corners:
        cv2.line(frame, a, b, color, thickness, cv2.LINE_AA)
        cv2.line(frame, b, c, color, thickness, cv2.LINE_AA)


def _label_tag(frame, text: str, x: int, y: int, color):
    """Solid color tag above the top-left corner of a detection box."""
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    pad   = 5
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    bx1, by1 = x, max(0, y - th - pad * 2)
    bx2, by2 = x + tw + pad * 2, y
    # clamp so label stays on screen when box touches the top edge
    if by2 <= by1:
        by2 = by1 + th + pad * 2
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)
    cv2.putText(frame, text, (bx1 + pad, by2 - pad),
                font, scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_detections(frame, results, conf_threshold: float) -> int:
    """Render all detection boxes and return the count of drawn detections."""
    if not results:
        return 0

    boxes  = results[0].boxes
    names  = results[0].names

    # First pass: semi-transparent fill for all boxes at once
    overlay = frame.copy()
    kept = []
    for box in boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        cls_id       = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color        = _class_color(cls_id)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        kept.append((x1, y1, x2, y2, cls_id, conf))
    cv2.addWeighted(overlay, 0.13, frame, 0.87, 0, frame)

    # Second pass: corner brackets + label tags (drawn on top of fills)
    for x1, y1, x2, y2, cls_id, conf in kept:
        color = _class_color(cls_id)
        _corner_box(frame, x1, y1, x2, y2, color, thickness=2)
        label = f"{names[cls_id]}  {conf * 100:.0f}%"
        _label_tag(frame, label, x1, y1, color)

    return len(kept)


def draw_hud(frame, fps: float, n_detections: int):
    """Top header bar + bottom info strip for exhibition display."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0),        (w, 46),    (8, 8, 8), -1)
    cv2.rectangle(overlay, (0, h - 34),   (w, h),     (8, 8, 8), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # Cyan accent line under header
    cv2.line(frame, (0, 46), (w, 46), (0, 200, 255), 1)
    cv2.line(frame, (0, h - 34), (w, h - 34), (0, 200, 255), 1)

    cv2.putText(frame, "SISTEMA DE VISION INTELIGENTE - FISE UPBBGA 2026",
                (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (0, 220, 255), 2, cv2.LINE_AA)

    stats = f"FPS {fps:4.1f}   |   Objetos: {n_detections}"
    (tw, _), _ = cv2.getTextSize(stats, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(frame, stats, (w - tw - 14, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 230, 160), 1, cv2.LINE_AA)

    cv2.putText(frame, "YOLOv8s  |  NVIDIA Jetson Orin Nano  |  CUDA",
                (14, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (80, 180, 80), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CSI camera + YOLOv8 object detection — Jetson Orin Nano",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sensor-id",   type=int,   default=0)
    parser.add_argument("--mode",        choices=SENSOR_MODES.keys(), default="720p60")
    parser.add_argument("--flip",        type=int,   default=0,
                        help="nvvidconv flip-method: 0=none, 2=180°, 4=horiz, 6=vert")
    parser.add_argument("--conf",        type=float, default=0.35,
                        help="Minimum detection confidence (default: 0.35)")
    parser.add_argument("--infer-every", type=int,   default=1,
                        help="Run YOLO every N frames — increase if GPU is bottleneck")
    args = parser.parse_args()

    cap_w, cap_h, cap_fps = SENSOR_MODES[args.mode]
    scale  = min(DISPLAY_WIDTH / cap_w, DISPLAY_HEIGHT / cap_h)
    disp_w = int(cap_w * scale)
    disp_h = int(cap_h * scale)

    pipeline = gstreamer_pipeline(
        sensor_id=args.sensor_id,
        width=cap_w, height=cap_h, fps=cap_fps,
        display_width=disp_w, display_height=disp_h,
        flip_method=args.flip,
    )
    print(f"Pipeline:\n  {pipeline}\n")

    print("Loading YOLOv8s …")
    model = YOLO(MODEL_PATH)
    model.to("cuda")
    print("YOLOv8s loaded on CUDA\n")

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("\nERROR: Could not open CSI camera. Check:")
        print("  • nvargus-daemon running?  →  sudo systemctl start nvargus-daemon")
        print("  • Camera connected?        →  ls /dev/video*")
        return 1

    print(f"Running — {disp_w}x{disp_h} display | capture {cap_w}x{cap_h}@{cap_fps}fps")
    print(f"Confidence: {args.conf} | infer every {args.infer_every} frame(s)")
    print("Press  q / ESC  to quit   |   s  to save snapshot\n")

    results     = None
    n_det       = 0
    frame_n     = 0
    fps_counter = 0
    fps_start   = time.time()
    fps_display = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Frame grab failed — camera disconnected?")
            break

        frame_n     += 1
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_start   = time.time()

        if frame_n % args.infer_every == 0:
            results = model(frame, device=0, verbose=False)

        n_det = draw_detections(frame, results, args.conf)
        draw_hud(frame, fps_display, n_det)

        cv2.imshow("Vision Inteligente — Jetson Orin Nano", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            ts    = time.strftime("%Y%m%d_%H%M%S")
            fname = f"snapshot_{ts}.jpg"
            cv2.imwrite(fname, frame)
            print(f"Snapshot: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
