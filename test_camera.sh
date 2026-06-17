#!/bin/bash
echo "=== GStreamer Nvidia plugins ==="
gst-inspect-1.0 nvarguscamerasrc 2>&1 | head -3

echo ""
echo "=== Pipeline test (5 frames) ==="
gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=5 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! fakesink

echo ""
echo "=== OpenCV GStreamer support ==="
source "$(dirname "$0")/.venv/bin/activate"
python3 -c "import cv2; info=cv2.getBuildInformation(); lines=[l for l in info.split('\n') if 'GStreamer' in l]; print('\n'.join(lines))"
