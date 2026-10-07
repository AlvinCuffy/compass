#!/usr/bin/env bash
# Install everything the photo->talking-video renderer needs.
# Safe to re-run. Uses --break-system-packages because this is a disposable
# sandbox rather than a managed Python environment.
set -euo pipefail
cd "$(dirname "$0")"

pip3 install --quiet --break-system-packages -r requirements.txt

# mediapipe drags in the non-headless opencv build, which needs libGL that this
# image does not have; make sure the headless one is the one on disk.
pip3 uninstall -y -q opencv-python opencv-contrib-python 2>/dev/null || true
pip3 install --quiet --break-system-packages --force-reinstall --no-cache-dir \
  opencv-contrib-python-headless

python3 - <<'PY'
import cv2, mediapipe, numpy, scipy, imageio_ffmpeg
print("numpy      ", numpy.__version__)
print("opencv     ", cv2.__version__)
print("mediapipe  ", mediapipe.__version__)
print("ffmpeg     ", imageio_ffmpeg.get_ffmpeg_exe())
PY
echo "ready."
