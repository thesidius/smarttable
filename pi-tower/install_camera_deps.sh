#!/usr/bin/env bash
# Camera dependencies for the dice-cam tower (Raspberry Pi OS Bookworm, 64-bit).
#
# apt, not pip: picamera2 binds to the system libcamera build. A pip install
# pulls a copy that does not match the system libraries and fails at import.
# On Bookworm, pip into the system environment is blocked (PEP 668) anyway.
set -euo pipefail

echo "== apt update =="
sudo apt update

echo "== installing camera stack =="
sudo apt install -y \
    python3-picamera2 \
    python3-opencv \
    python3-numpy \
    python3-pip

echo
echo "== versions =="
# picamera2 exposes no __version__ attribute (checked on 0.3.36), so ask dpkg.
dpkg-query -W -f='${Package} ${Version}\n' \
    python3-picamera2 python3-libcamera python3-opencv python3-numpy rpicam-apps
python3 -c "import picamera2, cv2, numpy, libcamera; print('imports OK')"

echo
echo "== camera detection =="
rpicam-hello --list-cameras || {
    echo
    echo "No camera detected. Check, in order:"
    echo "  1. Ribbon is in the CAMERA port -- on the Pi 3 B that is the port"
    echo "     nearer the HDMI connector. The display port is the other one and"
    echo "     they look identical."
    echo "  2. Ribbon orientation: blue stiffener faces the ethernet/USB side of"
    echo "     the Pi; bare contacts face the HDMI side. Latch fully seated."
    echo "  3. Both ends -- the camera-module end pops loose easily."
    echo "  4. grep camera_auto_detect /boot/firmware/config.txt   (want =1)"
    echo "  5. Power the Pi down fully before reseating. Hot-plugging CSI can"
    echo "     damage the module."
    exit 1
}

echo
echo "Camera stack ready. Next: python3 camera_check.py"
