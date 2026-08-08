#!/usr/bin/env bash
# Launch the camera service.
#
# This script used to export LIBCAMERA_RPI_TUNING_FILE, on the finding that an
# export before launch beat libcamera's early path resolution where an
# os.environ assignment inside Python did not. That finding is now obsolete and
# the export has been REMOVED, because it was not merely unnecessary -- it was
# actively misleading.
#
# picamera2 pops the variable in its own constructor when it is called without
# a tuning= argument:
#
#     os.environ.pop("LIBCAMERA_RPI_TUNING_FILE", None)  # Use default tuning
#     -- picamera2.py:337, v0.7.1+rpt20260609
#
# So the export was being discarded before libcamera ever read it, while
# everything that checked "is the NoIR tuning set?" saw the variable we had set
# and reported success. camera_service.py now passes the tuning file through
# Picamera2(tuning=...) instead, which is the supported API, and reads it back
# afterwards so /health reports the outcome rather than the intention.
#
# The service is normally run by systemd, not by hand:
#
#     sudo systemctl status dicecam
#     journalctl -u dicecam -f
#
# This script remains the ExecStart target and the way to run it in a terminal.
set -euo pipefail

TUNING=/usr/share/libcamera/ipa/rpi/vc4/imx219_noir.json
if [[ ! -f "$TUNING" ]]; then
    echo "WARNING: NoIR tuning file not found at $TUNING" >&2
    echo "The service will start on the IR-cut tuning and say so in /health." >&2
fi

cd "$(dirname "$0")"
exec python3 camera_service.py
