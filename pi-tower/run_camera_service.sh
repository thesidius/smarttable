#!/usr/bin/env bash
# Launch the camera service with the NoIR tuning file in the ENVIRONMENT.
#
# Why not just set it inside Python? camera_service.py does try, as a fallback,
# but it cannot be relied on: libcamera resolves the tuning path early -- early
# enough that an os.environ assignment made at the top of the script can still
# lose the race. Verified on this Pi: the in-script set left the running service
# on imx219.json while a plain `export` gets imx219_noir.json every time.
#
# This matters more than it looks. The wrong tuning does not fail, it just
# quietly costs ~0.14 of Otsu separability and 46 grey levels of numeral-to-body
# contrast on every frame the service ever captures.
set -euo pipefail

export LIBCAMERA_RPI_TUNING_FILE=/usr/share/libcamera/ipa/rpi/vc4/imx219_noir.json

if [[ ! -f "$LIBCAMERA_RPI_TUNING_FILE" ]]; then
    echo "FATAL: NoIR tuning file not found at $LIBCAMERA_RPI_TUNING_FILE" >&2
    echo "Refusing to start on the wrong colour pipeline." >&2
    exit 1
fi

cd "$(dirname "$0")"
exec python3 camera_service.py
