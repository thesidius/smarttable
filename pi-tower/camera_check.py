#!/usr/bin/env python3
"""
camera_check.py -- first-light verification & characterization for the dice-cam tower.

What it does, in order:
  1. Enumerates cameras; reports sensor model, tuning, and every sensor mode.
  2. Auto-exposure/auto-white-balance capture; reports whether the frame is
     actually well exposed (not just "it returned an image").
  3. Reads back the values AE/AWB converged on, FREEZES them, captures again.

Why the locked capture matters: auto-exposure hunts. Frame-to-frame brightness
drift will (a) make dice-face reading inconsistent between captures and (b)
generate false positives in frame-differencing motion detection later. The
locked values printed here are the ones to bake into the capture config.

Usage:
    python3 camera_check.py
    python3 camera_check.py --tag ir-flood --settle 4
    python3 camera_check.py --size 1640x1232 --rot 90

Outputs to --outdir (default ~/dicecam-captures/):
    <stamp>_<tag>_auto.jpg    + _auto.json
    <stamp>_<tag>_locked.jpg  + _locked.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

try:
    import numpy as np
except ImportError:
    sys.exit("numpy missing -> sudo apt install -y python3-numpy")

try:
    import cv2
except ImportError:
    sys.exit("opencv missing -> sudo apt install -y python3-opencv")

# MUST be set before picamera2 initialises the camera manager.
#
# The NoIR module is electrically identical to the standard V2 and reports the
# same sensor id, so libcamera cannot tell them apart and defaults to
# imx219.json -- the tuning for the IR-cut-filtered version. On a NoIR module
# that tuning's AWB pushes red gain UP (measured 1.967) when the IR load means
# it should come DOWN (0.822 under the correct tuning), giving the notorious
# magenta cast. That is not merely cosmetic: on a die-face ROI it cost ~0.14 of
# Otsu separability and 46 grey levels of numeral/body gap.
#
# Override by setting LIBCAMERA_RPI_TUNING_FILE yourself before running.
NOIR_TUNING = "/usr/share/libcamera/ipa/rpi/vc4/imx219_noir.json"
if "LIBCAMERA_RPI_TUNING_FILE" not in os.environ and os.path.exists(NOIR_TUNING):
    os.environ["LIBCAMERA_RPI_TUNING_FILE"] = NOIR_TUNING

try:
    from picamera2 import Picamera2
except ImportError:
    sys.exit(
        "picamera2 missing -> sudo apt install -y python3-picamera2\n"
        "(install via apt, NOT pip -- it needs the system libcamera bindings)"
    )


# ---------------------------------------------------------------- exposure ---

def analyze(bgr):
    """Grayscale exposure/detail statistics for a captured frame."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    f = gray.astype(np.float32)
    total = f.size

    p1, p5, p50, p95, p99 = (float(x) for x in np.percentile(f, [1, 5, 50, 95, 99]))
    mean, std = float(f.mean()), float(f.std())

    # Focus proxy. A whole-frame Laplacian variance is worthless here: this rig
    # points at a mostly-empty tray, so flat regions dominate and drag the
    # number to near zero even when the lens is tack sharp (first-light frame
    # scored 28 frame-wide but 135 on an in-focus die face). So tile the frame
    # and keep the sharpest tile -- if anything in the scene resolves crisply,
    # the lens is focused at that depth.
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    h, w = gray.shape
    th, tw = h // 6, w // 6
    tiles = sorted(float(lap[r:r + th, c:c + tw].var())
                   for r in range(0, 6 * th, th)
                   for c in range(0, 6 * tw, tw))

    return {
        "laplacian_var_best_tile": round(tiles[-1], 1),
        "laplacian_var_p90_tile": round(tiles[int(0.9 * (len(tiles) - 1))], 1),
        "mean": round(mean, 2),
        "std": round(std, 2),
        "p1": round(p1, 1), "p5": round(p5, 1), "median": round(p50, 1),
        "p95": round(p95, 1), "p99": round(p99, 1),
        # clipping: pixels pinned at the ends of the range carry no information
        "clipped_high_pct": round(100.0 * float((gray >= 250).sum()) / total, 3),
        "clipped_low_pct": round(100.0 * float((gray <= 5).sum()) / total, 3),
        # dynamic range actually in use
        "rms_contrast": round(std / mean, 4) if mean > 0 else 0.0,
        # frame-wide detail energy -- kept for reference, but see the tiled
        # numbers above; on a sparse scene this one under-reports badly
        "laplacian_var": round(float(lap.var()), 1),
    }


def verdict(st):
    """Turn the stats into a plain judgement + concrete next action."""
    problems, notes = [], []

    if st["clipped_high_pct"] > 1.0:
        problems.append(
            f"OVEREXPOSED: {st['clipped_high_pct']}% of pixels blown to white "
            "(detail there is unrecoverable)"
        )
    if st["clipped_low_pct"] > 5.0:
        problems.append(
            f"UNDEREXPOSED: {st['clipped_low_pct']}% of pixels crushed to black"
        )
    if st["p99"] < 90:
        problems.append(f"VERY DARK: even the 99th percentile is only {st['p99']}/255")
    if st["mean"] < 40:
        problems.append(f"DARK: mean brightness {st['mean']}/255 (want ~90-160)")
    elif st["mean"] > 200:
        problems.append(f"BRIGHT: mean brightness {st['mean']}/255 (want ~90-160)")

    if st["rms_contrast"] < 0.15:
        notes.append(
            f"Low global contrast (RMS {st['rms_contrast']}) -- flat/washed-out scene"
        )
    if st["laplacian_var_best_tile"] < 60:
        notes.append(
            f"Nothing in frame resolves sharply (best tile Laplacian var "
            f"{st['laplacian_var_best_tile']}). Either the lens is off focus, or "
            "there is genuinely no fine detail in the scene -- put something "
            "textured in frame before concluding. The V2 module's lens is "
            "manual-focus and ships set to ~infinity; close overhead work needs "
            "it turned in."
        )

    return ("WELL EXPOSED" if not problems else "NEEDS ATTENTION"), problems, notes


# ------------------------------------------------------------------ camera ---

def report_camera(picam2):
    print("=" * 68)
    print("SENSOR")
    print("=" * 68)

    props = picam2.camera_properties
    model = props.get("Model", "<unknown>")
    print(f"  Model           : {model}")
    print(f"  Pixel array     : {props.get('PixelArraySize')}")
    print(f"  Unit cell (nm)  : {props.get('UnitCellSize')}")
    print(f"  Rotation        : {props.get('Rotation')}")
    print(f"  Colour filter   : {props.get('ColorFilterArrangement')}")

    if "imx219" not in str(model).lower():
        print(f"\n  !! Expected imx219 (Camera Module V2 NoIR), got '{model}'.")

    print("\n  Sensor modes (native readout, before ISP scaling):")
    print(f"    {'#':<3}{'size':<14}{'format':<14}{'bit':<6}{'max fps':<10}")
    for i, m in enumerate(picam2.sensor_modes):
        print(
            f"    {i:<3}{str(m['size']):<14}{str(m.get('format', '')):<14}"
            f"{str(m.get('bit_depth', '')):<6}{round(m.get('fps', 0), 1):<10}"
        )

    ctrl = picam2.camera_controls
    for name in ("ExposureTime", "AnalogueGain", "FrameDurationLimits"):
        if name in ctrl:
            lo, hi, dflt = ctrl[name]
            print(f"\n  {name}: min={lo} max={hi} default={dflt}")
    return model


def grab(picam2, path, rot):
    """One request -> image file, numpy array, and metadata all from the same frame."""
    req = picam2.capture_request()
    try:
        arr = req.make_array("main")          # RGB888 config => BGR byte order (cv2-native)
        meta = req.get_metadata()
        if rot:
            code = {90: cv2.ROTATE_90_CLOCKWISE,
                    180: cv2.ROTATE_180,
                    270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rot]
            cv2.imwrite(path, cv2.rotate(arr, code))
        else:
            req.save("main", path)
    finally:
        req.release()
    return arr, meta


def show_meta(meta):
    cg = meta.get("ColourGains")
    print(f"    ExposureTime : {meta.get('ExposureTime')} us")
    print(f"    AnalogueGain : {round(float(meta.get('AnalogueGain', 0)), 3)}")
    print(f"    DigitalGain  : {round(float(meta.get('DigitalGain', 0)), 3)}")
    if cg:
        print(f"    ColourGains  : R={round(float(cg[0]), 3)} B={round(float(cg[1]), 3)}")
    if meta.get("Lux") is not None:
        print(f"    Lux          : {round(float(meta['Lux']), 1)}")


def print_stats(st, label):
    v, problems, notes = verdict(st)
    print(f"    mean {st['mean']}  median {st['median']}  std {st['std']}  "
          f"p1 {st['p1']}  p99 {st['p99']}")
    print(f"    clipped: {st['clipped_high_pct']}% white / {st['clipped_low_pct']}% black")
    print(f"    RMS contrast {st['rms_contrast']}   detail: sharpest tile "
          f"{st['laplacian_var_best_tile']} / p90 tile {st['laplacian_var_p90_tile']} "
          f"/ whole frame {st['laplacian_var']}")
    print(f"    -> {label}: {v}")
    for p in problems:
        print(f"       * {p}")
    for n in notes:
        print(f"       - {n}")
    return v


# -------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser(description="Dice-cam first-light check")
    ap.add_argument("--outdir", default=os.path.expanduser("~/dicecam-captures"))
    ap.add_argument("--tag", default="check", help="label folded into filenames")
    ap.add_argument("--settle", type=float, default=3.0,
                    help="seconds to let AE/AWB converge before the auto capture")
    ap.add_argument("--size", default=None,
                    help="capture WxH, e.g. 1640x1232 (default: full sensor res)")
    ap.add_argument("--rot", type=int, default=0, choices=[0, 90, 180, 270],
                    help="rotate SAVED image only; does not change FOV")
    ap.add_argument("--no-lock", action="store_true",
                    help="skip the locked capture (auto only)")
    ap.add_argument("--pin-framerate", action="store_true",
                    help="also pin FrameDurationLimits to the locked exposure")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = os.path.join(args.outdir, f"{stamp}_{args.tag}")

    cams = Picamera2.global_camera_info()
    if not cams:
        sys.exit(
            "No cameras detected.\n"
            "  - Check the ribbon is in the CAMERA port (the one nearer the HDMI\n"
            "    end of the Pi 3 B), contacts facing the correct way, latch down.\n"
            "  - Cross-check with:  rpicam-hello --list-cameras\n"
            "  - If still nothing, confirm camera_auto_detect=1 in\n"
            "    /boot/firmware/config.txt and reboot."
        )
    print(f"Cameras found: {len(cams)}")
    for c in cams:
        print(f"  [{c.get('Num')}] {c.get('Model')}  @ {c.get('Id')}")

    picam2 = Picamera2()
    model = report_camera(picam2)

    cfg_main = {"format": "RGB888"}
    if args.size:
        w, h = (int(x) for x in args.size.lower().split("x"))
        cfg_main["size"] = (w, h)
    picam2.configure(picam2.create_still_configuration(main=cfg_main))

    picam2.start()
    try:
        # ---- pass 1: auto ------------------------------------------------
        print("\n" + "=" * 68)
        print(f"AUTO CAPTURE  (settling {args.settle}s for AE/AWB)")
        print("=" * 68)
        time.sleep(args.settle)

        auto_path = f"{base}_auto.jpg"
        arr, meta = grab(picam2, auto_path, args.rot)
        print(f"  Frame: {arr.shape[1]}x{arr.shape[0]}  ->  {auto_path}")
        print("  Converged control values:")
        show_meta(meta)
        st_auto = analyze(arr)
        print("  Exposure analysis:")
        v_auto = print_stats(st_auto, "AUTO")

        exp = int(meta.get("ExposureTime", 0))
        gain = float(meta.get("AnalogueGain", 1.0))
        cg = meta.get("ColourGains")
        locked_controls = {
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": exp,
            "AnalogueGain": gain,
        }
        if cg:
            locked_controls["ColourGains"] = (float(cg[0]), float(cg[1]))

        json.dump({"mode": "auto", "model": model, "metadata_subset": {
            "ExposureTime": exp, "AnalogueGain": gain,
            "ColourGains": [float(cg[0]), float(cg[1])] if cg else None,
            "Lux": meta.get("Lux")},
            "stats": st_auto, "verdict": v_auto},
            open(f"{base}_auto.json", "w"), indent=2)

        if args.no_lock:
            return

        # ---- pass 2: locked ----------------------------------------------
        print("\n" + "=" * 68)
        print("LOCKED CAPTURE  (AE/AWB frozen at the values above)")
        print("=" * 68)

        if args.pin_framerate:
            # Exposure cannot exceed the frame duration; pin both so the
            # pipeline can't silently shorten the exposure.
            fd = max(exp + 2000, 33333)
            try:
                picam2.set_controls({"FrameDurationLimits": (fd, fd)})
                print(f"  FrameDurationLimits pinned to {fd} us")
            except Exception as e:
                print(f"  (could not pin FrameDurationLimits: {e})")

        picam2.set_controls(locked_controls)
        time.sleep(1.5)  # let the frozen values propagate through the pipeline

        locked_path = f"{base}_locked.jpg"
        arr2, meta2 = grab(picam2, locked_path, args.rot)
        print(f"  Frame: {arr2.shape[1]}x{arr2.shape[0]}  ->  {locked_path}")
        print("  Read-back (should match the requested lock):")
        show_meta(meta2)

        drift = abs(int(meta2.get("ExposureTime", 0)) - exp)
        if drift > max(50, exp * 0.02):
            print(f"    !! exposure drifted {drift} us from the requested lock "
                  "-- AE may not have fully disengaged; try --pin-framerate")
        else:
            print("    lock holding (exposure stable)")

        st_lock = analyze(arr2)
        print("  Exposure analysis:")
        v_lock = print_stats(st_lock, "LOCKED")

        json.dump({"mode": "locked", "model": model,
                   "locked_controls": {k: (list(v) if isinstance(v, tuple) else v)
                                       for k, v in locked_controls.items()},
                   "stats": st_lock, "verdict": v_lock},
                  open(f"{base}_locked.json", "w"), indent=2)

        # ---- the payload: numbers to bake into the real capture config ----
        print("\n" + "=" * 68)
        print("BAKE THESE INTO THE CAPTURE CONFIG")
        print("=" * 68)
        print("  picam2.set_controls({")
        print('      "AeEnable": False,')
        print('      "AwbEnable": False,')
        print(f'      "ExposureTime": {exp},')
        print(f'      "AnalogueGain": {round(gain, 3)},')
        if cg:
            print(f'      "ColourGains": ({round(float(cg[0]), 3)}, {round(float(cg[1]), 3)}),')
        print("  })")
        print(f"\n  Saved: {base}_auto.jpg / _locked.jpg (+ .json sidecars)")

    finally:
        picam2.stop()
        picam2.close()


if __name__ == "__main__":
    main()
