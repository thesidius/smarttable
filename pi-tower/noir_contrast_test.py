#!/usr/bin/env python3
"""
noir_contrast_test.py -- does dice numeral paint hold contrast under IR?

The camera is a NoIR module (no IR-cut filter). If the painted numerals stay
separable from the die body under IR-only illumination, we get lighting that is
invisible to players and completely immune to ambient room light. If they wash
out -- many pigments are effectively transparent in near-IR, so a black numeral
on a white die can vanish into the body -- we need visible light plus an IR-cut
filter instead.

This is a two-photo test. Same dice, same position, same framing; only the
light changes.

  # 1. room lights on, IR off
  python3 noir_contrast_test.py capture --label visible

  # 2. room lights OFF, IR illuminator on, dice untouched
  python3 noir_contrast_test.py capture --label ir

  # 3. compare -- ROI should tightly frame ONE die face in both
  python3 noir_contrast_test.py compare \
      ~/dicecam-captures/*_visible.jpg ~/dicecam-captures/*_ir.jpg \
      --roi 820,540,180,180

Capture always locks exposure/gain/white balance after letting AE converge, so
each shot is a fixed, reproducible measurement rather than whatever AE felt
like doing. Under IR the auto exposure will land somewhere very different from
visible light -- that is expected and fine; what matters is whether the
numerals separate from the body, not the absolute brightness.
"""

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime

try:
    import numpy as np
    import cv2
except ImportError:
    sys.exit("need numpy + opencv -> sudo apt install -y python3-numpy python3-opencv")


# ------------------------------------------------------------------ metrics ---

def metrics(gray):
    """Readability metrics for a patch that should contain a die face."""
    f = gray.astype(np.float32)
    p5, p95 = (float(x) for x in np.percentile(f, [5, 95]))
    mean, std = float(f.mean()), float(f.std())

    # Otsu picks the threshold that best splits the histogram into two classes.
    # The normalised between-class variance (eta) is the key number here: it is
    # how cleanly "numeral" separates from "die body" by brightness alone.
    # ~0 = one blob, no separation; ->1 = two clean, well-separated populations.
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lo, hi = f[f <= thr], f[f > thr]
    if lo.size and hi.size and f.var() > 0:
        w0, w1 = lo.size / f.size, hi.size / f.size
        eta = float(w0 * w1 * (lo.mean() - hi.mean()) ** 2 / f.var())
    else:
        eta = 0.0

    return {
        "mean": round(mean, 2),
        "std": round(std, 2),
        "rms_contrast": round(std / mean, 4) if mean > 0 else 0.0,
        # robust Michelson -- percentiles instead of min/max so one hot pixel
        # or one dead pixel cannot define the whole result
        "michelson": round((p95 - p5) / (p95 + p5), 4) if (p95 + p5) > 0 else 0.0,
        "otsu_threshold": round(float(thr), 1),
        "otsu_separability": round(eta, 4),
        "dark_class_mean": round(float(lo.mean()), 1) if lo.size else None,
        "light_class_mean": round(float(hi.mean()), 1) if hi.size else None,
        "class_gap": round(float(hi.mean() - lo.mean()), 1) if (lo.size and hi.size) else 0.0,
        "edge_energy": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
        "clipped_high_pct": round(100.0 * float((gray >= 250).sum()) / gray.size, 3),
        "clipped_low_pct": round(100.0 * float((gray <= 5).sum()) / gray.size, 3),
    }


def rate(m):
    """Judge one image's numeral readability."""
    eta, gap = m["otsu_separability"], m["class_gap"]
    if eta >= 0.55 and gap >= 60:
        return "GOOD", "numerals separate cleanly from the die body"
    if eta >= 0.35 and gap >= 35:
        return "MARGINAL", "separable but thin -- thresholding will be fragile"
    return "POOR", "numerals do not separate from the body by brightness"


# ------------------------------------------------------------------ capture ---

def do_capture(args):
    # Must be set before picamera2 initialises the camera manager. libcamera
    # cannot distinguish a NoIR module from a standard V2 (same sensor id), so
    # it defaults to the IR-cut tuning and produces a heavy magenta cast that
    # measurably degrades numeral separability. See camera_check.py for the
    # numbers. Override by exporting LIBCAMERA_RPI_TUNING_FILE yourself.
    noir_tuning = "/usr/share/libcamera/ipa/rpi/vc4/imx219_noir.json"
    if "LIBCAMERA_RPI_TUNING_FILE" not in os.environ and os.path.exists(noir_tuning):
        os.environ["LIBCAMERA_RPI_TUNING_FILE"] = noir_tuning

    try:
        from picamera2 import Picamera2
    except ImportError:
        sys.exit("picamera2 missing -> sudo apt install -y python3-picamera2")

    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(args.outdir, f"{stamp}_{args.label}.jpg")

    picam2 = Picamera2()
    cfg_main = {"format": "RGB888"}
    if args.size:
        w, h = (int(x) for x in args.size.lower().split("x"))
        cfg_main["size"] = (w, h)
    picam2.configure(picam2.create_still_configuration(main=cfg_main))
    picam2.start()
    try:
        print(f"[{args.label}] letting AE/AWB converge ({args.settle}s)...")
        time.sleep(args.settle)

        meta = picam2.capture_metadata()
        exp = int(meta.get("ExposureTime", 0))
        gain = float(meta.get("AnalogueGain", 1.0))
        cg = meta.get("ColourGains")

        controls = {"AeEnable": False, "AwbEnable": False,
                    "ExposureTime": exp, "AnalogueGain": gain}
        if cg:
            controls["ColourGains"] = (float(cg[0]), float(cg[1]))
        picam2.set_controls(controls)
        time.sleep(1.5)

        req = picam2.capture_request()
        try:
            arr = req.make_array("main")
            req.save("main", path)
            meta2 = req.get_metadata()
        finally:
            req.release()

        print(f"[{args.label}] locked: exposure={exp}us gain={round(gain, 3)}"
              + (f" colourgains=({round(float(cg[0]),3)},{round(float(cg[1]),3)})" if cg else ""))
        if meta2.get("Lux") is not None:
            print(f"[{args.label}] scene Lux (visible-weighted; unreliable under IR): "
                  f"{round(float(meta2['Lux']), 1)}")
        print(f"[{args.label}] saved -> {path}")

        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        full = metrics(gray)
        print(f"[{args.label}] whole-frame: mean {full['mean']} "
              f"clipped {full['clipped_high_pct']}% white / {full['clipped_low_pct']}% black")

        json.dump({"label": args.label, "path": path,
                   "locked": {"ExposureTime": exp, "AnalogueGain": gain,
                              "ColourGains": [float(cg[0]), float(cg[1])] if cg else None},
                   "full_frame": full},
                  open(path.replace(".jpg", ".json"), "w"), indent=2)

        print(f"\nNext: repeat under the other lighting WITHOUT moving the dice,"
              f"\nthen run:  python3 {os.path.basename(__file__)} compare <visible.jpg> <ir.jpg> --roi x,y,w,h")
    finally:
        picam2.stop()
        picam2.close()


# ------------------------------------------------------------------ compare ---

def do_compare(args):
    # expand globs ourselves -- PowerShell/scp workflows often pass them unexpanded
    paths = []
    for p in args.images:
        hits = sorted(glob.glob(os.path.expanduser(p)))
        paths.extend(hits if hits else [os.path.expanduser(p)])
    if len(paths) < 2:
        sys.exit("need at least two images to compare")

    roi = None
    if args.roi:
        x, y, w, h = (int(v) for v in args.roi.split(","))
        roi = (x, y, w, h)
    else:
        print("!! No --roi given: measuring the WHOLE frame. Background, tray and\n"
              "   shadows will dominate the statistics and the comparison will be\n"
              "   close to meaningless. Pass --roi x,y,w,h around ONE die face.\n")

    rows, crops = [], []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            sys.exit(f"could not read {p}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if roi:
            x, y, w, h = roi
            if y + h > gray.shape[0] or x + w > gray.shape[1]:
                sys.exit(f"ROI {roi} falls outside {p} ({gray.shape[1]}x{gray.shape[0]})")
            gray = gray[y:y + h, x:x + w]
        crops.append((os.path.basename(p), gray))
        m = metrics(gray)
        m["name"] = os.path.basename(p)
        rows.append(m)

    keys = ["mean", "std", "rms_contrast", "michelson",
            "otsu_separability", "class_gap", "edge_energy",
            "clipped_high_pct", "clipped_low_pct"]
    w0 = max(len(k) for k in keys) + 2
    print("=" * 68)
    print("NUMERAL READABILITY" + (f"  (ROI {roi})" if roi else "  (FULL FRAME)"))
    print("=" * 68)
    print(" " * w0 + "".join(f"{r['name'][:20]:>22}" for r in rows))
    for k in keys:
        print(f"{k:<{w0}}" + "".join(f"{r[k]:>22}" for r in rows))

    print("\n" + "-" * 68)
    best = None
    for r in rows:
        grade, why = rate(r)
        print(f"  {r['name']:<28} {grade:<10} {why}")
        if best is None or r["otsu_separability"] > best["otsu_separability"]:
            best = r
    print("-" * 68)
    print(f"\n  Strongest separation: {best['name']} "
          f"(otsu_separability {best['otsu_separability']}, class_gap {best['class_gap']})")
    print("\n  Read it this way:")
    print("    otsu_separability -- how cleanly numeral splits from body (the number that matters)")
    print("    class_gap         -- grey-level distance between those two populations")
    print("    If the IR shot grades POOR while visible grades GOOD, the paint is")
    print("    IR-transparent: go with visible light + an IR-cut filter.")
    print("    If IR grades GOOD or MARGINAL, IR illumination is viable -- and it")
    print("    buys invisible lighting plus immunity to ambient light.")

    if args.montage:
        h = min(c.shape[0] for _, c in crops)
        tiles = []
        for name, c in crops:
            scale = h / c.shape[0]
            t = cv2.resize(c, (int(c.shape[1] * scale), h))
            t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
            cv2.putText(t, name[:24], (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 255), 1, cv2.LINE_AA)
            tiles.append(t)
        cv2.imwrite(args.montage, np.hstack(tiles))
        print(f"\n  Side-by-side montage -> {args.montage}")


# --------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser(description="NoIR dice-numeral contrast test")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="capture one lighting condition (locked exposure)")
    c.add_argument("--label", required=True, help="e.g. visible / ir / ir-plus-visible")
    c.add_argument("--outdir", default=os.path.expanduser("~/dicecam-captures"))
    c.add_argument("--settle", type=float, default=3.0)
    c.add_argument("--size", default=None, help="WxH, default full sensor res")
    c.set_defaults(func=do_capture)

    m = sub.add_parser("compare", help="compare two or more captures")
    m.add_argument("images", nargs="+")
    m.add_argument("--roi", default=None,
                   help="x,y,w,h around ONE die face -- same pixels in every image")
    m.add_argument("--montage", default=None, help="write a side-by-side crop image here")
    m.set_defaults(func=do_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
