#!/usr/bin/env python3
"""
crop_pipeline.py -- turn tray frames into canonical die crops + a manifest.

This is the shared foundation under all three stage-2 approaches (whole-crop
classifier, geometric top-face, vision model). They must consume byte-identical
crops or the comparison between them measures the crop pipeline instead of the
approaches. So the crop is defined once, here, and everything downstream reads
the manifest.

Canonical crop definition -- deliberate choices, not defaults:

  SQUARE, centred on the detection.
      Dice are roughly square in bbox. Forcing square before resize means no
      approach ever sees an aspect-distorted die.

  FIXED CONTEXT RATIO (--context, default 1.5x the larger bbox side).
      Normalises scale. A d20 and a d6 arrive the same size in the output, so a
      classifier cannot cheat by learning "big blob = d20" -- which would stop
      working the moment the camera height changes.

  LOSSLESS PNG.
      JPEG ringing around high-contrast numerals is exactly the signal we care
      about. Training data must not have compression artefacts baked in.

  NO ROTATION NORMALISATION.
      A die's orientation in the tray is genuinely random and the classifier
      has to cope with it. Straightening crops here would hide that problem
      until deployment.

Every crop records its provenance, and CROP_VERSION is stamped into the
manifest: bump it whenever the crop geometry changes, because that invalidates
every label collected against the old version.

Usage:
    python3 crop_pipeline.py --glob '~/dicecam-captures/*_locked.jpg' --out ~/dice-data
    python3 crop_pipeline.py --image foo.jpg --out ~/dice-data --contact-sheet
"""

import argparse
import glob as globmod
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dice_detect import (tray_mask, segment, detect, flag,   # noqa: E402
                         fit_quad_to_frame, quad_scale, reference_geometry)

CROP_VERSION = 1
CONFIG = os.path.expanduser("~/.config/dicecam/tray.json")


def canonical_crop(img, bbox, context, out_size):
    """Square, scale-normalised, fixed-size crop centred on the detection."""
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * context

    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = int(round(cx + side / 2)), int(round(cy + side / 2))

    # A die resting against a tray wall wants a crop that runs off the frame.
    # Replicate the edge rather than filling black: a hard black border is a
    # strong artificial edge that a classifier will happily latch onto.
    pl, pt = max(0, -x0), max(0, -y0)
    pr, pb = max(0, x1 - img.shape[1]), max(0, y1 - img.shape[0])
    padded = bool(pl or pt or pr or pb)

    sub = img[max(0, y0):min(img.shape[0], y1), max(0, x0):min(img.shape[1], x1)]
    if sub.size == 0:
        return None, None
    if padded:
        sub = cv2.copyMakeBorder(sub, pt, pb, pl, pr, cv2.BORDER_REPLICATE)

    # INTER_AREA is the correct downscale filter -- it averages over the source
    # footprint instead of point-sampling, so glitter speckle does not alias
    # into fake structure.
    interp = cv2.INTER_AREA if sub.shape[0] > out_size else cv2.INTER_CUBIC
    return cv2.resize(sub, (out_size, out_size), interpolation=interp), {
        "crop_box": [x0, y0, x1, y1],
        "edge_padded": padded,
        "source_px": int(sub.shape[0]),
    }


def process(path, quad, quad_frame, args, records):
    img = cv2.imread(path)
    if img is None:
        print(f"  !! unreadable: {path}")
        return 0, 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    try:
        quad, note = fit_quad_to_frame(quad, quad_frame, gray.shape)
    except ValueError as e:
        print(f"  !! SKIPPED {os.path.basename(path)}: {e}")
        return 0, 0
    if note:
        print(f"     ({os.path.basename(path)}: quad {note})")

    scale = quad_scale(quad)
    mask = tray_mask(gray.shape, quad, scale)
    tray_area = float(np.count_nonzero(mask))
    if tray_area == 0:
        print(f"  !! empty tray mask on {os.path.basename(path)} -- wrong calibration?")
        return 0, 0

    binary, _, _ = segment(gray, mask, args.window, scale)
    dets, _ = detect(binary, tray_area, args.min_frac, args.max_frac)
    ref = reference_geometry(dets)

    # Capture provenance. Crops taken through different camera pipelines are
    # not interchangeable training data -- the same die under imx219.json vs
    # imx219_noir.json differs by a factor of 6 in R/G. Record what we can so a
    # mixed dataset is detectable later instead of quietly poisoning a model.
    sidecar_path = os.path.splitext(path)[0] + ".json"
    provenance = {}
    if os.path.exists(sidecar_path):
        try:
            sc = json.load(open(sidecar_path))
            provenance = {"capture": sc.get("locked_controls") or
                          sc.get("locked") or sc.get("metadata_subset")}
        except (ValueError, OSError):
            pass

    stem = os.path.splitext(os.path.basename(path))[0]
    written = suspect = 0
    for d in dets:
        reasons = flag(d, ref)
        crop, meta = canonical_crop(img, d["bbox"], args.context, args.size)
        if crop is None:
            continue
        name = f"{stem}_d{d['id']:02d}.png"
        cv2.imwrite(os.path.join(args.out, "crops", name), crop,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])
        records.append({
            "crop": name,
            "crop_version": CROP_VERSION,
            "source": os.path.abspath(path),
            "bbox": d["bbox"],
            "centroid": d["centroid"],
            "area_px": d["area_px"],
            "solidity": d["solidity"],
            "aspect": d["aspect"],
            "suspect": bool(reasons),
            "suspect_reasons": reasons,
            "out_size": args.size,
            "context": args.context,
            **provenance,
            **meta,
            "label": None,        # filled in by labelling, never by this script
        })
        written += 1
        suspect += bool(reasons)
    return written, suspect


def contact_sheet(out_dir, records, cols=8, thumb=128):
    """One image showing every crop, for eyeballing a batch fast."""
    if not records:
        return None
    rows = (len(records) + cols - 1) // cols
    sheet = np.full((rows * (thumb + 18), cols * thumb, 3), 30, np.uint8)
    for i, r in enumerate(records):
        c = cv2.imread(os.path.join(out_dir, "crops", r["crop"]))
        if c is None:
            continue
        c = cv2.resize(c, (thumb, thumb), interpolation=cv2.INTER_AREA)
        rr, cc = divmod(i, cols)
        y, x = rr * (thumb + 18), cc * thumb
        sheet[y:y + thumb, x:x + thumb] = c
        col = (0, 0, 255) if r["suspect"] else (0, 220, 0)
        cv2.putText(sheet, f"{i}", (x + 3, y + thumb + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
    p = os.path.join(out_dir, "contact_sheet.png")
    cv2.imwrite(p, sheet)
    return p


def main():
    ap = argparse.ArgumentParser(description="Canonical die-crop pipeline")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image")
    g.add_argument("--glob", help="quoted glob, e.g. '~/dicecam-captures/*.jpg'")
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--quad", help="tray corners; else saved calibration")
    ap.add_argument("--quad-frame", help="resolution --quad was measured at, e.g. 1640x1232")
    ap.add_argument("--size", type=int, default=128, help="output crop size (px)")
    ap.add_argument("--context", type=float, default=1.5,
                    help="crop side as a multiple of the larger bbox side")
    ap.add_argument("--window", type=int, default=None,
                    help="local-variance window in px; default scales with tray size")
    ap.add_argument("--min-frac", type=float, default=0.002)
    ap.add_argument("--max-frac", type=float, default=0.45)
    ap.add_argument("--contact-sheet", action="store_true")
    args = ap.parse_args()

    paths_probe = ([os.path.expanduser(args.image)] if args.image
                   else sorted(globmod.glob(os.path.expanduser(args.glob))))

    if args.quad:
        quad = np.array([int(v) for v in args.quad.split(",")], np.float32).reshape(4, 2)
        if args.quad_frame:
            quad_frame = [int(v) for v in args.quad_frame.lower().split("x")]
        else:
            # A bare --quad has no resolution attached, so assume it was read off
            # the first input image. Wrong assumptions here are silent, so say so.
            probe = cv2.imread(paths_probe[0]) if paths_probe else None
            if probe is None:
                sys.exit("--quad given but the first input image is unreadable")
            quad_frame = [probe.shape[1], probe.shape[0]]
            print(f"note: --quad assumed to be in {quad_frame[0]}x{quad_frame[1]} "
                  f"coords (from {os.path.basename(paths_probe[0])}). "
                  f"Use --quad-frame WxH to say otherwise.")
        qsrc = "--quad"
    elif os.path.exists(CONFIG):
        cfg = json.load(open(CONFIG))
        quad = np.array(cfg["quad"], np.float32).reshape(4, 2)
        quad_frame = cfg.get("frame") or [0, 0]
        if not quad_frame[0]:
            sys.exit(f"{CONFIG} has no 'frame' field -- recalibrate so the quad's "
                     f"resolution is recorded")
        qsrc = f"{CONFIG} (calibrated at {quad_frame[0]}x{quad_frame[1]})"
    else:
        sys.exit(
            "No tray calibration. Crops taken without one will include tray walls\n"
            "and desk clutter, and every label collected against them is wasted.\n"
            "Calibrate first:\n"
            "  python3 tray_framing_check.py --image FRAME.jpg --grid\n"
            "  python3 tray_framing_check.py --image FRAME.jpg --quad ... --save-quad"
        )

    paths = paths_probe
    if not paths:
        sys.exit("no input images matched")

    os.makedirs(os.path.join(args.out, "crops"), exist_ok=True)
    print(f"Crop pipeline v{CROP_VERSION}   tray from: {qsrc}")
    print(f"{args.size}x{args.size} px, context {args.context}x, lossless PNG")
    print(f"{len(paths)} frame(s) -> {args.out}\n")

    records, total, suspects = [], 0, 0
    for p in paths:
        n, s = process(p, quad, quad_frame, args, records)
        total, suspects = total + n, suspects + s
        print(f"  {os.path.basename(p):<50} {n:>3} crops"
              + (f"  ({s} suspect)" if s else ""))

    manifest = os.path.join(args.out, "manifest.jsonl")
    with open(manifest, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\n  {total} crops, {suspects} suspect, {total - suspects} clean")
    print(f"  Manifest -> {manifest}")
    if args.contact_sheet:
        p = contact_sheet(args.out, records)
        if p:
            print(f"  Contact sheet -> {p}")

    print(f"\n  Every record has label=null. Labelling writes into the manifest;")
    print(f"  this script never does. If CROP_VERSION changes, existing labels")
    print(f"  no longer describe the crops they were made against -- regenerate.")


if __name__ == "__main__":
    main()
