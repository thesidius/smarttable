#!/usr/bin/env python3
"""
Phase 0 -- does learned segmentation separate touching dice?

Per docs/instance-segmentation-plan.md. Runs SAM2 automatic mask generation over
frames with known ground truth and reports how many dice it finds, after
filtering.

Run with the ML venv, not the app's Python:
    .venv-ml\\Scripts\\python.exe pi-tower\\sam2_phase0.py

The filter matters as much as the model. AMG returns masks at every scale --
the whole tray, each die, each die's top facet, each painted numeral, wall
seams, shadows. 50-200 raw masks for 8 dice is normal, and skipping the filter
makes a working model look broken.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

# Ground truth, hand-verified. These are tray crops, so every mask is in-tray
# and the quad filter from the plan is not needed here.
FRAMES = [
    ("_boxtest.png", 8, "8 dice, some touching -- the frame the VLM boxed 8/8"),
    ("_ambient.png", 8, "8 scattered: 4x d12, 4x d20"),
    ("_group_tray.png", 7, "7 dice in a TIGHT PILE -- the case classical CV cannot do"),
]


def load_masks(result):
    """Masks as a list of bool arrays, largest first."""
    if result.masks is None:
        return []
    arr = result.masks.data.cpu().numpy().astype(bool)
    return sorted(arr, key=lambda m: -int(m.sum()))


def filter_masks(masks, img_area, lo_frac, hi_frac, nest_thresh=0.80):
    """Plan's filter: area band, then nesting.

    Nesting is what removes facets and painted numerals -- individually the most
    numerous false positives, because every die contributes several.
    """
    kept, rejected = [], {"too_small": 0, "too_large": 0, "nested": 0}
    lo, hi = lo_frac * img_area, hi_frac * img_area

    for m in masks:                       # largest first, so parents precede children
        a = int(m.sum())
        if a < lo:
            rejected["too_small"] += 1
            continue
        if a > hi:
            rejected["too_large"] += 1
            continue
        # >80% contained inside something already kept => a part, not an object
        if any((m & k).sum() > nest_thresh * a for k in kept):
            rejected["nested"] += 1
            continue
        kept.append(m)
    return kept, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sam2.1_b.pt")
    ap.add_argument("--data", default="test-data")
    ap.add_argument("--points-stride", type=int, default=32)
    # Area band as a fraction of the FRAME, since these are tray crops. A die is
    # roughly 1/12 to 1/6 of a tray edge, so ~0.5-4% of tray area; the band is
    # deliberately generous per the plan ("0.5x min to 1.5x max").
    ap.add_argument("--lo-frac", type=float, default=0.004)
    ap.add_argument("--hi-frac", type=float, default=0.10)
    ap.add_argument("--out", default="test-data/_sam2")
    args = ap.parse_args()

    from ultralytics import SAM
    import torch
    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()} | "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"model {args.model} | points_stride {args.points_stride}\n")

    model = SAM(args.model)
    os.makedirs(args.out, exist_ok=True)

    print("%-18s %-6s %-7s %-7s %-8s %s" %
          ("frame", "truth", "raw", "kept", "time", "verdict"))
    rows = []
    for name, truth, note in FRAMES:
        path = os.path.join(args.data, name)
        if not os.path.exists(path):
            print("%-18s  MISSING" % name)
            continue
        im = Image.open(path).convert("RGB")
        area = im.width * im.height

        # points_stride is a parameter of Predictor.generate(), NOT a call kwarg:
        # ultralytics validates kwargs against its config schema and rejects it
        # with "not a valid YOLO argument". The default is already 32, which is
        # what the plan recommends, so only reach for the predictor route when
        # something other than the default is actually needed.
        t0 = time.time()
        if args.points_stride == 32:
            res = model(path, verbose=False)[0]
        else:
            model(path, verbose=False)                    # build the predictor
            model.predictor.args.points_stride = args.points_stride
            res = model(path, verbose=False)[0]
        dt = time.time() - t0

        raw = load_masks(res)
        kept, rej = filter_masks(raw, area, args.lo_frac, args.hi_frac)
        delta = len(kept) - truth
        verdict = "exact" if delta == 0 else ("+%d" % delta if delta > 0 else str(delta))
        print("%-18s %-6d %-7d %-7d %-8.1fs %s" %
              (name, truth, len(raw), len(kept), dt, verdict))
        rows.append({"frame": name, "truth": truth, "raw": len(raw),
                     "kept": len(kept), "seconds": round(dt, 1),
                     "rejected": rej, "note": note})

        # overlay so the masks can be judged by eye, not just counted
        ov = np.array(im).astype(np.float32)
        rng = np.random.default_rng(0)
        for m in kept:
            col = rng.integers(60, 255, 3).astype(np.float32)
            ov[m] = 0.55 * ov[m] + 0.45 * col
        Image.fromarray(ov.astype(np.uint8)).save(
            os.path.join(args.out, name.replace(".jpg", "").replace(".png", "") + "_sam2.png"))

    json.dump(rows, open(os.path.join(args.out, "results.json"), "w"), indent=2)
    print("\noverlays + results.json -> %s" % args.out)
    print("Counting is not the whole gate -- look at the overlays. The pile row is")
    print("the one that matters: it is what classical CV cannot do at all.")


if __name__ == "__main__":
    main()
