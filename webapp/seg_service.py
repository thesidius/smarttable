#!/usr/bin/env python3
"""
seg_service.py -- SAM2 instance segmentation, Phase 1.

Runs in its OWN process and its OWN interpreter. The control panel is on Python
3.14, for which no CUDA torch wheel exists; this needs 3.13 + torch-cu126. A
service rather than a subprocess because SAM2 takes seconds to load and would
otherwise reload on every roll.

    .venv-ml\\Scripts\\python.exe webapp\\seg_service.py

POST /segment  {"image": "<base64 png/jpg>"}
  -> {"count": N, "dice": [{"bbox", "area", "crop": "<base64 png>"}], "seconds"}

Each returned crop is the die COMPOSITED ONTO NEUTRAL GREY using its mask, not
a rectangular cut-out. That is the entire point of Phase 1: a bounding box
around a die in a pile still contains its neighbours, and neighbouring dice are
what made the reader misread -- measured, a d20 showing 20 read as "14" in a
pile and correctly as "20" once isolated.

Measured on this rig (docs/dice-reading.md, Phase 0): 8/8 scattered dice, 7/7
in a tight pile, 1.4-2.4 s on a 4090. The distance-transform approach it
replaces reported 20 dice for 8.
"""

import base64
import io
import os
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

MODEL_NAME = os.environ.get("DICECAM_SAM_MODEL", "sam2.1_b.pt")
PORT = int(os.environ.get("DICECAM_SEG_PORT", "8090"))

# Fraction-of-frame area band for a plausible die. Deliberately generous: real
# single-die areas span 2.9x between a d6 and a d20, so a tight band would drop
# the extremes of a mixed set.
LO_FRAC = float(os.environ.get("DICECAM_SEG_LO", "0.004"))
HI_FRAC = float(os.environ.get("DICECAM_SEG_HI", "0.10"))
NEST_THRESH = 0.80
BORDER_PX = 4          # a die never touches the frame edge; the tray walls are inside it
MIN_ASPECT, MAX_ASPECT = 0.45, 2.2
MIN_FILL = 0.45        # mask area / bbox area -- a die ~0.7-0.8, a sliver ~0.1
CROP_CONTEXT = 1.25          # a little background so the die is not flush to the edge
NEUTRAL = 128                # mid grey; see _crop()

# BUMP THIS whenever the crop geometry changes -- context ratio, compositing
# background, mask cleanup, anything that alters the pixels a reader or a
# classifier sees. Labels are collected against a specific crop definition and
# a change silently invalidates them, which is not detectable after the fact
# unless the definition was recorded at the time.
#
# 1  first live version: mask-composited onto grey, 1.25x square context
# 2  largest-connected-component cleanup -- a stray mask pixel had been
#    off-centring and inflating crops, so every crop's framing changed
CROP_VERSION = 2

app = Flask(__name__)
_model = None


def model():
    global _model
    if _model is None:
        from ultralytics import SAM
        import torch
        print(f"loading {MODEL_NAME} | cuda={torch.cuda.is_available()}", flush=True)
        _model = SAM(MODEL_NAME)
    return _model


def _largest_component(m):
    """Drop everything but the biggest connected blob in a mask.

    SAM2 masks routinely carry a few stray pixels tens of percent of the frame
    away from the die -- a speck of specular highlight, a bit of a neighbour.
    They are far too small to change the area filter, but min/max on the raw
    coordinates is decided by the single most distant pixel, so one speck
    off-centres the crop and inflates it. Measured: a d20 crop was pushed to
    the right-hand third of its own image by one stray pixel, throwing away
    most of the resolution the reader gets to look at.
    """
    n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    if n <= 2:                                     # background + one blob
        return m
    counts = np.bincount(lab.ravel())
    counts[0] = 0                                  # label 0 is background
    return lab == counts.argmax()


def _filter(masks, area, shape):
    """Area band, shape plausibility, border rejection, then nesting.

    Nesting removes facets and painted numerals -- the most numerous false
    positives, since every die contributes several and AMG returns masks at
    every scale (50-200 raw for 8 dice is normal).

    The shape and border rules exist because of what actually came through on
    the first live run: a thin tray-edge sliver, a LEGO brick sitting outside
    the tray, and an empty corner. All three touched the frame edge; no die
    does, because the tray walls are inside the crop. A die is also roughly
    round -- it fills most of its bounding box and is not far from square --
    which the sliver and the brick are not.
    """
    H, W = shape[:2]
    kept, rejected = [], {"area": 0, "border": 0, "shape": 0, "nested": 0}
    lo, hi = LO_FRAC * area, HI_FRAC * area

    for m in sorted(masks, key=lambda x: -int(x.sum())):
        a = int(m.sum())
        if a < lo or a > hi:
            rejected["area"] += 1
            continue

        # Before any geometry is measured: fill and aspect are computed from the
        # bounding box, so a stray speck would corrupt the shape test as well as
        # the crop.
        m = _largest_component(m)
        a = int(m.sum())

        ys, xs = np.where(m)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        if x0 <= BORDER_PX or y0 <= BORDER_PX or x1 >= W - 1 - BORDER_PX or y1 >= H - 1 - BORDER_PX:
            rejected["border"] += 1
            continue

        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        aspect = bw / float(bh)
        fill = a / float(bw * bh)          # a die fills ~0.7-0.8 of its box; a sliver ~0.1
        if not (MIN_ASPECT <= aspect <= MAX_ASPECT) or fill < MIN_FILL:
            rejected["shape"] += 1
            continue

        if any((m & k).sum() > NEST_THRESH * a for k in kept):
            rejected["nested"] += 1
            continue
        kept.append(m)
    return kept, rejected


def _crop(rgb, mask):
    """One die on neutral grey, square, mask-composited.

    This is the canonical crop now -- what the reader sees and what the Label
    tab files as training data -- so the definition from the retired
    crop_pipeline.py carries over, with one deliberate divergence:

      SQUARE, centred on the mask. Dice are roughly square in bbox; forcing
          square before any resize means nothing downstream sees an
          aspect-distorted die.
      FIXED CONTEXT RATIO (CROP_CONTEXT). A little background so the die is not
          flush to the edge.
      LOSSLESS PNG. JPEG ringing sits exactly on the high-contrast numeral
          edges that carry the signal.
      NO ROTATION NORMALISATION. Orientation in the tray is genuinely random
          and the reader has to cope; straightening here would hide that until
          deployment.
      NOT RESIZED to a fixed side -- the divergence. crop_pipeline normalised
          scale so a classifier could not cheat by learning "big blob = d20".
          These crops go to a vision model, for which downsampling a 625 px die
          to a common size throws away exactly the numeral resolution that
          decides the read. If a classifier is trained on them later, normalise
          then, from the stored PNG.

    Grey rather than black or white: a hard black border is a strong artificial
    edge that a reader latches onto, and white blows out against pale dice.
    Mid-grey sits between the darkest and palest dice measured here (near-black
    ~20 to pale ~180+), so it does not masquerade as part of any of them.
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None, None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = int(max(x1 - x0, y1 - y0) * CROP_CONTEXT)

    out = np.full((side, side, 3), NEUTRAL, np.uint8)
    sx, sy = int(cx - side / 2), int(cy - side / 2)
    # intersection of the crop window with the frame
    ax0, ay0 = max(0, sx), max(0, sy)
    ax1, ay1 = min(rgb.shape[1], sx + side), min(rgb.shape[0], sy + side)
    if ax1 <= ax0 or ay1 <= ay0:
        return None, None
    sub = rgb[ay0:ay1, ax0:ax1]
    sub_m = mask[ay0:ay1, ax0:ax1]
    dst = out[ay0 - sy:ay1 - sy, ax0 - sx:ax1 - sx]
    dst[sub_m] = sub[sub_m]          # ONLY this die's pixels; neighbours stay grey

    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG")
    return buf.getvalue(), [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


# Well-separated hues, not random. A random palette produced two near-identical
# tints on adjacent dice, which is precisely the case the overlay exists to
# make obvious.
PALETTE = [(255, 60, 60), (60, 220, 90), (70, 130, 255), (255, 200, 40),
           (230, 70, 230), (40, 230, 230), (255, 130, 40), (150, 100, 255),
           (120, 255, 160), (255, 90, 150)]


def _edge(m, width=5):
    """Boundary band of a boolean mask, in pure numpy."""
    e = np.zeros_like(m)
    e[:-1, :] |= m[:-1, :] ^ m[1:, :]
    e[1:, :] |= m[:-1, :] ^ m[1:, :]
    e[:, :-1] |= m[:, :-1] ^ m[:, 1:]
    e[:, 1:] |= m[:, :-1] ^ m[:, 1:]
    for _ in range(width - 1):                     # thicken so it survives JPEG
        d = e.copy()
        d[:-1, :] |= e[1:, :]; d[1:, :] |= e[:-1, :]
        d[:, :-1] |= e[:, 1:]; d[:, 1:] |= e[:, :-1]
        e = d
    return e & m


def _overlay(rgb, kept):
    """Tinted masks over a dimmed frame -- the merge/miss diagnostic.

    When a roll reads wrong, "segmentation put two dice in one mask" and "the
    reader misread a clean crop" look identical from the values alone. This is
    what tells them apart, so it has to be readable at a glance:

      - background dimmed, so anything NOT segmented is visibly dark. A die
        that SAM2 missed entirely is otherwise invisible in an overlay.
      - a fixed, well-separated palette rather than random colours.
      - a hard outline per mask. Fill alone is ambiguous on dice that already
        carry strong colour; a boundary is not. Two dice inside one outline is
        a merge, and that reads instantly.
    """
    ov = rgb.astype(np.float32) * 0.38
    for i, m in enumerate(kept):
        col = np.array(PALETTE[i % len(PALETTE)], np.float32)
        ov[m] = 0.72 * rgb[m] + 0.28 * col          # keep the pips legible
        ov[_edge(m)] = col
    buf = io.BytesIO()
    Image.fromarray(ov.astype(np.uint8)).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@app.get("/health")
def health():
    import torch
    return jsonify({"ok": True, "model": MODEL_NAME,
                    "cuda": torch.cuda.is_available(),
                    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                    "loaded": _model is not None})


@app.post("/segment")
def segment():
    body = request.get_json(force=True) or {}
    b64 = body.get("image")
    if not b64:
        return jsonify({"error": "image (base64) required"}), 400
    try:
        im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"could not decode image: {e}"}), 400

    rgb = np.array(im)
    t0 = time.time()
    tmp = os.path.join(os.environ.get("TEMP", "."), "_seg_in.png")
    im.save(tmp)
    try:
        res = model()(tmp, verbose=False)[0]
    except Exception as e:
        return jsonify({"error": f"segmentation failed: {e}"}), 500

    raw = [] if res.masks is None else res.masks.data.cpu().numpy().astype(bool)
    kept, rej = _filter(list(raw), im.width * im.height, rgb.shape)

    dice = []
    for m in kept:
        png, bbox = _crop(rgb, m)
        if png is None:
            continue
        dice.append({"bbox": bbox, "area": int(m.sum()),
                     "crop": base64.b64encode(png).decode()})

    # Top-left reading order, so ids are stable enough for a human to follow.
    dice.sort(key=lambda d: (d["bbox"][1] // 200, d["bbox"][0]))
    for i, d in enumerate(dice):
        d["id"] = i

    overlay = None
    if body.get("overlay"):
        overlay = base64.b64encode(_overlay(rgb, kept)).decode()

    return jsonify({"count": len(dice), "raw_masks": len(raw),
                    "rejected": rej, "dice": dice, "overlay": overlay,
                    "seconds": round(time.time() - t0, 2), "model": MODEL_NAME,
                    "crop_version": CROP_VERSION,
                    "crop_context": CROP_CONTEXT})


if __name__ == "__main__":
    print(f"seg_service on :{PORT}  model={MODEL_NAME}")
    model()                      # load up front, not on the first roll
    app.run(host="0.0.0.0", port=PORT, threaded=True)
