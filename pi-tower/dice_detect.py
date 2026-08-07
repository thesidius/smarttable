#!/usr/bin/env python3
"""
dice_detect.py -- find the dice in a tray frame and cut them out.

This is stage 1 of reading dice. Everything downstream -- classifying which
numeral is face-up, whatever form that eventually takes -- needs individual die
crops, and building a labelled training set needs thousands of them. So this
script's job is to turn a tray frame into N crops plus a JSON manifest, and to
be honest about which detections it is unsure of.

Segmentation strategy: LOCAL VARIANCE, not brightness.

Measured on a real frame, the tray floor sits at mean 95.9 with std 1.1 -- it
is almost perfectly featureless. The dice are near-black bodies carrying
bright numerals, so any window covering a die contains both extremes at once.
That makes local standard deviation an enormous discriminator (floor ~1,
die ~40+) while absolute brightness is nearly useless: the die body and the
tray floor are similar shades, and both drift with lighting.

The useful consequence is that this works the same under IR or visible light,
and does not care about the magenta cast or the exposure lock, because it keys
on local structure rather than on absolute level or colour.

Usage:
    python3 dice_detect.py --image ~/dicecam-captures/foo.jpg
    python3 dice_detect.py --capture --crops ~/dice-crops
    python3 dice_detect.py --image foo.jpg --quad 500,250,1205,240,1250,960,445,965
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

CONFIG = os.path.expanduser("~/.config/dicecam/tray.json")


# ------------------------------------------------------------------- masking ---

def fit_quad_to_frame(quad, quad_frame, frame_shape):
    """Rescale a tray quad calibrated at one resolution to another.

    Tray calibration is in pixels, so it is only meaningful against the
    resolution it was measured at. Applying a 1640x1232 quad to a 3280x2464
    frame masks a quarter of the tray and silently yields garbage crops -- no
    error, just quietly wrong training data. So convert explicitly.

    Only valid when both resolutions share an aspect ratio. On IMX219,
    1640x1232 and 3280x2464 are the same field of view at different binning
    (scaling is exact), but 1920x1080 is a *sensor crop* with a genuinely
    narrower FOV -- scaling into it would point the mask at the wrong part of
    the scene, so refuse instead.
    """
    h, w = frame_shape[:2]
    qw, qh = int(quad_frame[0]), int(quad_frame[1])
    if (w, h) == (qw, qh):
        return np.asarray(quad, np.float32), None
    ar_q, ar_f = qw / qh, w / h
    if abs(ar_q - ar_f) > 0.01:
        raise ValueError(
            f"tray calibration is for {qw}x{qh} (aspect {ar_q:.3f}) but this frame "
            f"is {w}x{h} (aspect {ar_f:.3f}). A different aspect ratio means a "
            f"different field of view -- almost certainly a cropped sensor mode -- "
            f"so the quad cannot be rescaled. Recalibrate at this resolution."
        )
    s = w / qw
    return np.asarray(quad, np.float32) * s, f"rescaled x{s:g} from {qw}x{qh}"


# Width in px of the tray in the frame these kernel sizes were actually tuned
# against (20260805-183728_mountcheck, 1640x1232, quad spanning x 445..1250).
# This is a measurement, not a round number -- guessing 750 here silently
# inflated every kernel by ~7%, which was enough for the morphological close to
# bridge the gap between adjacent dice and merge detections that had previously
# been separate.
REF_TRAY_W = 805.0


def _odd(v, lo):
    v = max(int(lo), int(round(v)))
    return v if v % 2 else v + 1


def quad_scale(quad):
    """How large is the tray here, relative to the reference calibration?

    Every spatial parameter below (variance window, morphology kernels) is a
    physical size -- "about a fifth of a die" -- expressed in pixels. Hard-code
    the pixel value and the whole pipeline silently changes behaviour with
    resolution: at 3280x2464 a 21px window is half the relative size it was at
    1640x1232, so dice fragment instead of closing into solid blobs and every
    detection comes back flagged. Scale them with the tray instead.
    """
    q = np.asarray(quad, np.float32)
    return max(0.1, float(q[:, 0].max() - q[:, 0].min()) / REF_TRAY_W)


def tray_mask(shape, quad, scale=1.0):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, np.asarray(quad, dtype=np.int32), 255)
    # pull in slightly: the tray walls meet the floor in a dark seam that
    # otherwise reads as high-variance and gets detected as a die
    k = _odd(15 * scale, 5)
    return cv2.erode(m, np.ones((k, k), np.uint8), iterations=1)


# -------------------------------------------------------------- segmentation ---

def local_std(gray, k=21):
    """Per-pixel standard deviation over a kxk window."""
    f = gray.astype(np.float32)
    mean = cv2.boxFilter(f, -1, (k, k))
    meansq = cv2.boxFilter(f * f, -1, (k, k))
    return np.sqrt(np.maximum(meansq - mean * mean, 0))


def fill_holes(binary):
    """Flood from outside and invert: turns outlined shapes into solid ones."""
    pad = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = pad.copy()
    cv2.floodFill(ff, np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8),
                  (0, 0), 255)
    return cv2.bitwise_or(binary, cv2.bitwise_not(ff)[1:-1, 1:-1])


def estimate_background(gray, die_diam):
    """The tray floor, with its lighting gradient, and the dice removed.

    A global threshold cannot separate dice from floor here: measured under a
    side-mounted lamp, the same empty tray floor reads 86 on the left and 167 on
    the right (region MAD 42). Any single cut-off either keeps the bright half of
    the floor or loses the dark dice.

    Morphological closing with a kernel larger than a die deletes dark objects
    smaller than the kernel -- i.e. the dice -- while following the floor's
    gradient, which varies over a far longer distance. Subtracting that leaves
    the dice and nothing else. Computed downscaled because a 200px kernel on a
    full frame is far too slow on a Pi 3.
    """
    f = 4
    small = cv2.resize(gray, None, fx=1.0 / f, fy=1.0 / f, interpolation=cv2.INTER_AREA)
    k = _odd(die_diam * 1.8 / f, 5)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kern)
    return cv2.resize(bg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)


def segment(gray, mask, k=None, scale=1.0, bg_subtract=False):
    """Dice as (a) departures from the local floor level, OR (b) high local variance.

    *** (a) IS OFF BY DEFAULT: IT CURRENTLY MAKES DETECTION WORSE. ***

    The diagnosis behind it is sound -- see estimate_background() for the
    measured floor gradient -- but this implementation regressed a well-lit
    frame from 4 blobs / 2 clean down to 2 / 1, missing six of seven dice. Most
    likely the background kernel is far too large (1.8x die diameter on a tray
    only ~8 dice wide swallows real structure) and fill_holes then floods
    regions that the tray rim has closed off. It is left behind a flag rather
    than deleted because the underlying problem is real and still unsolved, and
    rather than enabled because shipping a known regression is worse than
    shipping a known limitation.

    Tuning it properly needs a set of labelled frames across lighting
    conditions, not one-shot iteration against whatever is on the tray now.

    Both are needed, because which one carries the signal flips with lighting:

      Well lit  -- die bodies are near-black against a grey floor, and SMOOTH.
                   Brightness separates them cleanly; local variance does not
                   see them at all (only their numerals), so a variance-only
                   detector misses whole dice.
      Dim/narrow-band -- die bodies sink to the floor's own level and only the
                   painted numerals carry anything. Now variance is the only
                   signal and brightness is useless.

    Detection ran into both failure modes on the same rig within an hour, so
    neither cue alone is sufficient. OR them together and fill holes so a die
    outlined by its numerals becomes a solid body.
    """
    if k is None:
        k = _odd(21 * scale, 9)
    if not np.count_nonzero(mask):
        sys.exit("tray mask is empty -- check the quad")

    # (a) departure from the local floor -- opt-in, see docstring
    if bg_subtract:
        die_diam = max(24.0, 120.0 * scale)
        bg = estimate_background(gray, die_diam)
        resid = gray.astype(np.float32) - bg.astype(np.float32)
        rv = resid[mask > 0]
        mad = float(np.median(np.abs(rv - np.median(rv)))) or 1.0
        delta = max(10.0, 3.5 * 1.4826 * mad)
        bright_or_dark = (np.abs(resid) > delta)
    else:
        bright_or_dark = np.zeros(gray.shape, bool)

    # (b) local variance, as before
    sd = local_std(gray, k)
    norm = np.clip(sd, 0, 60) / 60.0 * 255
    norm = norm.astype(np.uint8)
    thr, _ = cv2.threshold(norm[mask > 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(float(thr), 40.0)          # floor it; else Otsu splits pure noise

    binary = ((bright_or_dark | (norm >= thr)) & (mask > 0)).astype(np.uint8) * 255

    mk = _odd(9 * scale, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    if bg_subtract:
        binary = fill_holes(binary)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return binary, sd, thr * 60.0 / 255.0


def add_shapes(dets, labels):
    """Fill in inradius and n_peaks for every detection. Returns the reference.

    Two passes, because peak separation has to be expressed in units of a die
    and we do not know how big a die is until we have measured some. Pass one
    gets each blob's inradius with a minimal separation; pass two re-counts
    peaks using a separation scaled to the median inradius. Inradius is safe to
    median over a clump-contaminated population -- that is the whole point of
    using it instead of area.
    """
    for d in dets:
        x, y, w, h = d["bbox"]
        p = 4                                  # pad so the blob never touches
        y0, x0 = max(0, y - p), max(0, x - p)   # the crop edge, which would
        sub = labels[y0:y + h + p, x0:x + w + p]  # truncate its distance transform
        d["_mask"] = (sub == d["_label"]).astype(np.uint8)
        _, r = blob_shape(d["_mask"], 3)
        d["inradius"] = round(r, 1)

    # Only inradius is available at this point -- n_peaks is what we are about
    # to compute -- so scale the separation off the inradius median directly
    # rather than calling reference_geometry(), which also wants n_peaks.
    # Reference for ONE die. Use a low percentile rather than the median: the
    # median is pulled upward by every clump in frame, and a clump-inflated
    # reference makes the separation window so wide that adjacent die centres
    # merge into a single peak -- which is the failure this is here to avoid.
    # The 30th percentile still sits on a real die when most blobs are singles,
    # and stays near a single die even when several are merged.
    inr = sorted(d["inradius"] for d in dets if d.get("inradius"))
    ref_r = float(np.percentile(inr, 30)) if inr else 0.0

    for d in dets:
        n, _ = blob_shape(d.pop("_mask"),
                          min_sep=max(3.0, 1.2 * ref_r),
                          abs_thresh=(0.60 * ref_r if ref_r else None))
        d["n_peaks"] = n
        d.pop("_label", None)
    return ref_r


def detect(binary, tray_area, min_frac, max_frac):
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, 8)
    lo, hi = min_frac * tray_area, max_frac * tray_area

    keep, rejected = [], {"too_small": 0, "too_large": 0}
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < lo:
            rejected["too_small"] += 1
            continue
        if area > hi:
            # Do NOT drop it. A tight group of seven dice is a normal roll
            # outcome and legitimately large; discarding it produced ZERO
            # detections -- silent total failure, the worst response possible.
            # Keep it, let the clump logic try to split it, and let flag() say
            # so if it cannot.
            rejected["oversized_kept"] = rejected.get("oversized_kept", 0) + 1
        comp = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
        hull_area = cv2.contourArea(cv2.convexHull(
            cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]))
        keep.append({
            "id": len(keep),
            "_label": i,
            "bbox": [int(x), int(y), int(w), int(h)],
            "centroid": [round(float(cents[i][0]), 1), round(float(cents[i][1]), 1)],
            "area_px": int(area),
            "area_frac_of_tray": round(area / tray_area, 5),
            "aspect": round(w / h, 2) if h else 0,
            # kept for diagnostics only -- measured useless as a clump test:
            # a single d6 scored 0.815 while a merged d12+d10 scored 0.849
            "solidity": round(float(area) / hull_area, 3) if hull_area > 0 else 0,
        })
    add_shapes(keep, labels)
    return keep, rejected


def watershed_split(binary, expected_diam, gray=None):
    """Split touching dice into separate blobs. Returns a new binary, or None.

    Dice landing against each other merge into one variance blob -- 5 of 7 did
    on the first real frame -- and a merged blob yields a crop containing two
    dice, which is useless for both training and inference.

    The standard recipe for separating touching convex objects: the distance
    transform peaks once near the centre of each die, so local maxima give one
    seed per die even when their outlines have merged, and watershed on the
    *inverted* distance transform floods outward from those seeds until the
    basins meet -- which happens along the waist between two dice, exactly where
    the cut belongs.

    Seed separation is keyed to expected die size rather than a fraction of the
    global maximum. A global fraction breaks as soon as dice differ in size:
    threshold high enough to split two large dice and a small d4 loses its seed
    entirely and disappears.
    """
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    sep = _odd(max(3, expected_diam * 0.55), 3)
    dilated = cv2.dilate(dist, np.ones((sep, sep), np.uint8))
    # a peak is a pixel no lower than everything within `sep`, and far enough
    # from the edge to be a die centre rather than a bump on the outline
    peaks = ((dist >= dilated - 1e-5) & (dist > expected_diam * 0.18)).astype(np.uint8)

    n_peaks, peak_labels = cv2.connectedComponents(peaks)
    if n_peaks <= 2:                      # background + at most one seed
        return None                        # nothing to split

    markers = peak_labels.astype(np.int32) + 1
    markers[binary == 0] = 1               # background basin
    markers[(binary > 0) & (peaks == 0)] = 0   # unknown, to be flooded

    # Flood over the REAL IMAGE where we have one, not the inverted distance
    # transform.
    #
    # Distance-transform topography cuts at the geometric waist between two
    # blobs, which only exists if they merely touch. Dice that overlap in
    # projection form a convex union with no waist, so that cut lands nowhere
    # useful -- observed as a sliver shaved off the edge of a clump.
    #
    # The boundary between two dice IS visible in the image: bevel highlights
    # and the dark seam between adjacent bodies. cv2.watershed floods on image
    # gradient, so giving it the actual pixels makes basins stop at those real
    # edges instead of at an imaginary waist.
    if gray is not None:
        topo = cv2.cvtColor(cv2.GaussianBlur(gray, (3, 3), 0), cv2.COLOR_GRAY2BGR)
    else:
        dt8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        topo = cv2.cvtColor(255 - dt8, cv2.COLOR_GRAY2BGR)
    cv2.watershed(topo, markers)

    # Return the LABEL image, not a rebuilt binary. Watershed separates basins
    # with a 1-pixel boundary line, and an 8-connected component analysis leaks
    # diagonally across a 1-pixel gap -- the regions silently re-merge and the
    # split looks like it did nothing. Reading the labels directly sidesteps it.
    return markers


def to_gray(bgr, quad, channel="auto"):
    """Pick the grayscale source with the most usable signal. Returns (gray, note).

    cv2.cvtColor(BGR2GRAY) is fixed at 0.299R + 0.587G + 0.114B, which assumes
    roughly balanced illumination. Under narrow-band light that weighting is
    actively harmful. Measured on an empty tray floor under purple LEDs:

        R  mean  8.17  std 4.80  SNR 1.70   2.8% of pixels at hard zero
        G  mean  1.23  std 1.72  SNR 0.71  47.0% at hard zero
        B  mean 84.16  std 14.68 SNR 5.73   0.0% at hard zero

    Luminance gives green -- more noise than signal, half of it clipped away --
    a 59% weight, and blue, the only channel carrying the scene, 11%. The result
    is a tray floor at grey level 13 where die bodies are indistinguishable from
    the floor, so detection finds only the painted numerals.

    Choosing per-frame rather than hard-coding blue: this must also do the right
    thing once the lighting is fixed, when luminance will be the better source.
    """
    if channel and channel != "auto":
        idx = {"b": 0, "g": 1, "r": 2}.get(channel.lower())
        if idx is not None:
            return bgr[:, :, idx].copy(), f"{channel.upper()} channel (forced)"
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), "luminance (forced)"

    mask = tray_mask(bgr.shape, quad, quad_scale(quad))
    if not np.count_nonzero(mask):
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), "luminance (no tray mask)"

    cands = [("luminance", cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))]
    for i, n in ((0, "B"), (1, "G"), (2, "R")):
        cands.append((f"{n} channel", bgr[:, :, i]))

    best, best_score, rows = None, -1.0, []
    for name, ch in cands:
        v = ch[mask > 0].astype(np.float32)
        zero = float((v < 1).mean())
        # Contrast is what segmentation actually consumes, but discount channels
        # that are clipping to black -- their apparent spread is partly the
        # clipping itself, and clipped pixels carry no recoverable detail.
        score = float(v.std()) * (1.0 - zero)
        rows.append(f"{name} std {v.std():.1f} zero {100*zero:.0f}%")
        if score > best_score:
            best, best_score = (name, ch), score

    name, ch = best
    return ch.copy(), f"auto -> {name}  [{'; '.join(rows)}]"


def detect_from_labels(labels, tray_area, min_frac, max_frac):
    """Build detections from a watershed label image (0=boundary, 1=background)."""
    lo, hi = min_frac * tray_area, max_frac * tray_area
    keep, rejected = [], {"too_small": 0, "too_large": 0}
    for lab in np.unique(labels):
        if lab <= 1:
            continue
        comp = (labels == lab).astype(np.uint8)
        area = int(comp.sum())
        if area < lo:
            rejected["too_small"] += 1
            continue
        if area > hi:
            rejected["too_large"] += 1
            continue
        x, y, w, h = cv2.boundingRect(comp)
        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_area = cv2.contourArea(cv2.convexHull(max(cs, key=cv2.contourArea)))
        m = cv2.moments(comp, binaryImage=True)
        keep.append({
            "id": len(keep),
            "_label": int(lab),
            "bbox": [int(x), int(y), int(w), int(h)],
            "centroid": [round(m["m10"] / m["m00"], 1), round(m["m01"] / m["m00"], 1)]
                        if m["m00"] else [float(x + w / 2), float(y + h / 2)],
            "area_px": area,
            "area_frac_of_tray": round(area / tray_area, 5),
            "aspect": round(w / h, 2) if h else 0,
            "solidity": round(float(area) / hull_area, 3) if hull_area > 0 else 0,
        })
    add_shapes(keep, labels)
    return keep, rejected


def _split_one(binary, det, expected_diam, tray_area, min_frac, max_frac, gray=None):
    """Watershed a single clump, in its own bbox window. Returns pieces or None."""
    x, y, w, h = det["bbox"]
    p = 6
    y0, x0 = max(0, y - p), max(0, x - p)
    win = binary[y0:y + h + p, x0:x + w + p]
    if win.size == 0:
        return None

    # Keep only the clump itself: a neighbouring die may intrude into the bbox,
    # and flooding it too would produce pieces that duplicate an existing
    # detection.
    n, lbl = cv2.connectedComponents((win > 0).astype(np.uint8), 8)
    if n < 2:
        return None
    biggest = 1 + int(np.argmax([(lbl == i).sum() for i in range(1, n)]))
    isolated = np.where(lbl == biggest, 255, 0).astype(np.uint8)

    # Sweep the die diameter instead of trusting one estimate.
    #
    # Seed separation is the single parameter that decides whether a clump
    # splits into dice, slivers, or nothing -- and when EVERY die in frame is
    # part of a clump there is no clean die to measure. The inradius, which is
    # near-invariant for two touching dice, inflates badly for a dense pile:
    # measured 81 for a 7-dice group against ~50 for a lone die, which pushed
    # the estimate to 162 and produced one 77k-px piece plus rubble. Sweeping
    # around an area-based estimate and keeping the split whose piece count
    # matches the peak count is self-correcting.
    #
    # Topography is the distance transform, NOT the image. Flooding over real
    # pixels was measured far worse here (2 pieces covering 8% of the clump vs
    # 7 pieces covering 73%): the background marker wins territory through weak
    # floor-to-die gradients, and strong internal edges from numerals stall the
    # flood inside each die.
    clump_area = float(np.count_nonzero(isolated))
    want = max(2, int(det.get("n_peaks", 2)))
    base = (clump_area / want) ** 0.5

    # Do NOT target the peak count. It is not stable enough to steer on -- the
    # same physical pile reported 7, then 6, then 5 peaks across consecutive
    # frames as segmentation noise moved the distance maxima around. Steering on
    # it makes the sweep chase a moving target and settle on a coarse split
    # padded with slivers.
    #
    # Instead: reject any split that produces a sliver or loses half the clump,
    # then among the splits that survive take the FINEST one. More pieces means
    # more dice actually separated, and the quality gate is what stops that
    # preference from running away into over-segmentation.
    best, best_key = None, None
    for f in (0.55, 0.65, 0.8, 0.95, 1.1, 1.25, 1.45):
        labels = watershed_split(isolated, max(8.0, base * f))
        if labels is None:
            continue
        cand, _ = detect_from_labels(labels, tray_area, min_frac, max_frac)
        if len(cand) < 2:
            continue
        areas = sorted(c["area_px"] for c in cand)
        med = float(np.median(areas))
        if med <= 0:
            continue
        uniformity = areas[0] / med          # dice in one pile are similar sizes
        coverage = sum(areas) / max(1.0, clump_area)
        if uniformity < 0.30 or coverage < 0.50:
            continue
        key = (len(cand), coverage)
        if best_key is None or key > best_key:
            best, best_key = cand, key

    if best is None:
        return None
    pieces = best
    for pc in pieces:                      # back into full-frame coordinates
        pc["bbox"][0] += x0
        pc["bbox"][1] += y0
        pc["centroid"][0] += x0
        pc["centroid"][1] += y0
    return pieces


def detect_all(gray, quad, min_frac=0.002, max_frac=0.45, window=None, split=True):
    """Full detection: mask -> variance segment -> components -> split clumps.

    Shared by dice_detect and crop_pipeline so both see identical detections.
    """
    scale = quad_scale(quad)
    mask = tray_mask(gray.shape, quad, scale)
    tray_area = float(np.count_nonzero(mask))
    if tray_area == 0:
        return [], {"too_small": 0, "too_large": 0}, {"error": "empty tray mask"}

    binary, _, thr = segment(gray, mask, window, scale)
    dets, rejected = detect(binary, tray_area, min_frac, max_frac)
    info = {"tray_area": tray_area, "threshold": thr, "scale": scale, "split": None}

    if not split or not dets:
        return dets, rejected, info

    # Die size comes from the inradius, which does not inflate when dice merge,
    # so it needs no "only measure the clean ones" bootstrap -- the earlier
    # version of this depended on flag(), which meant a bad size heuristic could
    # also silently block good splits.
    ref = reference_geometry(dets)
    clumps = [d for d in dets if is_clump(d, ref)]
    info["peaks"] = {d["id"]: d.get("n_peaks") for d in dets}

    if not clumps:
        info["split"] = "not needed -- every blob has a single distance peak"
        return dets, rejected, info

    expected_diam = max(8.0, 2.0 * ref["inradius"])
    basis = (f"inradius {ref['inradius']:.0f}px, single-die area "
             f"{ref['single_area']:.0f}px over {len(dets)} blob(s); "
             f"{len(clumps)} genuine clump(s)")

    # Split ONLY the blobs identified as clumps, one at a time, and leave every
    # other detection untouched.
    #
    # Running watershed over the whole binary re-carved perfectly good single
    # dice too, turning whole-die boxes back into numeral-sized fragments. It
    # got accepted anyway because the guard recomputes its size reference on the
    # split output: when EVERYTHING shrinks uniformly, each piece still looks
    # proportionate and the "clean" count goes up. Splitting per clump removes
    # that whole failure mode -- a good detection is never a candidate for
    # damage in the first place.
    new_dets, n_split = [], 0
    for d in dets:
        if not is_clump(d, ref):
            new_dets.append(d)
            continue
        pieces = _split_one(binary, d, expected_diam, tray_area, min_frac,
                            max_frac, gray)

        # Every piece must look like a die, or the split is rejected wholesale.
        #
        # Without this the watershed "succeeds" by shaving a sliver off the edge
        # of a clump: it returns two components, the count of clean detections
        # goes up, and the guard accepts -- while the two dice the user actually
        # complained about are STILL merged, and now carry a single distance
        # peak each so nothing flags them any more. A split that leaves a
        # fragment behind is worse than no split, because it launders a known
        # problem into a clean-looking result.
        # Judge the pieces against EACH OTHER, not against a reference measured
        # elsewhere. When every die in frame is clumped there is no clean die to
        # compare to, and `ref` is then derived from the clump itself -- an
        # inflated inradius (74 for a 7-dice pile vs ~50 for a lone die) that
        # rejects every genuinely correct piece. Dice in one pile are all roughly
        # the same size, so internal consistency is the honest test:
        #   - no piece may be a sliver relative to its siblings
        #   - the split must account for most of the clump it came from
        ok = False
        if pieces and len(pieces) > 1:
            areas = sorted(p["area_px"] for p in pieces)
            med = float(np.median(areas))
            coverage = sum(areas) / max(1.0, float(d["area_px"]))
            ok = med > 0 and areas[0] >= 0.30 * med and coverage >= 0.50

        if ok:
            new_dets.extend(pieces)
            n_split += 1
        else:
            # Keep the clump AND keep it flagged, so downstream knows the box
            # holds more than one die rather than silently trusting it.
            new_dets.append(d)

    if not n_split:
        info["split"] = f"{len(clumps)} clump(s) found but none separable"
        return dets, rejected, info

    # Do NOT recompute shapes from the binary here. Split pieces already carry
    # inradius/n_peaks measured against the WATERSHED LABELS, which is the only
    # place the split actually exists -- the binary still shows the clump as one
    # connected region, so re-measuring there would hand each piece the merged
    # blob's geometry and undo the split in the statistics.
    for i, d in enumerate(new_dets):
        d["id"] = i
    new_rejected = rejected

    # Only accept the split if it produced MORE USABLE DICE, not merely more
    # blobs. Watershed over-segments happily -- on a blob whose silhouette is
    # ragged it will carve slivers with solidity 0.4 and call it progress. More
    # blobs is not the goal; more single, clean dice is. Measured this way the
    # split can never make the output worse than leaving the clump alone.
    def clean_count(ds):
        if not ds:
            return 0
        return sum(1 for d in ds if not flag(d, reference_geometry(ds)))

    before, after = clean_count(dets), clean_count(new_dets)
    if after <= before:
        info["split"] = (f"attempted (diam~{expected_diam:.0f}px from {basis}) and "
                         f"REJECTED: clean dice {before} -> {after}, no improvement. "
                         f"Keeping the unsplit blobs.")
        return dets, rejected, info

    info["split"] = (f"accepted: {len(dets)} -> {len(new_dets)} blobs, "
                     f"clean dice {before} -> {after} "
                     f"(diam~{expected_diam:.0f}px from {basis})")
    return new_dets, new_rejected, info


def blob_shape(comp_mask, min_sep, abs_thresh=None):
    """Describe one blob by its distance transform. Returns (n_peaks, inradius).

    The inradius -- radius of the largest circle that fits inside the blob -- is
    roughly INVARIANT to clumping. Two touching dice do not admit a bigger
    inscribed circle than one die does, because the waist between them pinches
    it. Area does not have that property: it simply doubles. That makes inradius
    the reliable size reference and area a misleading one.

    Peak count is the clump test. One die gives a single distance maximum at its
    centre; two touching dice give two, separated by roughly a die width.
    """
    # Zero border is not cosmetic. distanceTransform measures distance to the
    # nearest ZERO pixel, so a mask that fills its whole window has no zero to
    # measure against and returns unbounded values -- observed as an inradius of
    # 65533.8 (2^16-1), which then poisons the median every other test uses.
    # The border also makes the result correct for a blob touching the crop
    # edge, rather than merely finite.
    m = cv2.copyMakeBorder(comp_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    r = float(dist.max())
    if r <= 0:
        return 0, 0.0
    sep = _odd(max(3.0, min_sep), 3)
    dilated = cv2.dilate(dist, np.ones((sep, sep), np.uint8))

    # Threshold ABSOLUTELY when a single-die size is known, not relative to this
    # blob's own maximum.
    #
    # `dist > 0.55 * r` silently fails on exactly the case it exists for. Three
    # dice in contact form a union fat enough that its peak distance is far
    # larger than one die's, so 0.55 of it sits ABOVE the individual die centres
    # and only the fattest lobe survives. Measured: a blob holding three plainly
    # separate dice reported n_peaks = 1, so it was never classified as a clump
    # and no split was attempted -- it just came back as one confident die.
    thresh = abs_thresh if abs_thresh else 0.55 * r
    peaks = ((dist >= dilated - 1e-5) & (dist > thresh)).astype(np.uint8)
    n, _ = cv2.connectedComponents(peaks)
    return max(0, n - 1), r


def reference_geometry(dets):
    """Typical single-die size. Returns {inradius, single_area}.

    `single_area` is the median area of blobs showing exactly ONE distance peak
    — single dice by construction, so unlike a plain median it is not inflated
    by clumps.

    `inradius` resists clumping better than area does but is not immune to it:
    two dice touching at a point keep a pinched waist, yet two that overlap
    substantially in projection form a blobbier union that admits a larger
    inscribed circle. Measured here, a merged d12+d10 reached inradius 75
    against a 50 reference. Good enough for a lower bound on "is this even
    die-sized", not trustworthy as an upper one.
    """
    inr = [d["inradius"] for d in dets if d.get("inradius")]
    singles = [d["area_px"] for d in dets if d.get("n_peaks", 1) == 1]
    return {
        "inradius": float(np.median(inr)) if inr else 0.0,
        "single_area": float(np.median(singles)) if singles else 0.0,
    }


def is_clump(d, ref):
    """Two tests, both required, because each alone gives false positives.

    Peak count alone flags elongated single dice: a d6 photographed at an angle
    shows a top face and a side face joined at a narrow waist, which reads as
    two distance maxima. Measured here it did exactly that — while being the
    SMALLEST blob in frame, so it could not possibly hold two dice.

    Area alone cannot work either, since a polyhedral set spans 2.8x in area
    between a d6 and a d20.

    Together they are reliable: a real clump has multiple lobes AND is
    substantially larger than a lone die.
    """
    if d.get("n_peaks", 1) < 2:
        return False
    if not ref.get("single_area"):
        return True          # nothing to compare against; trust the peak count

    # 1.3x, not 1.6x. Two dice do not weigh twice one die when they are
    # different sizes: a d12 beside a d20 measured 1.56x the single-die
    # reference and slipped under a 1.6 gate, so a box plainly holding two dice
    # came back clean. The peak test carries most of the weight now that it uses
    # an absolute threshold, so this only has to exclude a lone die -- and a lone
    # die is 1.0x by construction.
    return d["area_px"] >= 1.3 * ref["single_area"]


def flag(d, ref):
    """Reasons to distrust a detection. Empty list == looks like one clean die.

    Deliberately NOT based on area relative to the median. A polyhedral set has
    genuinely different die sizes -- measured on this rig, real single dice
    spanned 8292 to 23457 px, a 2.8x spread -- so any area-vs-median rule either
    flags the legitimate d4 and d20 or misses real clumps. Worse, the median is
    computed over a population that includes clumps, which inflates it and makes
    small real dice look like fragments.

    Solidity is out for the same reason: measured here, a single d6 scored 0.815
    while a merged d12+d10 scored 0.849. Two dice side by side form a perfectly
    convex-looking blob. Convexity does not distinguish them.

    What does: counting die-sized lobes inside the blob, and sizing against the
    clump-invariant inradius rather than area.
    """
    reasons = []
    if is_clump(d, ref):
        reasons.append(f"contains {d.get('n_peaks')} dice -- touching, needs splitting")
    if ref.get("inradius") and d.get("inradius", 0) < 0.45 * ref["inradius"]:
        reasons.append("too small to be a die -- fragment or artefact")
    if not 0.45 <= d["aspect"] <= 2.2:
        reasons.append(f"implausible aspect {d['aspect']}")
    return reasons


# ------------------------------------------------------------------- outputs ---

def annotate(img, quad, dets, flags):
    out = img.copy()
    cv2.polylines(out, [np.asarray(quad, dtype=np.int32)], True, (0, 180, 0), 2)
    for d in dets:
        x, y, w, h = d["bbox"]
        bad = bool(flags[d["id"]])
        col = (0, 0, 255) if bad else (0, 255, 0)
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 3)
        cv2.putText(out, f"#{d['id']}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, col, 2, cv2.LINE_AA)
        if bad:
            cv2.putText(out, "?", (x + w - 24, y + 28), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, col, 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description="Detect dice in a tray frame")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--capture", action="store_true")
    g.add_argument("--image")
    ap.add_argument("--quad", help="tray corners x1,y1,...,x4,y4")
    ap.add_argument("--size", default="1640x1232")
    ap.add_argument("--crops", default=None, help="directory to write per-die crops")
    ap.add_argument("--pad", type=int, default=8, help="pixels of context around each crop")
    ap.add_argument("--window", type=int, default=None,
                    help="local-variance window in px; default scales with tray size")
    ap.add_argument("--quad-frame", help="resolution --quad was measured at, e.g. 1640x1232")
    ap.add_argument("--no-split", action="store_true",
                    help="skip the watershed split of touching dice")
    ap.add_argument("--min-frac", type=float, default=0.002,
                    help="min blob area as a fraction of tray area")
    ap.add_argument("--max-frac", type=float, default=0.45,
                    help="a tight group of 7 dice is legitimately large; do not set this low")
    ap.add_argument("--overlay", default=None)
    args = ap.parse_args()

    if args.capture:
        noir = "/usr/share/libcamera/ipa/rpi/vc4/imx219_noir.json"
        if "LIBCAMERA_RPI_TUNING_FILE" not in os.environ and os.path.exists(noir):
            os.environ["LIBCAMERA_RPI_TUNING_FILE"] = noir
        from picamera2 import Picamera2
        import time
        p = Picamera2()
        main_cfg = {"format": "RGB888", "size": tuple(int(v) for v in args.size.lower().split("x"))}
        p.configure(p.create_still_configuration(main=main_cfg))
        p.start(); time.sleep(3)
        img = p.capture_array("main"); p.stop(); p.close()
        src = os.path.expanduser("~/dicecam-captures/dice_detect_capture.jpg")
        cv2.imwrite(src, img)
    else:
        src = os.path.expanduser(args.image)
        img = cv2.imread(src)
        if img is None:
            sys.exit(f"could not read {src}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    if args.quad:
        quad = np.array([int(v) for v in args.quad.split(",")], np.float32).reshape(4, 2)
        qf = ([int(v) for v in args.quad_frame.lower().split("x")]
              if args.quad_frame else [w, h])
        quad, note = fit_quad_to_frame(quad, qf, gray.shape)
        qsrc = "--quad" + (f" ({note})" if note else "")
    elif os.path.exists(CONFIG):
        cfg = json.load(open(CONFIG))
        quad = np.array(cfg["quad"], np.float32).reshape(4, 2)
        quad, note = fit_quad_to_frame(quad, cfg.get("frame") or [w, h], gray.shape)
        qsrc = CONFIG + (f" ({note})" if note else "")
    else:
        quad = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        qsrc = "WHOLE FRAME (no tray calibration -- expect junk outside the tray)"

    print(f"Frame {w}x{h}   tray from: {qsrc}")

    dets, rejected, info = detect_all(gray, quad, args.min_frac, args.max_frac,
                                      args.window, split=not args.no_split)
    tray_area, raw_thr = info["tray_area"], info["threshold"]

    median_area = float(np.median([d["area_px"] for d in dets])) if dets else 0
    flags = {d["id"]: flag(d, median_area) for d in dets}
    clean = [d for d in dets if not flags[d["id"]]]

    print(f"Tray area {int(tray_area)} px   variance threshold {raw_thr:.1f} "
          f"(floor measures ~1)")
    print(f"Watershed split: {info['split']}")
    print(f"\nDetected {len(dets)} blob(s): {len(clean)} clean, "
          f"{len(dets) - len(clean)} suspect")
    print(f"  rejected: {rejected['too_small']} too small, "
          f"{rejected['too_large']} too large\n")

    print(f"  {'#':<4}{'bbox':<26}{'area':<9}{'aspect':<9}{'solidity':<10}note")
    for d in dets:
        note = "; ".join(flags[d["id"]]) or "ok"
        print(f"  {d['id']:<4}{str(d['bbox']):<26}{d['area_px']:<9}"
              f"{d['aspect']:<9}{d['solidity']:<10}{note}")

    if args.crops:
        os.makedirs(args.crops, exist_ok=True)
        base = os.path.splitext(os.path.basename(src))[0]
        for d in dets:
            x, y, cw, ch = d["bbox"]
            p = args.pad
            crop = img[max(0, y - p):min(h, y + ch + p),
                       max(0, x - p):min(w, x + cw + p)]
            name = f"{base}_die{d['id']:02d}{'_suspect' if flags[d['id']] else ''}.png"
            cv2.imwrite(os.path.join(args.crops, name), crop)
            d["crop"] = name
        print(f"\n  Crops -> {args.crops}")

    out = args.overlay or os.path.splitext(src)[0] + "_dice.jpg"
    cv2.imwrite(out, annotate(img, quad, dets, flags))
    manifest = os.path.splitext(out)[0] + ".json"
    json.dump({"source": src, "tray_quad": np.asarray(quad).astype(int).tolist(),
               "detections": dets,
               "flags": {str(k): v for k, v in flags.items()}},
              open(manifest, "w"), indent=2)
    print(f"  Overlay -> {out}   (green = clean, red '?' = suspect)")
    print(f"  Manifest -> {manifest}")

    if len(dets) - len(clean):
        print("\n  Suspect detections are usually dice resting against each other:")
        print("  the variance blobs merge into one component. Separating them needs")
        print("  a watershed split, which is only worth adding once we know how often")
        print("  it actually happens in real rolls.")


if __name__ == "__main__":
    main()
