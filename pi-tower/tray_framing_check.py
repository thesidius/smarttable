#!/usr/bin/env python3
"""
tray_framing_check.py -- is the whole dice tray in frame, and readable everywhere?

Answers the questions that decide whether a mount is good enough:

  1. Is the entire tray inside the frame, with margin on all four sides?
  2. Does the tray fill enough of the frame to be worth the pixels?
  3. Is every corner of the tray as bright and as sharp as the centre?

(3) is the one that catches bad mounts. An over-tilted or too-close camera
shows up first at the corners -- as vignetting (corner darker than centre) and
as depth-of-field falloff (corner softer than centre, because it sits at a
different distance than the centre does). Both are invisible in a quick glance
at a preview and both quietly wreck dice reading in exactly the places dice
end up: against the walls.

Usage:
    python3 tray_framing_check.py --capture
    python3 tray_framing_check.py --image ~/dicecam-captures/foo.jpg
    python3 tray_framing_check.py --image foo.jpg --quad 400,200,1200,210,1210,980,390,970

Writes an annotated overlay next to the source image (or --overlay PATH).
"""

import argparse
import os
import sys

import cv2
import numpy as np

TARGET_TRAY_IN = (7.5, 5.0)   # tray footprint we are designing around
CONFIG = os.path.expanduser("~/.config/dicecam/tray.json")


# ------------------------------------------------------------------ detect ---

def find_tray_quad(gray):
    """Largest convex 4-gon in the frame. Returns (quad, confidence_note).

    A dark tray on a dark desk is the hard case, and it is the normal one here:
    measured on a real frame the tray floor sat at 95.9, the tray wall at 96.8
    and the desk at 110.0 -- 14 grey levels across the whole scene. Median-scaled
    Canny thresholds see nothing at all at that contrast, so equalise local
    contrast first (CLAHE) and threshold off the equalised image, then sweep a
    few sensitivities rather than betting on one.
    """
    # Strategy 1: flood fill the tray floor.
    #
    # Edge detection loses here -- it reliably finds the LEGO baseplate (bright
    # studs, high contrast) and misses the tray entirely. But the tray floor has
    # a property nothing else in frame has: it is almost perfectly uniform
    # (measured std 1.1, against 2.0 for the desk and 33.4 for LEGO), and it is
    # ringed by its own walls. So seed inside it and grow outward with a tight
    # tolerance; the walls stop the fill. FIXED_RANGE compares every pixel to the
    # seed rather than to its neighbour, so the fill cannot creep up a gradient
    # and escape over the rim.
    # Two traps, both hit for real on the first attempts:
    #   - "take the largest valid fill" rewards leakage. The desk (110) is only
    #     14 levels off the floor (96) and they connect around the rim, so a
    #     loose tolerance swallows the whole scene and wins on size.
    #   - so instead: sweep tolerance ASCENDING and stop at the first one that
    #     produces a convincingly rectangular blob. A leaked fill is a ragged
    #     shape that shares little of its bounding rect; a clean tray floor
    #     fills its own bounding rect almost completely.
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    for tol in (4, 6, 8, 10, 12, 14):
        cands = []
        for fy in (0.40, 0.50, 0.60):
            for fx in (0.40, 0.50, 0.60):
                mask = np.zeros((h + 2, w + 2), np.uint8)
                flags = (8 | cv2.FLOODFILL_MASK_ONLY
                         | cv2.FLOODFILL_FIXED_RANGE | (255 << 8))
                cv2.floodFill(blur, mask, (int(w * fx), int(h * fy)), 0,
                              (tol,) * 3, (tol,) * 3, flags)
                region = mask[1:-1, 1:-1]
                area = int(np.count_nonzero(region))
                if not (0.05 * h * w < area < 0.85 * h * w):
                    continue
                cs, _ = cv2.findContours(region, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
                if not cs:
                    continue
                c = max(cs, key=cv2.contourArea)
                (_, (rw, rh), _) = cv2.minAreaRect(c)
                if rw * rh <= 0:
                    continue
                cands.append((cv2.contourArea(c) / (rw * rh), area, c))

        good = [x for x in cands if x[0] >= 0.85]
        if good:
            rectangularity, area, c = max(good, key=lambda x: x[1])
            ap = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            quad = (ap.reshape(4, 2) if len(ap) == 4
                    else cv2.boxPoints(cv2.minAreaRect(c)))
            return quad, (f"flood-fill on tray floor, tolerance {tol} "
                          f"({100.0 * area / (h * w):.1f}% of frame, "
                          f"rectangularity {rectangularity:.2f})")

    # Strategy 2: fall back to edges. Works when the tray has a lit rim.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    sm = cv2.bilateralFilter(clahe, 9, 75, 75)
    frame_area = gray.shape[0] * gray.shape[1]
    kernel = np.ones((3, 3), np.uint8)

    exact, exact_area = None, 0          # true 4-gon: preferred
    fallback, fallback_area = None, 0    # minAreaRect of best blob: approximate

    for lo, hi in ((10, 40), (20, 60), (30, 90), (50, 130)):
        edges = cv2.dilate(cv2.Canny(sm, lo, hi), kernel, iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < 0.05 * frame_area or area > 0.98 * frame_area:
                continue
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx) and area > exact_area:
                exact, exact_area = approx.reshape(4, 2), area
            elif area > fallback_area:
                # Tilt makes the tray a trapezoid, so a rotated rect is only an
                # approximation -- but a usable one, and far better than failing.
                fallback = cv2.boxPoints(cv2.minAreaRect(c))
                fallback_area = area

    if exact is not None:
        note = "ok" if exact_area > 0.10 * frame_area else \
            f"LOW CONFIDENCE -- only {100*exact_area/frame_area:.1f}% of frame"
        return exact, note
    if fallback is not None:
        return fallback, (f"APPROXIMATE -- no clean 4-gon; using bounding rect of the "
                          f"largest contour ({100*fallback_area/frame_area:.1f}% of frame). "
                          f"Verify against the overlay, or pass --quad.")
    return None, "no candidate found"


def order_quad(q):
    """Order corners TL, TR, BR, BL so corner labels mean something."""
    q = np.array(q, dtype=np.float32)
    s, d = q.sum(axis=1), np.diff(q, axis=1).ravel()
    return np.array([q[np.argmin(s)], q[np.argmin(d)],
                     q[np.argmax(s)], q[np.argmax(d)]], dtype=np.float32)


# ----------------------------------------------------------------- measure ---

def patch_stats(gray, cx, cy, half=60):
    """Brightness + sharpness of a patch, clamped to frame bounds."""
    h, w = gray.shape
    x0, x1 = max(0, int(cx - half)), min(w, int(cx + half))
    y0, y1 = max(0, int(cy - half)), min(h, int(cy + half))
    p = gray[y0:y1, x0:x1]
    if p.size == 0:
        return None
    return {"mean": float(p.mean()),
            "std": float(p.std()),
            "sharp": float(cv2.Laplacian(p, cv2.CV_64F).var())}


def analyse(gray, quad):
    h, w = gray.shape
    quad = order_quad(quad)
    xs, ys = quad[:, 0], quad[:, 1]

    # margin from tray extremes to frame edge, as % of frame dimension
    margins = {
        "left":   100.0 * xs.min() / w,
        "right":  100.0 * (w - xs.max()) / w,
        "top":    100.0 * ys.min() / h,
        "bottom": 100.0 * (h - ys.max()) / h,
    }
    tray_area = cv2.contourArea(quad.astype(np.int32))
    coverage = 100.0 * tray_area / (w * h)

    centre = patch_stats(gray, xs.mean(), ys.mean())
    corners = {}
    for name, (cx, cy) in zip(("TL", "TR", "BR", "BL"), quad):
        # pull the sample slightly inward so it lands on tray floor, not the rim
        ix = cx + (xs.mean() - cx) * 0.15
        iy = cy + (ys.mean() - cy) * 0.15
        corners[name] = patch_stats(gray, ix, iy)

    return quad, margins, coverage, centre, corners


def report(margins, coverage, centre, corners, clipped_edges):
    print("=" * 68)
    print("TRAY FRAMING")
    print("=" * 68)

    print(f"  Tray covers {coverage:.1f}% of the frame")
    if coverage < 25:
        print("    ! Low -- most pixels are spent on things that are not the tray.")
        print("      Lower the camera or crop, or dice faces get few pixels each.")
    elif coverage > 85:
        print("    ! Very tight -- little room for a die resting against a wall.")

    print("\n  Margin to frame edge:")
    for k, v in margins.items():
        flag = ""
        if v < 0.5:
            flag = "  <-- TRAY IS CUT OFF (or touching the edge)"
        elif v < 3:
            flag = "  <-- very tight"
        print(f"    {k:<7}{v:6.1f}%{flag}")
    if clipped_edges:
        print(f"\n  !! Tray touches the frame edge at: {', '.join(clipped_edges)}")
        print("     Dice in that region will be clipped. Reposition before testing.")

    # Sharpness is only comparable between patches that both contain something
    # to resolve. An empty tray corner against a centre full of dice reports a
    # catastrophic "focus falloff" that is pure artefact -- the corner is flat
    # because it is empty, not because it is soft. Require real local variation
    # in BOTH patches before saying anything about focus.
    FLAT_STD = 8.0
    centre_flat = centre["std"] < FLAT_STD

    print("\n  Uniformity (tray corners vs centre):")
    print(f"    {'':<8}{'bright':>9}{'vs ctr':>9}{'sharp':>10}{'vs ctr':>9}")
    print(f"    {'centre':<8}{centre['mean']:9.1f}{'--':>9}{centre['sharp']:10.1f}{'--':>9}")
    worst_b, worst_s, judged = 100.0, 100.0, 0
    for name, st in corners.items():
        if st is None:
            print(f"    {name:<8}   (outside frame)")
            worst_b = 0
            continue
        rb = 100.0 * st["mean"] / centre["mean"] if centre["mean"] else 0
        worst_b = min(worst_b, rb)
        if st["std"] < FLAT_STD or centre_flat:
            print(f"    {name:<8}{st['mean']:9.1f}{rb:8.0f}%{st['sharp']:10.1f}"
                  f"{'  (flat)':>9}")
        else:
            rs = 100.0 * st["sharp"] / centre["sharp"] if centre["sharp"] else 0
            worst_s = min(worst_s, rs)
            judged += 1
            print(f"    {name:<8}{st['mean']:9.1f}{rb:8.0f}%{st['sharp']:10.1f}{rs:8.0f}%")

    if judged < 4:
        print(f"\n  Focus not judged for {4 - judged} corner(s): nothing there to")
        print("    resolve, so a sharpness number would be meaningless. To check")
        print("    focus across the tray, put a die in each corner and re-run.")

    print()
    if worst_b < 60:
        print(f"  ! Vignetting: dimmest corner is {worst_b:.0f}% of centre brightness.")
        print("    Light the tray more evenly, or the corners will read worse than")
        print("    the middle and dice against the walls will be the ones that fail.")
    if judged and worst_s < 40:
        print(f"  ! Focus falloff: softest corner is {worst_s:.0f}% of centre sharpness.")
        print("    The corners sit at a different distance than the centre; with the")
        print("    lens wide open the depth of field may not reach them. Refocus for")
        print("    the middle distance, or accept softer corners.")
    if worst_b >= 60 and (not judged or worst_s >= 40) and not clipped_edges:
        print("  Brightness uniformity OK -- corners are comparable to the centre."
              + ("" if judged == 4 else " (Focus unjudged; see above.)"))


def draw_grid(img, step=100):
    """Labelled coordinate grid, so tray corners can be read off by eye."""
    out = img.copy()
    h, w = out.shape[:2]
    for x in range(0, w, step):
        heavy = (x % (step * 5) == 0)
        cv2.line(out, (x, 0), (x, h), (0, 255, 255) if heavy else (60, 60, 60),
                 2 if heavy else 1)
        if heavy:
            cv2.putText(out, str(x), (x + 4, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2, cv2.LINE_AA)
    for y in range(0, h, step):
        heavy = (y % (step * 5) == 0)
        cv2.line(out, (0, y), (w, y), (0, 255, 255) if heavy else (60, 60, 60),
                 2 if heavy else 1)
        if heavy:
            cv2.putText(out, str(y), (6, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def draw(img, quad, corners):
    out = img.copy()
    q = quad.astype(np.int32)
    cv2.polylines(out, [q], True, (0, 255, 0), 3)

    # 5% inset "safe area" -- dice should stay inside this
    c = q.mean(axis=0)
    safe = (c + (q - c) * 1.05).astype(np.int32)
    cv2.polylines(out, [safe], True, (0, 200, 255), 1, cv2.LINE_AA)

    for name, (cx, cy) in zip(("TL", "TR", "BR", "BL"), quad):
        ix = int(cx + (quad[:, 0].mean() - cx) * 0.15)
        iy = int(cy + (quad[:, 1].mean() - cy) * 0.15)
        cv2.rectangle(out, (ix - 60, iy - 60), (ix + 60, iy + 60), (255, 0, 255), 2)
        cv2.putText(out, name, (ix - 55, iy - 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 0, 255), 2, cv2.LINE_AA)

    h, w = out.shape[:2]
    cv2.line(out, (w // 2, 0), (w // 2, h), (80, 80, 80), 1)
    cv2.line(out, (0, h // 2), (w, h // 2), (80, 80, 80), 1)
    return out


# -------------------------------------------------------------------- main ---

def capture(size):
    noir = "/usr/share/libcamera/ipa/rpi/vc4/imx219_noir.json"
    if "LIBCAMERA_RPI_TUNING_FILE" not in os.environ and os.path.exists(noir):
        os.environ["LIBCAMERA_RPI_TUNING_FILE"] = noir
    from picamera2 import Picamera2
    import time
    p = Picamera2()
    main = {"format": "RGB888"}
    if size:
        main["size"] = tuple(int(v) for v in size.lower().split("x"))
    p.configure(p.create_still_configuration(main=main))
    p.start()
    try:
        time.sleep(3)
        return p.capture_array("main")
    finally:
        p.stop(); p.close()


def main():
    ap = argparse.ArgumentParser(description="Dice tray framing check")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--capture", action="store_true")
    g.add_argument("--image")
    ap.add_argument("--size", default="1640x1232")
    ap.add_argument("--quad", help="manual tray corners x1,y1,...,x4,y4 (any order)")
    ap.add_argument("--save-quad", action="store_true",
                    help=f"persist the quad to {CONFIG} and reuse it next run")
    ap.add_argument("--grid", action="store_true",
                    help="write a labelled coordinate grid to read corners off")
    ap.add_argument("--overlay", default=None)
    args = ap.parse_args()

    if args.capture:
        img = capture(args.size)
        src = os.path.expanduser("~/dicecam-captures/framing_capture.jpg")
        cv2.imwrite(src, img)
        print(f"Captured -> {src}")
    else:
        src = os.path.expanduser(args.image)
        img = cv2.imread(src)
        if img is None:
            sys.exit(f"could not read {src}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"Frame: {w}x{h}")

    if args.grid:
        out = args.overlay or os.path.splitext(src)[0] + "_grid.jpg"
        cv2.imwrite(out, draw_grid(img))
        print(f"Coordinate grid -> {out}")
        print("Read the four tray-floor corners off the grid, then re-run with")
        print("  --quad x1,y1,x2,y2,x3,y3,x4,y4 --save-quad")
        return

    # Precedence: explicit --quad > saved calibration > auto-detect.
    #
    # Auto-detect is a convenience, not the intended path. A dice tray is dark
    # and matte by design and sits on a dark desk; measured contrast across the
    # whole scene was 14 grey levels, which defeats edge detection (it finds the
    # LEGO baseplate instead) and flood fill (the desk is within tolerance and
    # connects around the rim). Since the camera is fixed once mounted, the tray
    # quad is a one-time calibration -- store it and stop guessing.
    if args.quad:
        v = [int(x) for x in args.quad.split(",")]
        if len(v) != 8:
            sys.exit("--quad needs 8 comma-separated integers")
        quad = np.array(v, dtype=np.float32).reshape(4, 2)
        note = "manual (--quad)"
    elif os.path.exists(CONFIG):
        import json
        quad = np.array(json.load(open(CONFIG))["quad"], dtype=np.float32).reshape(4, 2)
        note = f"saved calibration ({CONFIG})"
    else:
        quad, note = find_tray_quad(gray)
        if quad is None:
            sys.exit(
                "Could not auto-detect the tray, which is expected for a dark tray\n"
                "on a dark surface -- do not fight it, just calibrate once:\n"
                "  1. python3 tray_framing_check.py --image FOO.jpg --grid\n"
                "  2. read the four tray-floor corners off the grid\n"
                "  3. re-run with --quad x1,y1,x2,y2,x3,y3,x4,y4 --save-quad\n"
                "Re-calibrate only when the mount moves."
            )
        note = "AUTO-DETECTED (" + note + ")"

    if args.save_quad:
        import json
        os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
        json.dump({"quad": quad.astype(int).tolist(), "frame": [w, h]},
                  open(CONFIG, "w"), indent=2)
        print(f"Saved calibration -> {CONFIG}")
    print(f"Tray detection: {note}\n")

    # Every number below is derived from the detected quad, so a wrong quad
    # yields a confident-looking report about the wrong object. Say so loudly:
    # the first version of this script cheerfully analysed a LEGO baseplate.
    shaky = any(k in note for k in ("APPROXIMATE", "LOW CONFIDENCE"))
    if shaky:
        print("!" * 68)
        print("!! Detection is not trustworthy -- CHECK THE OVERLAY before believing")
        print("!! anything below. If the green outline is not the tray, re-run with")
        print("!! --quad and the four corner coordinates.")
        print("!" * 68 + "\n")

    quad, margins, coverage, centre, corners = analyse(gray, quad)
    clipped = [k for k, v in margins.items() if v < 0.5]
    report(margins, coverage, centre, corners, clipped)

    if shaky:
        print("\n  (Reminder: detection was flagged unreliable -- verify the overlay.)")

    out = args.overlay or os.path.splitext(src)[0] + "_framing.jpg"
    cv2.imwrite(out, draw(img, quad, corners))
    print(f"\n  Overlay -> {out}")
    print("  Green = detected tray, orange = 5% safe area, magenta = corner samples.")
    print(f"  Designing around a {TARGET_TRAY_IN[0]}x{TARGET_TRAY_IN[1]}\" tray.")


if __name__ == "__main__":
    main()
