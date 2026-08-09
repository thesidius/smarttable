#!/usr/bin/env python3
"""
tray_map.py -- image <-> tray-plane millimetres.

Phase 0 of docs/geometric-face-reading.md. That method predicts where a die's
top face is rather than searching for it, and every step of the prediction is
in millimetres: the inradius r comes off a pair of calipers, and the offset
from the silhouette centroid is r scaled by the viewing angle. None of that can
be applied to an image without a mapping between pixels and the tray plane.

The tray is planar and the camera is fixed, so one homography does it. It is
built from the four clicked corners, which is a measurement of where the tray
actually is -- not from the nominal mount height and tilt, which described the
previous camera at a previous position and are now wrong twice over.

The scale is NOT uniform across the frame. The camera looks at the tray
obliquely, so the far edge is compressed: measured on this rig, the far edge
spans 2186 px where the near edge spans 2759 px for the same physical length.
Anything that converts a pixel distance to millimetres has to do it locally,
which is what px_per_mm_at() is for.
"""

import numpy as np

try:
    import cv2
except ImportError:                                  # pragma: no cover
    cv2 = None

# Inner floor of the 3D-printed tray. A parameter, not a constant: change it
# and every millimetre downstream changes with it.
TRAY_LONG_MM = 190.5      # 7.5"
TRAY_SHORT_MM = 127.0     # 5"


def order_quad(quad):
    """Clicked in any order -> [top-left, top-right, bottom-right, bottom-left].

    Same convention the Pi's /framing uses, so the two agree about which corner
    is which.
    """
    q = np.asarray(quad, np.float32).reshape(4, 2)
    s = q.sum(axis=1)
    d = np.diff(q, axis=1).ravel()               # y - x
    return np.array([q[np.argmin(s)], q[np.argmin(d)],
                     q[np.argmax(s)], q[np.argmax(d)]], np.float32)


def _edge_lengths(o):
    top = np.linalg.norm(o[1] - o[0])
    bottom = np.linalg.norm(o[2] - o[3])
    left = np.linalg.norm(o[3] - o[0])
    right = np.linalg.norm(o[2] - o[1])
    return float(top), float(bottom), float(left), float(right)


def infer_orientation(quad):
    """Which physical tray side runs across the image? Decide, do not assume.

    Getting this backwards scales every millimetre by 1.5 and would not throw --
    it would just make every predicted offset wrong by half again, which is
    exactly the kind of silent error this project keeps finding.

    The camera is tilted, so the depth axis is FORESHORTENED: it shows fewer
    pixels per millimetre than the across-image axis. So whichever image axis
    has the lower pixels-per-millimetre is the one pointing away from the
    camera. Try both assignments and keep the self-consistent one.
    """
    o = order_quad(quad)
    top, bottom, left, right = _edge_lengths(o)
    horiz_px = (top + bottom) / 2.0
    vert_px = (left + right) / 2.0
    # Assignment A: the long side runs across the image.
    a = (horiz_px / TRAY_LONG_MM, vert_px / TRAY_SHORT_MM)
    # Assignment B: the long side runs into the image (away from the camera).
    b = (horiz_px / TRAY_SHORT_MM, vert_px / TRAY_LONG_MM)
    # The consistent one is the one where the depth (vertical) axis is the
    # compressed one, because that is what an oblique view does.
    if b[1] < b[0] and not (a[1] < a[0]):
        return TRAY_SHORT_MM, TRAY_LONG_MM, b
    if a[1] < a[0] and not (b[1] < b[0]):
        return TRAY_LONG_MM, TRAY_SHORT_MM, a
    # Both or neither are consistent: fall back on whichever is less lopsided,
    # and let the caller's die-size check catch it if this is wrong.
    ra, rb = max(a) / min(a), max(b) / min(b)
    return ((TRAY_LONG_MM, TRAY_SHORT_MM, a) if ra <= rb
            else (TRAY_SHORT_MM, TRAY_LONG_MM, b))


class TrayMap:
    """Homography between image pixels and tray-plane millimetres."""

    def __init__(self, quad, width_mm=None, height_mm=None):
        if cv2 is None:
            raise RuntimeError("tray_map needs OpenCV")
        self.quad = order_quad(quad)
        if width_mm is None or height_mm is None:
            width_mm, height_mm, _ = infer_orientation(quad)
        self.width_mm, self.height_mm = float(width_mm), float(height_mm)
        dst = np.array([[0, 0], [self.width_mm, 0],
                        [self.width_mm, self.height_mm], [0, self.height_mm]],
                       np.float32)
        self.H = cv2.getPerspectiveTransform(self.quad, dst)      # image -> mm
        self.Hinv = np.linalg.inv(self.H)                         # mm -> image

    def to_mm(self, pts):
        p = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.H).reshape(-1, 2)

    def to_px(self, pts):
        p = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.Hinv).reshape(-1, 2)

    def px_per_mm_at(self, pt):
        """Local scale at an image point. Not a constant across the tray."""
        mm = self.to_mm([pt])[0]
        a = self.to_px([mm, [mm[0] + 1.0, mm[1]], [mm[0], mm[1] + 1.0]])
        return (float(np.linalg.norm(a[1] - a[0])),
                float(np.linalg.norm(a[2] - a[0])))

    def inside(self, pt, margin_mm=0.0):
        x, y = self.to_mm([pt])[0]
        return (-margin_mm <= x <= self.width_mm + margin_mm and
                -margin_mm <= y <= self.height_mm + margin_mm)

    def describe(self):
        c = [self.quad.mean(axis=0)]
        near = self.to_px([[self.width_mm / 2, self.height_mm]])[0]
        far = self.to_px([[self.width_mm / 2, 0.0]])[0]
        return {
            "tray_mm": [self.width_mm, self.height_mm],
            "px_per_mm_centre": [round(v, 2) for v in self.px_per_mm_at(c[0])],
            "px_per_mm_near_edge": [round(v, 2) for v in self.px_per_mm_at(near)],
            "px_per_mm_far_edge": [round(v, 2) for v in self.px_per_mm_at(far)],
        }
