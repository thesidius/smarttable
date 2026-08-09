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


# --------------------------------------------------------------- camera pose ---
#
# A homography maps ONE plane. The dice do not live on one plane: a die's body
# centroid sits at the inradius above the floor and its top face at twice that,
# which is the entire basis of docs/geometric-face-reading.md. Predicting where
# the top face lands therefore needs the camera pose, not just a homography.
#
# Recovered from the calibration quad, which is a square of known size, via
# solvePnP. Two cautions learned by doing it:
#
#   * WINDING decides the sign. The first attempt put the camera 136 mm BELOW
#     the tray floor, because the object points were listed in the winding
#     opposite to the image points.
#
#   * The residual does not go to zero and that is expected. Swept across focal
#     lengths from 1200 to 9000 px the best reprojection error was 19.4 px --
#     irreducible, so the four clicked points are not exactly a square under any
#     pinhole model. On a 4608 px frame that is 0.4%, consistent with clicking
#     on a scaled-down preview, where one displayed pixel is ~6 real ones.
#     Treat a residual of this order as click precision; a much larger one means
#     the quad is not the square you think it is.

IMX708_F_PX = 3625.0     # self-calibrated from the square constraint; the
                         # datasheet-derived guesses were 3386 and 3547


def solve_pose(quad, square_mm, frame_size, f_px=IMX708_F_PX, quad_height_mm=0.0):
    """Camera pose from a square of known size. Returns a dict, or None.

    quad_height_mm lifts the calibration square above the floor -- the tray's
    45 degree skirt means the clicked corners may not be on the plane the dice
    rest on, and that offset propagates straight into every predicted position.
    """
    if cv2 is None:
        raise RuntimeError("tray_map needs OpenCV")
    q = order_quad(quad).astype(np.float64)
    W, H = frame_size
    h = float(square_mm) / 2.0
    # Reversed winding: image order is clockwise seen from the camera.
    obj = np.array([[-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0]],
                   np.float64)[::-1].copy()
    K = np.array([[f_px, 0, W / 2.0], [0, f_px, H / 2.0], [0, 0, 1]], np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, q, K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    cam = (-R.T @ tvec).ravel()
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    return {
        "rvec": rvec, "tvec": tvec, "K": K,
        "camera_mm": [float(cam[0]), float(cam[1]),
                      float(cam[2]) + float(quad_height_mm)],
        "height_mm": float(cam[2]) + float(quad_height_mm),
        "tilt_deg": float(np.degrees(np.arccos(
            abs(float((R.T @ np.array([0, 0, 1.0]))[2]))))),
        "reproj_px": float(np.linalg.norm(proj.reshape(-1, 2) - q, axis=1).mean()),
    }


def project(pose, pts_mm):
    """Tray-space millimetres (x, y, z above the floor) -> image pixels."""
    p = np.asarray(pts_mm, np.float64).reshape(-1, 3)
    out, _ = cv2.projectPoints(p, pose["rvec"], pose["tvec"], pose["K"], None)
    return out.reshape(-1, 2)
