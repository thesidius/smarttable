#!/usr/bin/env python3
"""
tray_geometry.py -- what survived of the classical detector.

This file is the remnant of dice_detect.py, which found dice by local variance,
split touching ones by watershed, and counted them by distance-transform peaks.
All of that is gone; SAM2 on the workstation does the job. The numbers that
decided it, on the same 8-dice tray:

    classical, scattered dice ... reported 20, then 27, then 36, then 73
    classical, touching dice .... could not separate them at all (0 splits)
    SAM2, scattered ............. 8/8
    SAM2, tight pile ............ 7/7

It also cost 67 s of Pi 3 CPU per capture, which was the entire reason a
capture was not near-instant. Deleting it made /roll a crop-and-send.

What remains is the one piece that was never about detection: converting a tray
calibration measured at one resolution into another. That is still needed --
/roll crops to the tray and /framing samples inside it -- and it is still the
guard that turns a silent quarter-of-the-tray mask into an error.

The full history, including why local variance was the right idea and where it
broke down, is in docs/dice-reading.md and git.
"""

import numpy as np


def fit_quad_to_frame(quad, quad_frame, frame_shape):
    """Rescale a tray quad calibrated at one resolution to another.

    Tray calibration is in pixels, so it is only meaningful against the
    resolution it was measured at. Applying a 1640x1232 quad to a 3280x2464
    frame masks a quarter of the tray and silently yields garbage crops -- no
    error, just quietly wrong data. So convert explicitly.

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
