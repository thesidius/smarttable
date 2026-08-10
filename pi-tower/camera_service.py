#!/usr/bin/env python3
"""
camera_service.py -- thin HTTP camera head, runs ON the Pi.

The control panel lives on TheBeast (10.0.0.5); the camera does not. This
service is the seam between them, and its job is deliberately small: own the
camera, hand back a frame cropped to the tray. It used to also detect and count
the dice; that was measured wrong by up to 9x and cost 67 s of Pi 3 CPU per
capture, and SAM2 on the workstation replaced it. Nothing here is on the
accuracy path any more.

Design decision -- ONE camera configuration, never switched.

picamera2 can reconfigure between video and still modes, but doing that live,
per request, on a Pi 3 is slow and a reliable source of hangs. Instead the
camera runs continuously in a video configuration with two streams:

    main  largest allocatable RGB888  -- what /capture and /roll operate on
    lores  640x480 YUV420             -- what the MJPEG preview serves

The ISP produces both from the same frame, so the preview costs almost nothing
and never fights the capture path. The main stream is the largest size the
sensor offers that the Pi's contiguous memory will actually allocate -- see
Camera.detect() and the SENSORS table.

Endpoints:
    GET  /health                  service + camera state
    GET  /stream                  multipart MJPEG preview
    GET  /snapshot.jpg            single full-res JPEG (calibration clicking)
    POST /capture                 freeze a frame, returns its id
    POST /roll                    capture + crop; re-meters if the light drifted
    GET  /framing                 tray framing report
    GET  /calibration             read tray quad
    POST /calibration             write tray quad
    POST /lock  POST /unlock      freeze / release AE+AWB
    GET  /exposure                current settings + the sensor's real ranges
    POST /exposure                set exposure / gain / white balance by hand
    POST /autoexpose              meter the DICE, not the tray, then lock
    GET  /focus   POST /focus     read / pin the lens (Camera Module 3)
    POST /autofocus               sharpness sweep over the tray, then pin it
    GET  /file/<name>             serve a produced image
"""

import io
import json
import os
import sys
import threading
import time

# libcamera cannot tell a NoIR module from a standard V2 and defaults to the
# IR-cut tuning, which costs real numeral separability -- ~0.14 Otsu and 46 grey
# levels of numeral-to-body contrast on every frame, with no error to show for
# it.
#
# THE ENVIRONMENT VARIABLE NO LONGER WORKS. Setting LIBCAMERA_RPI_TUNING_FILE,
# whether from a launcher script or from Python, is silently undone: picamera2
# manages the tuning file itself and, when its constructor is called without a
# tuning= argument, does
#
#     os.environ.pop("LIBCAMERA_RPI_TUNING_FILE", None)  # Use default tuning
#
# (picamera2.py:337, v0.7.1+rpt20260609). It pops the variable BEFORE libcamera
# reads it, so the export loses every time no matter who wins the race. This was
# caught only because a post-reboot journal showed libcamera announcing
# "Using tuning file .../imx219.json" while /health cheerfully reported the NoIR
# tuning active -- /health was reporting the variable, which was set, rather
# than the outcome, which was wrong.
#
# Pass it through the supported API instead. picamera2 then sets the variable
# itself, immediately before opening the camera, and leaves it set.
TUNING_DIR = "/usr/share/libcamera/ipa/rpi/vc4"

# Per sensor: the tuning file, and the main-stream sizes to try LARGEST FIRST.
#
# imx219 is the Camera Module V2 NoIR, hence the _noir tuning -- without it
# libcamera assumes an IR-cut filter that module does not have.
#
# imx708 is Camera Module 3. The STANDARD module has an IR-cut filter, so it
# takes the plain tuning; imx708_noir.json and the _wide variants exist for the
# other SKUs and the sensor reports the same id for all of them, so this cannot
# be auto-detected -- override with DICECAM_TUNING if the module is swapped.
#
# Size ladder rather than one value because a Pi 3 may simply not have the
# contiguous memory for the full sensor: 4608x2592 RGB888 is 35.8 MB per buffer.
# Try for the resolution, fall back rather than refuse to start.
SENSORS = {
    "imx219": {"tuning": "imx219_noir.json", "sizes": [(3280, 2464)]},
    "imx708": {"tuning": "imx708.json",      "sizes": [(4608, 2592), (2304, 1296)]},
}
DEFAULT_SENSOR = {"tuning": None, "sizes": [(2304, 1296), (1920, 1080)]}

import cv2                                    # noqa: E402
import numpy as np                            # noqa: E402
from flask import Flask, Response, jsonify, request, send_from_directory  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tray_geometry import fit_quad_to_frame                           # noqa: E402

try:
    from picamera2 import Picamera2
except ImportError:
    sys.exit("picamera2 missing -> sudo apt install -y python3-picamera2")

# Resolution is worth paying for, which is why the SENSORS ladder tries the
# largest size first. Measured on the V2: at 1640x1232 the reader typed d20s
# unstably; at 3280x2464 it got 4/4 right, and the residual error became a
# consistent d12->d10 confusion rather than noise. It buys nothing for the CV
# silhouette -- it buys accuracy in the model, which is where reading happens.
#
# The preview is unaffected by any of this; it comes off lores.
#
# Only the WIDTH is fixed. The height is derived from the main stream's aspect
# ratio at open time, because lores shows the same field of view through a
# separate scaler: hardcoding 640x480 against a 16:9 sensor does not crop, it
# ANAMORPHICALLY SQUASHES -- the Camera Module 3 preview came out visibly
# narrowed horizontally while the captures were correct. 640x480 was right for
# the 4:3 V2 and silently wrong for everything else, which is the reason this
# is now computed rather than written down.
LORES_W = 640
CONFIG = os.path.expanduser("~/.config/dicecam/tray.json")
EXPOSURE = os.path.expanduser("~/.config/dicecam/exposure.json")

# Captures go to external storage when it is there.
#
# The SD card is 7.4 GB with a Desktop OS using 5 GB of it, which left ~100 MB
# of headroom -- and full-sensor frames are 1-2 MB each. It filled, and a full
# card does not announce itself: cv2.imwrite returns False, /capture still
# returned an id, and the failure surfaced later as "no such capture" on another
# screen entirely.
#
# FALL BACK rather than assume. /media/paul/pi_storage is a udisks user mount,
# so it may simply not be there after a reboot, and a service that dies because
# a USB stick was pulled is worse than one that quietly uses the SD card and
# says so. /health reports which is in use.
USB_WORK = "/media/paul/pi_storage/dicecam"
SD_WORK = os.path.expanduser("~/dicecam-web")


def _pick_work():
    override = os.environ.get("DICECAM_WORK")
    if override:
        return override, "env"
    parent = os.path.dirname(USB_WORK)
    if os.path.isdir(parent) and os.access(parent, os.W_OK):
        return USB_WORK, "usb"
    return SD_WORK, "sd-card"


WORK, WORK_SOURCE = _pick_work()
try:
    os.makedirs(WORK, exist_ok=True)
except OSError:
    WORK, WORK_SOURCE = SD_WORK, "sd-card (usb unwritable)"
    os.makedirs(WORK, exist_ok=True)

# Full-sensor frames are ~1-2 MB each and a roll writes several. Left alone this
# filled a 6.8 GB SD card to 100%, at which point cv2.imwrite silently wrote
# nothing while /capture still returned an id -- so the failure surfaced minutes
# later as "no such capture" on a completely unrelated screen.
# Retention stays even on the 59 GB stick -- unbounded growth is a slow leak,
# not a safe default -- but there is no reason to be stingy when the space is
# there.
KEEP_FILES = int(os.environ.get("DICECAM_KEEP", "600" if WORK_SOURCE == "usb" else "60"))
MIN_FREE_MB = 200

app = Flask(__name__)


def _prune():
    """Keep the newest KEEP_FILES artefacts. Cheap, and runs before each write."""
    try:
        entries = [(os.path.getmtime(os.path.join(WORK, f)), f)
                   for f in os.listdir(WORK)]
    except OSError:
        return
    for _, f in sorted(entries, reverse=True)[KEEP_FILES:]:
        try:
            os.remove(os.path.join(WORK, f))
        except OSError:
            pass


def _free_mb():
    st = os.statvfs(WORK)
    return (st.f_bavail * st.f_frsize) / (1024 * 1024)


def _cma_free_mb():
    """Free contiguous memory. This, not free RAM, is what the camera runs out of."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("CmaFree:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def _write_image(path, img, params=None):
    """Write, and FAIL LOUDLY if it did not happen.

    cv2.imwrite returns False on a full disk rather than raising. Trusting it
    is how a full card became a confusing error on another screen an hour later.
    """
    ok = cv2.imwrite(path, img, params or [])
    if not ok or not os.path.exists(path) or os.path.getsize(path) == 0:
        raise IOError(f"could not write {os.path.basename(path)} "
                      f"({_free_mb():.0f} MB free on the Pi)")
    return path


class Camera:
    """Owns the Picamera2 instance and a background frame grabber."""

    # Buffers for a full-sensor RGB888 main stream plus the RAW stream come out
    # of CMA, and want ~100 MB. At boot the desktop's KMS framebuffers get there
    # first: measured 49 s into a boot, free_cma was 3 MB of 256 MB, and the
    # allocation failed with "Cannot allocate memory". Ninety seconds later the
    # same machine had 217 MB free.
    #
    # So the contention is transient and waiting is the correct response, not
    # reserving more CMA (which permanently costs a 1 GB Pi general-purpose RAM
    # to fix a problem that lasts a minute) and not relying on systemd's
    # Restart= (which recovers, but only after the process has already died with
    # a traceback that looks like a hardware fault).
    OPEN_ATTEMPTS = 10
    OPEN_BACKOFF_S = 6

    # Explicit, and small. picamera2 defaults a video configuration to 6 buffers,
    # which at 4608x2592 RGB888 would ask CMA for 215 MB of the 256 MB it has --
    # the single biggest reason full resolution would fail to allocate. Nothing
    # here needs depth: the scene is stationary dice and the preview comes off
    # the lores stream.
    BUFFER_COUNT = 3

    @staticmethod
    def _lores_for(main_size):
        """Preview size matching the main stream's aspect. Even height for YUV420."""
        mw, mh = main_size
        h = int(round(LORES_W * mh / float(mw)))
        return (LORES_W, h - (h % 2))

    @staticmethod
    def detect():
        """(sensor_model, tuning_path, size_ladder) before opening anything.

        global_camera_info() reads the sensor id without taking the camera, so
        the configuration can be chosen for the hardware actually fitted rather
        than hardcoded -- which is what turned a module swap into an afternoon.
        """
        try:
            info = Picamera2.global_camera_info()
        except Exception:
            info = []
        if not info:
            raise RuntimeError("no camera detected by libcamera")
        model = str(info[0].get("Model", "")).lower()
        spec = SENSORS.get(model, DEFAULT_SENSOR)

        override = os.environ.get("DICECAM_TUNING")
        name = override or spec["tuning"]
        path = None
        if name:
            path = name if os.path.isabs(name) else os.path.join(TUNING_DIR, name)
            if not os.path.exists(path):
                print(f"[camera] tuning {path} missing; using libcamera default",
                      flush=True)
                path = None
        return model, path, list(spec["sizes"])

    def _open(self):
        """Open and configure the camera, waiting out transient CMA pressure.

        Returns (picam2, tuning_actually_in_force, main_size).
        """
        model, tuning, sizes = self.detect()
        self.sensor = model
        self.tuning_expected = tuning
        last = None
        for attempt in range(1, self.OPEN_ATTEMPTS + 1):
            for size in sizes:
                picam2 = None
                lores = self._lores_for(size)
                try:
                    # tuning= is the only mechanism that works; see SENSORS.
                    picam2 = Picamera2(tuning=tuning)
                    picam2.configure(picam2.create_video_configuration(
                        main={"size": tuple(size), "format": "RGB888"},
                        lores={"size": lores, "format": "YUV420"},
                        buffer_count=self.BUFFER_COUNT,
                        raw=None,        # the RAW stream is another full-size buffer
                    ))
                    if tuple(size) != tuple(sizes[0]):
                        print(f"[camera] {sizes[0][0]}x{sizes[0][1]} would not "
                              f"allocate; running at {size[0]}x{size[1]}", flush=True)
                    # Read the tuning back AFTER construction, not before.
                    # picamera2 sets this itself when handed a tuning file and
                    # pops it when not, so post-construction it reports what
                    # libcamera was actually given -- which is the thing the old
                    # env-var check only appeared to be checking.
                    return (picam2, os.environ.get("LIBCAMERA_RPI_TUNING_FILE"),
                            tuple(size), lores)
                except Exception as e:
                    last = e
                    try:
                        if picam2 is not None:
                            picam2.close()
                    except Exception:
                        pass

            # Every size failed. Retrying only helps if the failure is memory
            # pressure, which clears as the desktop finishes booting. Anything
            # else -- no camera on the bus, a bad cable -- will fail identically
            # forever, and spending 60s discovering that delays the honest error.
            if "allocate" not in str(last).lower():
                raise RuntimeError(f"camera present but could not be configured: {last}")
            if attempt == self.OPEN_ATTEMPTS:
                break
            print(f"[camera] open attempt {attempt}/{self.OPEN_ATTEMPTS} failed "
                  f"({last}); {_cma_free_mb():.0f} MB CMA free, retrying in "
                  f"{self.OPEN_BACKOFF_S}s", flush=True)
            time.sleep(self.OPEN_BACKOFF_S)
        raise RuntimeError(
            f"could not open the camera after {self.OPEN_ATTEMPTS} attempts over "
            f"~{self.OPEN_ATTEMPTS * self.OPEN_BACKOFF_S}s: {last}. "
            f"{_cma_free_mb():.0f} MB CMA free -- if this is 0 the framebuffers "
            f"took it, if it is large the camera itself is the problem.")

    def __init__(self):
        self.sensor = None
        self.tuning_expected = None
        (self.picam2, self.tuning_active, self.main_size,
         self.lores_size) = self._open()
        self.lock = threading.Lock()
        self.latest_main = None
        self.latest_lores = None
        self.frame_seq = 0
        self.meta = {}
        self.locked = False
        self.focus_pinned = None
        self.focus_sharpness = None
        self.mode = None                   # None | "auto-lock" | "manual"
        self.locked_controls = {}
        self.lock_lux = None
        self.running = True
        self.picam2.start()
        time.sleep(2)                      # let AE/AWB settle before first frame
        threading.Thread(target=self._loop, daemon=True).start()
        self._restore_lock()

    # --- focus (Camera Module 3 and anything else with a motorised lens) -----
    #
    # The V2 was fixed-focus, so focus never existed as a concept here. The CM3
    # has a voice-coil lens that defaults to CONTINUOUS autofocus, which is
    # actively wrong for this rig: it will re-hunt whenever the dice change,
    # so two captures of the same tray are focused differently. That breaks
    # comparability, and it would quietly destroy the template matching in
    # docs/geometric-face-reading.md, which assumes a fixed optical path.
    #
    # The camera and tray never move, so focus is a constant. Find it once, pin
    # it, persist it -- the same shape as the exposure lock, for the same reason.
    def has_focus(self):
        return "AfMode" in self.picam2.camera_controls

    def focus_state(self):
        if not self.has_focus():
            return {"supported": False}
        with self.lock:
            m = dict(self.meta)
        lo, hi, _ = self.picam2.camera_controls.get("LensPosition", (0.0, 10.0, 0.0))
        pos = m.get("LensPosition")
        return {"supported": True,
                "lens_position": round(float(pos), 3) if pos is not None else None,
                "focus_distance_m": (round(1.0 / float(pos), 3)
                                     if pos else None),
                "min": lo, "max": hi, "pinned": self.focus_pinned,
                "sharpness_at_pin": (round(self.focus_sharpness, 1)
                                     if self.focus_sharpness else None),
                "sharpness_now": (round(self._sharpness(), 1)
                                  if self.latest_main is not None else None)}

    def set_focus(self, lens_position):
        """Pin the lens. LensPosition is in DIOPTRES: 1/distance_in_metres."""
        lo, hi, _ = self.picam2.camera_controls.get("LensPosition", (0.0, 10.0, 0.0))
        p = max(lo, min(hi, float(lens_position)))
        self.picam2.set_controls({"AfMode": 0, "LensPosition": p})   # 0 = Manual
        self._await_lens(p)
        self.focus_pinned = p
        self.focus_sharpness = self._sharpness()
        self._persist()
        return self.focus_state()

    # Window used to judge focus: full resolution, centred on the tray. Full
    # resolution matters -- focus lives in the highest spatial frequencies, and
    # measuring on the downscaled preview would flatten the very peak being
    # searched for.
    FOCUS_WIN = 1200

    def _sharpness(self, box=None):
        """Variance of the Laplacian over a full-res window on the tray."""
        with self.lock:
            if self.latest_main is None:
                return None
            H, W = self.latest_main.shape[:2]
            if box:
                cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
            else:
                cx, cy = W // 2, H // 2
            half = self.FOCUS_WIN // 2
            x0, y0 = max(0, cx - half), max(0, cy - half)
            x1, y1 = min(W, cx + half), min(H, cy + half)
            crop = self.latest_main[y0:y1, x0:x1, 1].copy()   # green: sharpest channel
        return float(cv2.Laplacian(crop, cv2.CV_64F).var())

    def _wait_frames(self, n=2, timeout=6.0):
        """Block until n whole frames have been grabbed since now."""
        with self.lock:
            start = self.frame_seq
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                if self.frame_seq - start >= n:
                    return True
            time.sleep(0.02)
        return False

    def _await_lens(self, want, timeout=3.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                now = dict(self.meta).get("LensPosition")
            if now is not None and abs(float(now) - want) < 0.05:
                break
            time.sleep(0.03)
        # Two whole frames AFTER the lens reports arrival: one may already have
        # been mid-exposure while the lens was still moving.
        self._wait_frames(2)

    def focus_sweep(self, box=None):
        """Find the lens position that maximises sharpness ON THE TRAY, and pin it.

        Preferred over the sensor's own autofocus for the same reason
        highlight-priority metering is preferred over auto-exposure: the
        hardware optimises for a subject it chooses, and it does not know the
        subject is the dice. Measured, the AF cycle settled on 6.79 dioptres
        (~0.15 m) when the tray was ~0.21 m away -- plausible-looking, and soft
        where it mattered.

        Sharpness against lens position has a single peak, so a coarse sweep to
        bracket it followed by a fine sweep around the winner is enough, and is
        far more robust to a weak target than a hill-climb.
        """
        lo, hi, _ = self.picam2.camera_controls.get("LensPosition", (0.0, 15.0, 0.0))
        self.picam2.set_controls({"AfMode": 0})            # manual; we drive it
        trace, seen = [], {}

        def probe(p):
            p = round(max(lo, min(hi, p)), 3)
            if p in seen:                    # each position costs a lens move
                return seen[p]
            self.picam2.set_controls({"LensPosition": p})
            self._await_lens(p)
            v = self._sharpness(box)
            v = -1.0 if v is None else v
            seen[p] = v
            trace.append({"lens_position": p, "sharpness": round(v, 1)})
            return v

        step = (hi - lo) / 11.0
        best = max([lo + step * i for i in range(12)], key=probe)
        fine = [best + step * k / 3.0 for k in (-2, -1, 1, 2)]
        best = max([best] + [f for f in fine if lo <= f <= hi], key=probe)

        st = self.set_focus(best)
        st["sharpness"] = self.focus_sharpness
        st["trace"] = sorted(trace, key=lambda t: t["lens_position"])
        return st

    def autofocus(self):
        """Run one AF sweep, then pin whatever it found.

        Kept as the hardware path; focus_sweep() is the better default here.
        """
        if not self.has_focus():
            return {"supported": False}
        self.picam2.set_controls({"AfMode": 1})                      # 1 = Auto
        try:
            ok = self.picam2.autofocus_cycle()
        except Exception as e:
            return {"supported": True, "error": f"autofocus failed: {e}"}

        # WAIT for the grabber's metadata to catch up before reading the result.
        # autofocus_cycle() returns as soon as the sweep is done, but several
        # frames are still in flight and self.meta still holds the PRE-focus
        # lens position. Reading it immediately pinned the old value while
        # reporting the new one -- the response said lens_position 6.545 and
        # pinned 1.0, which is the same stale-read bug the exposure path has
        # _await_controls for.
        pos, stable = None, 0
        deadline = time.time() + 4.0
        while time.time() < deadline:
            with self.lock:
                now = dict(self.meta).get("LensPosition")
            if now is not None and pos is not None and abs(now - pos) < 1e-3:
                stable += 1
                if stable >= 3:
                    break
            else:
                stable = 0
            pos = now
            time.sleep(0.1)
        if pos is None:
            return {"supported": True, "error": "no lens position reported"}
        st = self.set_focus(pos)
        st["converged"] = bool(ok)
        return st

    def _restore_focus(self, saved):
        pos = saved.get("focus")
        if pos is None or not self.has_focus():
            return
        try:
            self.picam2.set_controls({"AfMode": 0, "LensPosition": float(pos)})
            self.focus_pinned = float(pos)
            dist = f" (~{1.0/float(pos):.2f} m)" if float(pos) else " (infinity)"
            print(f"[camera] restored pinned focus: {pos} dioptres{dist}",
                  flush=True)
        except Exception as e:
            print(f"[camera] could not restore focus: {e}", flush=True)

    def _restore_lock(self):
        """Re-apply the saved exposure lock on startup.

        Without this the lock silently dies on every service restart -- a fresh
        Camera object starts with AE/AWB free, and nothing says so except a
        banner nobody is looking at during a restart. That is how a run of
        captures ends up inconsistent: the segmentation threshold moves with the
        drifting exposure and detection quality wanders for no visible reason.

        A stale lock is still better than no lock (it is at least reproducible),
        and _lock_stale() will flag it loudly if the light has since changed.
        """
        if not os.path.exists(EXPOSURE):
            return
        try:
            saved = json.load(open(EXPOSURE))
            was = saved.get("sensor")
            if was and self.sensor and was != self.sensor:
                # Exposure values are meaningless across sensors: different
                # gain range, different tuning, different lens. Applying them
                # anyway would present as a badly broken camera rather than as
                # a stale setting, so drop them and start clean.
                print(f"[camera] discarding saved exposure: it was taken on "
                      f"{was}, this is {self.sensor}. Run auto-expose again.",
                      flush=True)
                os.remove(EXPOSURE)
                return
            self._restore_focus(saved)
            ctrl = dict(saved.get("controls") or {})
            if not ctrl:
                return            # focus-only save; there is no lock to restore
            if isinstance(ctrl.get("ColourGains"), list):
                ctrl["ColourGains"] = tuple(ctrl["ColourGains"])
            time.sleep(1.0)                 # let the pipeline accept controls
            self.picam2.set_controls(ctrl)
            self.locked = True
            self.mode = saved.get("mode", "auto-lock")
            self.locked_controls = saved["controls"]
            self.lock_lux = saved.get("lux")
            print(f"[camera] restored {self.mode} exposure from {EXPOSURE}: "
                  f"{ctrl.get('ExposureTime')}us gain {ctrl.get('AnalogueGain')}",
                  flush=True)
        except Exception as e:
            print(f"[camera] could not restore exposure lock: {e}", flush=True)

    def _loop(self):
        while self.running:
            try:
                req = self.picam2.capture_request()
                try:
                    main = req.make_array("main")
                    lores = req.make_array("lores")
                    meta = req.get_metadata()
                finally:
                    req.release()
                with self.lock:
                    self.latest_main = main
                    self.latest_lores = lores
                    self.meta = meta
                    self.frame_seq += 1
            except Exception as e:                     # keep the thread alive
                print(f"[camera] frame error: {e}", flush=True)
                time.sleep(0.5)
            # Cap the loop. A Pi 3 will happily spend every core on this and
            # then have nothing left for the detection request that matters.
            time.sleep(0.08)

    def frame(self):
        with self.lock:
            return None if self.latest_main is None else self.latest_main.copy()

    def preview_jpeg(self, quality=70):
        with self.lock:
            lores = None if self.latest_lores is None else self.latest_lores.copy()
        if lores is None:
            return None
        # I420 (Y,U,V), NOT YV12 (Y,V,U). Note that cv2.COLOR_YUV420p2BGR is an
        # *alias for the YV12 constant* (both are 99) -- the innocuous-looking
        # "420p" name does not mean "either planar 420", and reaching for it gets
        # the chroma planes backwards on picamera2's lores stream. The symptom is
        # red and blue transposed: preview read R 92.3 / B 20.7 against a
        # simultaneous main-stream frame at R 25.7 / B 114.5, green matching in
        # both. Green matching is what identifies it as a chroma swap rather than
        # a tuning or white-balance difference.
        bgr = cv2.cvtColor(lores, cv2.COLOR_YUV2BGR_I420)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    def controls(self):
        with self.lock:
            m = dict(self.meta)
        cg = m.get("ColourGains")
        return {
            "exposure_us": m.get("ExposureTime"),
            "analogue_gain": round(float(m.get("AnalogueGain", 0)), 3),
            "colour_gains": [round(float(cg[0]), 3), round(float(cg[1]), 3)] if cg else None,
            "lux": round(float(m["Lux"]), 1) if m.get("Lux") is not None else None,
            "locked": self.locked,
            "mode": self.mode,
            "lock_lux": round(float(self.lock_lux), 1) if self.lock_lux else None,
            "lock_stale": self._lock_stale(m.get("Lux")),
        }

    def _lock_stale(self, lux_now):
        """Has the room's light changed materially since we froze the settings?"""
        if not self.locked or not self.lock_lux or not lux_now:
            return None
        ratio = float(lux_now) / float(self.lock_lux)
        if ratio <= 1.5 and ratio >= 0.67:
            return None
        head = (f"Lighting changed since the settings were fixed "
                f"({self.lock_lux:.0f} -> {float(lux_now):.0f} lux).")
        if self.mode == "manual":
            # Manual values were chosen on purpose. Report the change, but do
            # not tell the operator to throw away a deliberate setting -- they
            # may well have dialled it in for exactly this reason.
            return (f"{head} These are MANUAL settings, so nothing has been "
                    f"changed for you. If the picture now looks wrong, adjust "
                    f"them; if it looks right, ignore this.")
        return (f"{head} The frozen exposure and white balance are for the OLD "
                f"light -- expect a colour cast and wrong brightness. Unlock, "
                f"let it re-converge, and re-lock -- or set values manually.")

    # A frame cannot expose for longer than it lasts, so ExposureTime is capped
    # by FrameDurationLimits regardless of what the sensor could do. Measured
    # here: camera_controls reports an ExposureTime maximum of 11.77 SECONDS,
    # but asking for 227 ms produced 47 ms, because the video configuration was
    # running ~21 fps. Nothing errors -- the picture just comes out dark.
    #
    # Long exposures are still worth ALLOWING, but not for the reason it is
    # tempting to assume. The obvious theory -- stationary dice mean motion blur
    # is free, so trade gain for time and get a cleaner image -- was measured
    # here and is FALSE at this light level:
    #
    #     47 ms @ gain 4.8   floor 131.6   noise 3.26   clipped 0.23%
    #    227 ms @ gain 1.0   floor 132.1   noise 3.33   clipped 0.25%
    #
    # Identical. The noise is photon-limited, not read- or gain-limited, and the
    # same total light arrives either way. So exposure time and analogue gain are
    # interchangeable, and the thing actually worth tuning is TOTAL brightness --
    # specifically, keeping the numerals below clipping. At gain 1.0: 110 ms
    # clipped nothing, 160 ms clipped 0.06%, 227 ms clipped 0.33%.
    #
    # The frame duration is raised to fit the requested exposure rather than
    # silently truncating it, because the truncation was invisible: a request
    # for 227 ms came back as 47 ms and simply looked dark. The cost is preview
    # frame rate, which is reported rather than hidden.
    MAX_FRAME_US = 2_000_000        # 0.5 fps floor; past here the UI feels broken

    def limits(self):
        """Control ranges, as the UI should present them.

        exposure_us reports the ACHIEVABLE maximum -- what this service will
        actually let you reach by raising the frame duration -- not the sensor's
        theoretical one. Reporting 11.77 s when a request for 0.23 s comes back
        clamped is worse than reporting nothing.
        """
        out = {}
        for name in ("ExposureTime", "AnalogueGain", "ColourGains"):
            c = self.picam2.camera_controls.get(name)
            if not c:
                continue
            lo, hi, default = c
            out[name] = {"min": lo, "max": hi, "default": default}
        if "ExposureTime" in out:
            sensor_max = out["ExposureTime"]["max"]
            out["ExposureTime"]["max"] = min(sensor_max, self.MAX_FRAME_US)
            out["ExposureTime"]["sensor_max"] = sensor_max
            out["ExposureTime"]["note"] = (
                "Exposures above the current frame duration raise it, which "
                "slows the preview to at most 1e6/exposure_us fps. The sensor "
                "itself would go to %.1f s." % (sensor_max / 1e6))
        return out

    def set_manual(self, exposure_us=None, analogue_gain=None, colour_gains=None):
        """Set exposure/gain/white-balance explicitly. Any subset; rest unchanged.

        Returns (applied, notes). APPLIED IS READ BACK FROM THE SENSOR, not
        echoed from the request: the sensor quantises ExposureTime to whole line
        times and clamps it to the frame duration, so asking for 60000us on a
        mode whose maximum is 47638us gets you 47638 with no error. Reporting
        the request back would make that invisible, and the operator would be
        left tuning a number the camera never used.
        """
        with self.lock:
            m = dict(self.meta)
        lim = self.limits()
        notes = []

        def clamp(name, value):
            c = lim.get(name)
            if not c:
                return value
            if value < c["min"] or value > c["max"]:
                notes.append(f"{name} {value:g} is outside the sensor's range "
                             f"{c['min']:g}..{c['max']:g} and was clamped")
                return max(c["min"], min(c["max"], value))
            return value

        ctrl = {"AeEnable": False, "AwbEnable": False}
        ctrl["ExposureTime"] = int(clamp(
            "ExposureTime",
            int(exposure_us) if exposure_us is not None else int(m.get("ExposureTime", 0))))
        ctrl["AnalogueGain"] = float(clamp(
            "AnalogueGain",
            float(analogue_gain) if analogue_gain is not None
            else float(m.get("AnalogueGain", 1.0))))
        cg = colour_gains if colour_gains is not None else m.get("ColourGains")
        if cg:
            ctrl["ColourGains"] = (float(clamp("ColourGains", float(cg[0]))),
                                   float(clamp("ColourGains", float(cg[1]))))

        # Make room in the frame for the exposure BEFORE asking for it. Without
        # this the exposure is truncated to the frame duration with no error --
        # a request for 227ms came back as 47ms and simply looked dark.
        want_us = ctrl["ExposureTime"]
        frame_us = max(want_us + 1000, 33333)
        ctrl["FrameDurationLimits"] = (frame_us, frame_us)
        if want_us > 100000:
            notes.append(f"{want_us/1000:.0f} ms exposure caps the preview at "
                         f"~{1e6/frame_us:.1f} fps. Fine for still dice; the "
                         f"live view will look choppy while you adjust.")

        self.picam2.set_controls(ctrl)

        # Wait for a frame taken UNDER the new settings before reading back.
        # set_controls is asynchronous and several frames are already in flight;
        # reading immediately returns the old values and looks like the request
        # was ignored.
        applied = self._await_controls(ctrl)

        # Report any remaining gap between request and reality. The clamp above
        # only catches values outside the ranges WE know about; libcamera has
        # its own, and the whole point of reading back is to notice when they
        # disagree with us rather than to trust our own arithmetic.
        got = applied.get("exposure_us")
        if got and abs(got - want_us) > max(100, want_us * 0.05):
            notes.append(f"asked for {want_us} us, sensor applied {got} us")

        self.locked = True
        self.mode = "manual"
        self.locked_controls = {k: (list(v) if isinstance(v, tuple) else v)
                                for k, v in ctrl.items()}
        self.lock_lux = applied.get("lux")
        self._persist()
        return applied, notes

    def _await_controls(self, want, timeout=3.0):
        """Poll metadata until the requested exposure/gain shows up, or give up."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                m = dict(self.meta)
            if (m.get("ExposureTime") is not None
                    and abs(int(m["ExposureTime"]) - want["ExposureTime"]) <= max(
                        50, want["ExposureTime"] * 0.02)
                    and abs(float(m.get("AnalogueGain", 0)) - want["AnalogueGain"]) < 0.05):
                break
            time.sleep(0.1)
        with self.lock:
            m = dict(self.meta)
        cg = m.get("ColourGains")
        return {
            "exposure_us": m.get("ExposureTime"),
            "analogue_gain": round(float(m.get("AnalogueGain", 0)), 3),
            "colour_gains": [round(float(cg[0]), 3), round(float(cg[1]), 3)] if cg else None,
            "lux": round(float(m["Lux"]), 1) if m.get("Lux") is not None else None,
        }

    # --- highlight-priority auto exposure -----------------------------------
    #
    # Ordinary AE meters the whole tray. Most of the tray is dark floor, so it
    # raises brightness until the FLOOR is mid-grey -- and the dice, which are
    # far brighter than the floor, blow out. Measured: auto-exposure settled at
    # a setting that clipped 0.23% of pixels, all of them on dice faces, and a
    # clipped numeral is unreadable at any resolution.
    #
    # So meter the dice instead: find the brightest exposure at which the dice
    # faces still stay below saturation, and freeze there.
    HL_TARGET = 245        # where the top of the dice histogram should sit (0-255)
    HL_PERCENTILE = 99.5   # ... measured here, not at the maximum
    HL_STEPS = 8
    # Gain is FIXED during the search, not inherited from whatever was set
    # before. Inheriting made the result depend on history: the same tray
    # metered to 55 ms because a gain of 4.0 happened to be left over, where
    # from a clean state it would have chosen ~220 ms. Both are correct
    # exposures; only one is reproducible.
    #
    # 1.0 is the default because it is the one value that is always available
    # and always means the same thing. It is NOT chosen for image quality --
    # measured on this rig, gain and exposure time are interchangeable, with
    # identical noise and clipping. Raising it buys preview frame rate and
    # nothing else, which is why it is a parameter rather than a constant.
    HL_GAIN = 1.0

    def _meter(self, box):
        """(top, floor, clipped_pct) for the metered region, or None.

        `top` is a high percentile of the WHOLE region, not of pixels selected
        as "brighter than the floor". The selecting version looked more precise
        and was wrong: the search's first step lands on a deliberately extreme
        exposure, the frame saturates, every pixel equals the floor, and
        "brighter than the floor" selects nothing -- so the probe that was
        supposed to report "far too bright" reported "no dice here" instead.

        A plain percentile has no such failure mode. The dice are far more than
        0.5% of the tray area, so the top 0.5% of it is dice regardless of
        exposure, and at saturation it correctly reads 255.

        The lores Y plane is the same ISP output as a capture, through the same
        tone curve, at 640x480 -- ample for a percentile, and far cheaper than a
        full-sensor frame when this runs eight times in a row.
        """
        with self.lock:
            lores = None if self.latest_lores is None else self.latest_lores.copy()
        if lores is None:
            return None
        w, h = self.lores_size
        y = lores[:h, :w].astype(np.float32)      # I420: Y plane first
        if box:
            x0, y0, x1, y1 = box
            y = y[y0:y1, x0:x1]
        if y.size < 100:
            return None
        return (float(np.percentile(y, self.HL_PERCENTILE)),
                float(np.median(y)),
                100.0 * float(np.mean(y >= 254)))

    # Bands for "still correctly exposed", on the same 99.5th-percentile
    # measure highlight metering targets. Direct evidence beats the lux proxy:
    # lux is the ISP's estimate of scene illuminance and moves with things that
    # do not matter here, while this measures the quantity actually at stake --
    # whether the dice faces are about to clip, or have sunk into the mud.
    OK_TOP_LO, OK_TOP_HI = 200.0, 251.0

    def exposure_health(self, box=None):
        """Is the current exposure still good for reading? (verdict, detail)."""
        m = self._meter(box)
        if m is None:
            return None, {"reason": "no frame"}
        top, floor, clipped = m
        if top - floor < 20:
            # Nothing bright in the tray. An empty tray is not a bad exposure,
            # and re-metering on it would lock to noise.
            return "empty", {"top": round(top, 1), "floor": round(floor, 1)}
        if top > self.OK_TOP_HI:
            v = "bright"
        elif top < self.OK_TOP_LO:
            v = "dark"
        else:
            v = "ok"
        return v, {"top": round(top, 1), "floor": round(floor, 1),
                   "clipped_pct": round(clipped, 3),
                   "band": [self.OK_TOP_LO, self.OK_TOP_HI]}

    def autoexpose(self, box=None, gain=None, _retry=False):
        """Brightest exposure that keeps the dice faces unsaturated, then lock.

        Bisection rather than arithmetic: output brightness is NOT linear in
        exposure time, because the ISP applies a tone curve. Bisection only
        needs the relationship to be monotonic, which it is.
        """
        lo, hi = 200, int(min(self.MAX_FRAME_US,
                              self.picam2.camera_controls["ExposureTime"][1]))
        gain = float(self.HL_GAIN if gain is None else gain)

        # Let white balance converge while the exposure search runs, then freeze
        # it with the result -- otherwise a good exposure gets locked alongside
        # whatever colour happened to be set beforehand.
        self.picam2.set_controls({"AeEnable": False, "AwbEnable": True})

        def probe(us):
            frame_us = max(us + 1000, 33333)
            self.picam2.set_controls({"ExposureTime": us, "AnalogueGain": gain,
                                      "FrameDurationLimits": (frame_us, frame_us)})
            self._await_controls({"ExposureTime": us, "AnalogueGain": gain})
            time.sleep(0.35)                       # let the tone curve settle
            return self._meter(box)

        # Is there anything in the tray to meter? Ask once, at the exposure
        # already in force -- which is known-reasonable -- rather than at a
        # search extreme where the answer would be meaningless either way.
        with self.lock:
            here = int(self.meta.get("ExposureTime") or 110000)
        m0 = probe(here)
        if m0 is None:
            return None, ["no frame available"]
        if m0[0] - m0[1] < 20:
            return None, ["Nothing bright in the tray to meter -- the tray "
                          "looks empty. Highlight metering needs the dice in "
                          "place. Roll them in and run it again."]

        trace, best = [], None
        for _ in range(self.HL_STEPS):
            mid = (lo + hi) // 2
            m = probe(mid)
            if m is None:
                return None, ["no frame available"]
            top, floor, clipped = m
            trace.append({"exposure_us": mid, "dice_top": round(top, 1),
                          "floor": round(floor, 1),
                          "clipped_pct": round(clipped, 3)})
            if top <= self.HL_TARGET:
                best = mid                          # headroom left; try brighter
                lo = mid + 1
            else:
                hi = mid - 1
            if lo > hi:
                break

        notes = []
        if best is None:
            # Even the dimmest exposure saturates: the scene itself is too
            # bright for this gain. Say so rather than locking something wrong.
            best = 200
            notes.append("Even the shortest exposure blows out the dice. Lower "
                         "the analogue gain or the lighting, then re-run.")
        else:
            at_best = [t for t in trace if t["exposure_us"] == best]
            got = at_best[0]["dice_top"] if at_best else self.HL_TARGET
            # Test the OUTCOME, not where the search happened to stop.
            # Requiring best >= ceiling-1 never fired: bisection converges
            # near the ceiling, not exactly on it (1937228 of 2000000), so a
            # tray that reached only 146 of 245 was accepted as final.
            # Falling short of the target at all means brightness ran out.
            if got < self.HL_TARGET - 15:
                # Out of exposure, not out of options. The ceiling exists to
                # keep the preview usable, not because longer is better --
                # measured on this rig, exposure time and analogue gain are
                # interchangeable, with identical noise and clipping. So trade
                # into gain rather than handing back an underexposed frame.
                #
                # This is what a dim room actually looks like: metering a tray
                # lit only by desk LEDs ran to the full 2 s ceiling at gain 1.0
                # and still reached only about half the target brightness.
                gmax = self.picam2.camera_controls.get(
                    "AnalogueGain", (1.0, 8.0, 1.0))[1]
                want = gain * (self.HL_TARGET / max(1.0, got))
                new_gain = min(gmax, want)
                if new_gain > gain * 1.05 and not _retry:
                    notes.append(
                        f"Exposure ran out at {best/1000:.0f} ms and the dice "
                        f"only reached {got:.0f} of {self.HL_TARGET}. Raising "
                        f"gain {gain:.2f}x -> {new_gain:.2f}x and re-metering.")
                    r, more = self.autoexpose(box=box, gain=new_gain, _retry=True)
                    return r, (notes + more) if r else notes + more
                notes.append(
                    f"At maximum exposure AND gain {gain:.2f}x the dice reach "
                    f"only {got:.0f} of {self.HL_TARGET}. The scene is too dark "
                    f"to expose properly -- add light.")

        applied, more = self.set_manual(exposure_us=best, analogue_gain=gain)
        self.mode = "auto-highlight"
        self._persist()
        return {"applied": applied, "trace": trace}, notes + more

    def _persist(self):
        try:
            os.makedirs(os.path.dirname(EXPOSURE), exist_ok=True)
            json.dump({"controls": self.locked_controls, "lux": self.lock_lux,
                       "mode": self.mode, "focus": self.focus_pinned,
                       "focus_sharpness": self.focus_sharpness,
                       # Stamped so a saved state is never applied to a
                       # different sensor -- gain ranges, tuning and lens are
                       # all sensor-specific, and silently restoring a V2 lock
                       # onto a CM3 would look like a broken camera.
                       "sensor": self.sensor,
                       "saved": time.strftime("%Y-%m-%dT%H:%M:%S")},
                      open(EXPOSURE, "w"), indent=2)
        except OSError as e:
            print(f"[camera] could not persist exposure settings: {e}", flush=True)

    def unlock_exposure(self):
        self.picam2.set_controls({"AeEnable": True, "AwbEnable": True})
        self.locked = False
        self.mode = None
        self.locked_controls = {}
        self.lock_lux = None
        # Remove the saved lock too, or the next restart silently re-locks to
        # settings the operator deliberately released.
        try:
            if os.path.exists(EXPOSURE):
                os.remove(EXPOSURE)
        except OSError:
            pass

    def lock_exposure(self):
        """Freeze AE/AWB at whatever they have converged to."""
        with self.lock:
            m = dict(self.meta)
        if not m:
            return None
        cg = m.get("ColourGains")
        ctrl = {"AeEnable": False, "AwbEnable": False,
                "ExposureTime": int(m.get("ExposureTime", 0)),
                "AnalogueGain": float(m.get("AnalogueGain", 1.0))}
        if cg:
            ctrl["ColourGains"] = (float(cg[0]), float(cg[1]))
        self.picam2.set_controls(ctrl)
        self.locked = True
        self.mode = "auto-lock"
        self.locked_controls = {k: (list(v) if isinstance(v, tuple) else v)
                                for k, v in ctrl.items()}
        # Remember how bright the room was when we froze. A lock is only valid
        # for the light it was taken under: locking under purple LEDs and then
        # switching on a lamp leaves the camera applying a 3.28x blue gain to a
        # scene that no longer needs it, and the result is a heavy false colour
        # cast that looks like a camera fault rather than a stale setting.
        self.lock_lux = m.get("Lux")
        self._persist()
        return self.locked_controls


def _camera_fault_hint():
    """Turn a failed camera open into something actionable.

    The kernel already knows why, and says so precisely -- but in dmesg, which
    is not where anyone looks when a web UI says the Pi is down.
    """
    try:
        import subprocess
        out = subprocess.run(["dmesg"], capture_output=True, text=True,
                             timeout=5).stdout
    except Exception:
        return ("Could not read dmesg. Check the ribbon cable and run "
                "'rpicam-hello --list-cameras' on the Pi.")
    probe = [l for l in out.splitlines()
             if "failed to read chip id" in l or "probe with driver" in l]
    if probe:
        return ("The sensor driver loaded but got no answer over I2C -- "
                "error -5 is EIO, meaning nothing is responding on the camera "
                "bus. This is a cable or connector fault, not software: "
                "reseat the ribbon at BOTH ends, check it is the right way "
                "round, and power-cycle. Kernel said: " + probe[-1].strip())
    if "imx" not in out and "unicam" not in out:
        return ("No camera was detected at boot. CSI cameras are probed by "
                "firmware at boot, so a camera connected while running will "
                "not appear -- reboot. If it still does not appear, reseat "
                "the ribbon at both ends.")
    return "Camera present in dmesg but could not be opened; see the journal."


# Serve in a DEGRADED state rather than dying when there is no camera.
#
# Dying meant systemd restarted us every ~70 s, each cycle burning ~10 s of Pi 3
# CPU on a retry loop that could not possibly succeed -- a CSI sensor does not
# appear without a reboot. Worse, the control panel showed "Pi unreachable",
# which is actively misleading: the Pi is fine, reachable, and the only broken
# thing is a ribbon cable. Staying up to say exactly that is more useful than
# exiting, and it costs nothing.
cam, CAMERA_ERROR, CAMERA_HINT = None, None, None
try:
    cam = Camera()
except Exception as _e:
    CAMERA_ERROR = str(_e)
    CAMERA_HINT = _camera_fault_hint()
    print(f"[camera] NOT AVAILABLE: {CAMERA_ERROR}\n[camera] {CAMERA_HINT}",
          flush=True)


def _no_camera():
    return jsonify({"error": "no camera", "detail": CAMERA_ERROR,
                    "hint": CAMERA_HINT}), 503


# ------------------------------------------------------------- calibration ---

def load_quad():
    if not os.path.exists(CONFIG):
        return None, None
    cfg = json.load(open(CONFIG))
    return np.array(cfg["quad"], np.float32).reshape(4, 2), cfg.get("frame")


def _calibration_state(frame_size=None):
    """Is there a tray calibration, and is it USABLE at the current frame size?

    "The file exists" is not the same question. After the Camera Module 3 swap
    the saved quad was still there and /health said calibrated, while /roll
    refused every request because a 4:3 quad cannot be rescaled onto a 16:9
    frame. A green pill telling the operator not to do the one thing they must
    do is worse than no pill at all.
    """
    quad, qframe = load_quad()
    if quad is None:
        return False, None
    if frame_size is None:
        return True, None
    try:
        fit_quad_to_frame(quad, qframe or list(frame_size),
                          (frame_size[1], frame_size[0]))
        return True, None
    except ValueError as e:
        return False, str(e)


@app.get("/calibration")
def get_calibration():
    quad, frame = load_quad()
    if quad is None:
        return jsonify({"calibrated": False, "quad": None, "frame": None})
    try:
        saved = json.load(open(CONFIG))
    except Exception:
        saved = {}
    extra = {k: saved.get(k) for k in ("feature", "square_mm", "height_mm", "saved")}
    if extra["feature"] is None:
        # Pre-dates the field. Say so rather than defaulting, because a wrong
        # guess here silently rescales everything downstream.
        extra["warning"] = ("this calibration predates the feature field, so "
                            "what was clicked is unknown -- recalibrate")
    return jsonify({"calibrated": True, **extra, "quad": quad.astype(int).tolist(),
                    "frame": frame})


# What the four corners ARE, not just where they are.
#
# The tray has two obvious square features and the calibration used to record
# only pixels, so every consumer had to guess which one had been clicked. That
# guess was wrong three times in one session: once read as the 95 mm floor when
# it was the walls, once the reverse, and once silently carried across a
# re-click that switched features. Each time it scaled every derived millimetre
# and nothing errored.
#
# Geometric face reading needs the plane the dice REST on, so the floor is the
# right feature for it -- the wall square sits ~10 mm higher, and that offset
# goes straight into every predicted position.
FEATURES = {
    "floor": {"square_mm": 95.0, "height_mm": 0.0,
              "desc": "flat floor, inside the 45 degree skirt"},
    "walls": {"square_mm": 115.0, "height_mm": 10.0,
              "desc": "top of the skirt where it meets the vertical wall"},
}


@app.post("/calibration")
def set_calibration():
    body = request.get_json(force=True)
    quad = body.get("quad")
    if not quad or len(quad) != 4 or any(len(p) != 2 for p in quad):
        return jsonify({"error": "quad must be 4 [x,y] pairs"}), 400
    feature = body.get("feature", "floor")
    if feature not in FEATURES:
        return jsonify({"error": f"feature must be one of {list(FEATURES)}"}), 400
    spec = FEATURES[feature]
    frame = body.get("frame") or list(cam.main_size)
    rec = {"quad": [[int(p[0]), int(p[1])] for p in quad],
           "frame": [int(frame[0]), int(frame[1])],
           "feature": feature,
           # Overridable: these are this tray's measurements, not constants.
           "square_mm": float(body.get("square_mm") or spec["square_mm"]),
           "height_mm": float(body.get("height_mm", spec["height_mm"])),
           "saved": time.strftime("%Y-%m-%dT%H:%M:%S")}
    os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
    json.dump(rec, open(CONFIG, "w"), indent=2)
    return jsonify({"saved": True, "path": CONFIG, **rec})


# ------------------------------------------------------------------ frames ---

@app.get("/health")
def health():
    if cam is None:
        # Answer, loudly and specifically. The control panel's "Pi unreachable"
        # banner was the wrong diagnosis for a ribbon cable.
        return jsonify({
            "ok": False, "camera": False,
            "error": CAMERA_ERROR, "hint": CAMERA_HINT,
            "calibrated": _calibration_state()[0],
            "work_dir": WORK, "work_source": WORK_SOURCE,
            "free_mb": round(_free_mb()),
        })
    f = cam.frame()
    _cal_ok, _cal_why = _calibration_state(cam.main_size)
    return jsonify({
        "ok": f is not None,
        "camera": True,
        "sensor": cam.sensor,
        "main_size": list(cam.main_size),
        "focus": cam.focus_state(),
        "lores_size": list(cam.lores_size),
        # cam.tuning_active is read back from picamera2 AFTER it opened the
        # camera, so it is the file libcamera was actually handed. The previous
        # version reported the environment variable we had set ourselves, which
        # reported success in precisely the case that silently failed -- it read
        # "NoIR active" for weeks while libcamera logged imx219.json.
        "tuning_file": cam.tuning_active or "(libcamera default)",
        "tuning_expected": cam.tuning_expected,
        "tuning_ok": cam.tuning_active == cam.tuning_expected,
        "tuning_warning": (None if cam.tuning_active == cam.tuning_expected else
                           f"libcamera loaded {cam.tuning_active or 'its default'} "
                           f"but this sensor ({cam.sensor}) should use "
                           f"{cam.tuning_expected}. Captures are usable but the "
                           f"colour pipeline is wrong, which costs numeral "
                           f"contrast and does not announce itself."),
        "cma_free_mb": round(_cma_free_mb()),
        "controls": cam.controls(),
        "calibrated": _cal_ok,
        "calibration_warning": _cal_why,
        "work_dir": WORK,
        "work_source": WORK_SOURCE,
        "free_mb": round(_free_mb()),
        "disk_warning": (None if _free_mb() >= MIN_FREE_MB else
                         f"only {_free_mb():.0f} MB free -- captures will fail"),
    })


@app.get("/stream")
def stream():
    if cam is None:
        return _no_camera()
    def gen():
        while True:
            jpg = cam.preview_jpeg()
            if jpg:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                       + jpg + b"\r\n")
            time.sleep(0.12)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/snapshot.jpg")
def snapshot():
    if cam is None:
        return _no_camera()
    f = cam.frame()
    if f is None:
        return jsonify({"error": "no frame yet"}), 503
    ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.post("/capture")
def capture():
    if cam is None:
        return _no_camera()
    f = cam.frame()
    if f is None:
        return jsonify({"error": "no frame yet"}), 503
    _prune()
    if _free_mb() < MIN_FREE_MB:
        return jsonify({"error": f"only {_free_mb():.0f} MB free on the Pi -- "
                                 f"refusing to capture rather than write a truncated "
                                 f"frame that fails later"}), 507
    cid = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(WORK, f"{cid}.jpg")
    try:
        _write_image(path, f, [cv2.IMWRITE_JPEG_QUALITY, 95])
    except IOError as e:
        return jsonify({"error": str(e)}), 507
    json.dump(cam.controls(), open(os.path.join(WORK, f"{cid}_controls.json"), "w"))
    return jsonify({"id": cid, "file": f"{cid}.jpg",
                    "width": f.shape[1], "height": f.shape[0],
                    "controls": cam.controls()})


@app.post("/lock")
def do_lock():
    if cam is None:
        return _no_camera()
    c = cam.lock_exposure()
    return (jsonify({"locked": True, "controls": c}) if c
            else (jsonify({"error": "no metadata yet"}), 503))


@app.post("/unlock")
def do_unlock():
    if cam is None:
        return _no_camera()
    cam.unlock_exposure()
    return jsonify({"locked": False})


def _metering_box():
    """Tray region in lores coordinates, or None if uncalibrated.

    Metering has to be confined to the tray. Without it the search meters the
    desk, which is the very mistake highlight metering exists to fix.
    """
    quad, qframe = load_quad()
    if quad is None or cam is None:
        return None
    try:
        w, h = cam.lores_size
        sx = w / float(qframe[0] if qframe else cam.main_size[0])
        sy = h / float(qframe[1] if qframe else cam.main_size[1])
        xs, ys = quad[:, 0] * sx, quad[:, 1] * sy
        return (max(0, int(xs.min())), max(0, int(ys.min())),
                min(w, int(xs.max())), min(h, int(ys.max())))
    except Exception:
        return None


@app.post("/autoexpose")
def do_autoexpose():
    """Expose for the DICE, not for the tray, then lock.

    Ordinary auto-exposure meters the whole frame. The tray floor is most of
    that frame and it is dark, so AE drives brightness up until the floor is
    mid-grey -- and the dice, far brighter, saturate. Measured: AE settled on a
    setting that clipped 0.23% of pixels, essentially all of them on dice faces.

    This searches for the brightest exposure at which the top of the DICE
    histogram still sits below saturation, and locks there.
    """
    if cam is None:
        return _no_camera()
    box = _metering_box()
    body = request.get_json(silent=True) or {}
    try:
        result, notes = cam.autoexpose(box=box, gain=body.get("analogue_gain"))
    except Exception as e:
        return jsonify({"error": f"auto-exposure failed: {e}"}), 500
    if result is None:
        return jsonify({"error": notes[0] if notes else "auto-exposure failed"}), 400
    return jsonify({**result, "notes": notes, "metered_box": box,
                    "controls": cam.controls()})


@app.get("/focus")
def get_focus():
    if cam is None:
        return _no_camera()
    return jsonify(cam.focus_state())


@app.post("/focus")
def put_focus():
    """Pin the lens. {"lens_position": 4.8}  -- dioptres, i.e. 1/metres.

    The tray floor sits ~207 mm below the lens, so ~4.8 is the expected value.
    """
    if cam is None:
        return _no_camera()
    if not cam.has_focus():
        return jsonify({"error": "this camera has no motorised lens"}), 400
    b = request.get_json(silent=True) or {}
    if b.get("lens_position") is None:
        return jsonify({"error": "lens_position (dioptres) required"}), 400
    try:
        return jsonify(cam.set_focus(b["lens_position"]))
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"bad value: {e}"}), 400


@app.post("/autofocus")
def do_autofocus():
    """Focus on the DICE, then pin. {"mode": "hardware"} for the sensor's own AF.

    Default is a sharpness sweep over the tray rather than the sensor's
    autofocus, for the same reason /autoexpose meters the dice rather than the
    frame: the hardware optimises for a subject it picks itself. Measured, its
    AF cycle chose 6.79 dioptres for a tray that was not at that distance, and
    the captures were soft.
    """
    if cam is None:
        return _no_camera()
    if not cam.has_focus():
        return jsonify({"error": "this camera has no motorised lens"}), 400
    body = request.get_json(silent=True) or {}
    if body.get("mode") == "hardware":
        r = cam.autofocus()
        return (jsonify(r), 500) if r.get("error") else jsonify(r)
    # Focus where the tray is, if we know; otherwise the middle of the frame,
    # which is where it is anyway on this mount.
    box = None
    quad, qframe = load_quad()
    if quad is not None:
        try:
            q, _ = fit_quad_to_frame(quad, qframe or list(cam.main_size),
                                     (cam.main_size[1], cam.main_size[0]))
            xs, ys = q[:, 0], q[:, 1]
            box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        except ValueError:
            box = None          # calibration is for another frame size
    return jsonify(cam.focus_sweep(box=box))


@app.get("/exposure")
def get_exposure():
    """Current settings plus the sensor's real ranges, for a UI to bound itself."""
    if cam is None:
        return _no_camera()
    return jsonify({"controls": cam.controls(), "limits": cam.limits()})


@app.post("/exposure")
def set_exposure():
    """Set exposure / gain / white balance by hand. Any subset.

        {"exposure_us": 30000, "analogue_gain": 4.0, "colour_gains": [1.5, 1.8]}

    Auto-exposure converges to a picture that looks reasonable, which is not the
    same as one that reads well: it meters the whole tray, most of which is dark
    floor, and pushes gain up until that is mid-grey -- blowing out the numerals,
    which are the only part that matters. Being able to set a shorter exposure
    and a lower gain by hand is how you trade a dark tray for numerals that are
    not clipped.

    Returns what the sensor ACTUALLY applied, not what was asked for.
    """
    if cam is None:
        return _no_camera()
    b = request.get_json(silent=True) or {}
    cg = b.get("colour_gains")
    if cg is not None and (not isinstance(cg, (list, tuple)) or len(cg) != 2):
        return jsonify({"error": "colour_gains must be [red, blue]"}), 400
    try:
        applied, notes = cam.set_manual(
            exposure_us=b.get("exposure_us"),
            analogue_gain=b.get("analogue_gain"),
            colour_gains=cg)
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"bad value: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"could not apply settings: {e}"}), 500

    requested = {k: b[k] for k in ("exposure_us", "analogue_gain", "colour_gains")
                 if b.get(k) is not None}
    return jsonify({"applied": applied, "requested": requested,
                    "notes": notes, "controls": cam.controls()})


# ----------------------------------------------------------------- capture ---

@app.post("/roll")
def roll():
    """Capture and crop to the tray. Segmentation and reading happen remotely.

    This used to also run the classical detector to produce an INDEPENDENT
    COUNT, on the reasoning that with no known dice set to check against, a
    second opinion was the only thing that could catch the reader silently
    omitting a die. The reasoning was sound; the detector was not. Measured on
    an 8-dice tray it reported 20, 27, 36 and finally 73 -- a "second opinion"
    that is wrong by 9x does not catch omissions, it manufactures false alarms
    until the alarm is ignored.

    SAM2 does that job now and got 8/8 scattered and 7/7 in a tight pile. It
    also does it on the workstation's GPU in ~3 s, where the classical pass cost
    67 s of Pi 3 CPU -- it was the entire reason a capture was not near-instant.

    So: crop and hand over the image. Nothing here is on the accuracy path any
    more, which is why nothing here needs to be fast or clever.
    """
    if cam is None:
        return _no_camera()
    body = request.get_json(silent=True) or {}

    # Re-meter when the light has moved, BEFORE capturing.
    #
    # A locked exposure is only valid for the light it was taken under, and a
    # session drifts: metered at 240 lux, by the next roll the room was at 87
    # and the frame came back at mean level 31. The reverse also happened --
    # light doubled and a capture came out 2.6% clipped, a hundred times the
    # 0.02% highlight metering achieves. Both were silent.
    #
    # Triggered on measured brightness rather than on the lux reading, because
    # the question is not "has the light changed" but "are the dice still
    # exposed properly", and one lores frame answers that directly.
    remeter = body.get("remeter", "auto")
    remetered = None
    if cam.locked and remeter is not False:
        box = _metering_box()
        verdict, detail = cam.exposure_health(box)
        if remeter == "force" or verdict in ("bright", "dark"):
            r, notes = cam.autoexpose(box=box)
            remetered = {"was": verdict, "detail": detail,
                         "applied": (r or {}).get("applied"), "notes": notes}
            print(f"[camera] re-metered before roll: dice top {detail.get('top')} "
                  f"was {verdict}", flush=True)
        else:
            remetered = False

    img = cam.frame()
    if img is None:
        return jsonify({"error": "no frame yet"}), 503

    quad, qframe = load_quad()
    if quad is None:
        return jsonify({"error": "not calibrated -- set the tray quad first"}), 400
    try:
        quad, _ = fit_quad_to_frame(quad, qframe or list(cam.main_size), img.shape)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Crop to the tray's bounding box. Everything outside is desk clutter that
    # only gives the reader more chances to hallucinate a die.
    h, w = img.shape[:2]
    xs, ys = quad[:, 0], quad[:, 1]
    x0, y0 = max(0, int(xs.min())), max(0, int(ys.min()))
    x1, y1 = min(w, int(xs.max())), min(h, int(ys.max()))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return jsonify({"error": "tray crop is empty -- check calibration"}), 400

    _prune()
    if _free_mb() < MIN_FREE_MB:
        return jsonify({"error": f"only {_free_mb():.0f} MB free on the Pi"}), 507
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}_tray.png"          # lossless: JPEG ringing sits exactly on
    try:                                # the numeral edges we care about
        _write_image(os.path.join(WORK, name), crop)
    except IOError as e:
        return jsonify({"error": str(e)}), 507

    return jsonify({
        "id": stamp,
        "tray_image": name,
        "tray_box": [x0, y0, x1, y1],
        "size": [crop.shape[1], crop.shape[0]],
        # Callers need to know the exposure moved: templates and any stored
        # reference are only valid for the settings they were captured under.
        "remetered": remetered,
        "controls": cam.controls(),
    })


@app.get("/framing")
def framing():
    """Brightness + sharpness uniformity across the tray, for mount tuning."""
    if cam is None:
        return _no_camera()
    img = cam.frame()
    if img is None:
        return jsonify({"error": "no frame yet"}), 503
    quad, qframe = load_quad()
    if quad is None:
        return jsonify({"error": "not calibrated"}), 400
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    quad, _ = fit_quad_to_frame(quad, qframe or list(cam.main_size), gray.shape)

    q = np.asarray(quad, np.float32)
    s, d = q.sum(axis=1), np.diff(q, axis=1).ravel()
    ordered = np.array([q[np.argmin(s)], q[np.argmin(d)],
                        q[np.argmax(s)], q[np.argmax(d)]])
    xs, ys = ordered[:, 0], ordered[:, 1]
    h, w = gray.shape

    def patch(cx, cy, half=50):
        x0, x1 = max(0, int(cx - half)), min(w, int(cx + half))
        y0, y1 = max(0, int(cy - half)), min(h, int(cy + half))
        p = gray[y0:y1, x0:x1]
        if p.size == 0:
            return None
        return {"mean": round(float(p.mean()), 1), "std": round(float(p.std()), 1),
                "sharp": round(float(cv2.Laplacian(p, cv2.CV_64F).var()), 1)}

    centre = patch(xs.mean(), ys.mean())
    corners = {}
    for name, (cx, cy) in zip(("TL", "TR", "BR", "BL"), ordered):
        corners[name] = patch(cx + (xs.mean() - cx) * 0.15,
                              cy + (ys.mean() - cy) * 0.15)

    return jsonify({
        "frame": [w, h],
        "coverage_pct": round(100.0 * cv2.contourArea(ordered.astype(np.int32)) / (w * h), 1),
        "margins_pct": {
            "left": round(100.0 * float(xs.min()) / w, 1),
            "right": round(100.0 * (w - float(xs.max())) / w, 1),
            "top": round(100.0 * float(ys.min()) / h, 1),
            "bottom": round(100.0 * (h - float(ys.max())) / h, 1),
        },
        "centre": centre, "corners": corners,
        # A flat patch has nothing to resolve, so its sharpness is meaningless.
        # The UI must not render a focus verdict for those.
        "flat_std_threshold": 8.0,
        "controls": cam.controls(),
    })


@app.get("/file/<path:name>")
def serve_file(name):
    return send_from_directory(WORK, name)


if __name__ == "__main__":
    # Log the tuning loudly, and say so when it is wrong. This line reads back
    # what picamera2 actually gave libcamera; the version that printed our own
    # environment variable said "noir" while libcamera loaded imx219.json.
    if cam.tuning_active == cam.tuning_expected:
        print(f"sensor: {cam.sensor} at {cam.main_size[0]}x{cam.main_size[1]}",
              flush=True)
        print(f"tuning: {cam.tuning_active}", flush=True)
    else:
        print(f"tuning: {cam.tuning_active} -- EXPECTED {cam.tuning_expected}; "
              f"the colour pipeline is wrong", flush=True)
    f = cam.focus_state()
    if f.get("supported"):
        print(f"focus : {f.get('lens_position')} dioptres"
              f"{'' if f.get('pinned') else '  (NOT PINNED -- run /autofocus)'}",
              flush=True)
    print(f"work dir: {WORK} ({WORK_SOURCE}, {_free_mb():.0f} MB free)", flush=True)
    app.run(host="0.0.0.0", port=8081, threaded=True)
