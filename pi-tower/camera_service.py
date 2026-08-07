#!/usr/bin/env python3
"""
camera_service.py -- thin HTTP camera head, runs ON the Pi.

The control panel lives on TheBeast (10.0.0.5); the camera does not. This
service is the seam between them: it owns the camera, does the cheap
high-volume work (capture, detect, crop) and hands back small results.

Design decision -- ONE camera configuration, never switched.

picamera2 can reconfigure between video and still modes, but doing that live,
per request, on a Pi 3 is slow and a reliable source of hangs. Instead the
camera runs continuously in a video configuration with two streams:

    main  1640x1232 RGB888  -- what capture/detect operate on
    lores  640x480  YUV420  -- what the MJPEG preview serves

The ISP produces both from the same frame, so the preview costs almost nothing
and never fights the capture path. 1640x1232 is not a compromise here: the whole
detection pipeline was tuned at that resolution, and a die is ~114 px across,
far more than any classifier needs.

Endpoints:
    GET  /health                  service + camera state
    GET  /stream                  multipart MJPEG preview
    GET  /snapshot.jpg            single full-res JPEG (calibration clicking)
    POST /capture                 freeze a frame, returns its id
    POST /analyze                 detect dice on a captured frame
    POST /roll                    capture + tray crop + independent die count
    GET  /framing                 tray framing report
    GET  /calibration             read tray quad
    POST /calibration             write tray quad
    POST /lock  POST /unlock      freeze / release AE+AWB
    GET  /file/<name>             serve a produced image
"""

import io
import json
import os
import sys
import threading
import time

# BEST-EFFORT ONLY -- launch via run_camera_service.sh instead.
#
# libcamera cannot tell a NoIR module from a standard V2 and defaults to the
# IR-cut tuning, which costs real numeral separability. Setting the variable
# here looks like it should work and does not reliably: libcamera resolves the
# tuning path early enough that this assignment can lose the race, and it did on
# this Pi -- the service came up on imx219.json with the code below in place,
# while a plain `export` before launch gets imx219_noir.json every time.
#
# Kept as a fallback for someone running the file directly. /health reports
# which tuning actually won, and that report is the one to trust.
NOIR_TUNING = "/usr/share/libcamera/ipa/rpi/vc4/imx219_noir.json"

# Captured BEFORE we touch it. This -- not os.environ afterwards -- is the
# honest signal for whether the NoIR tuning is really in force, because only a
# value inherited from the launcher is guaranteed to have beaten libcamera to
# it. Reporting os.environ after our own assignment would claim success in
# exactly the case that silently fails.
INHERITED_TUNING = os.environ.get("LIBCAMERA_RPI_TUNING_FILE")

if "LIBCAMERA_RPI_TUNING_FILE" not in os.environ and os.path.exists(NOIR_TUNING):
    os.environ["LIBCAMERA_RPI_TUNING_FILE"] = NOIR_TUNING

import cv2                                    # noqa: E402
import numpy as np                            # noqa: E402
from flask import Flask, Response, jsonify, request, send_from_directory  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dice_detect import (detect_all, flag, fit_quad_to_frame, to_gray,   # noqa: E402
                         reference_geometry, is_clump)

try:
    from picamera2 import Picamera2
except ImportError:
    sys.exit("picamera2 missing -> sudo apt install -y python3-picamera2")

# Full sensor. Measured: at 1640x1232 the reader typed d20s unstably; at
# 3280x2464 it got 4/4 of them right, and the residual error became a
# consistent d12->d10 confusion rather than noise. Circularity was unchanged,
# so this buys nothing for the CV silhouette -- it buys accuracy in the model,
# which is where reading actually happens.
#
# Cost on a Pi 3: an RGB888 frame here is ~24 MB and the grabber holds a copy,
# against 905 MB usable. Frame rate drops to single digits. Acceptable because
# nothing in this pipeline needs a fast main stream -- the preview comes from
# lores, which is unaffected.
MAIN_SIZE = (3280, 2464)
LORES_SIZE = (640, 480)
WORK = os.path.expanduser("~/dicecam-web")
CONFIG = os.path.expanduser("~/.config/dicecam/tray.json")
EXPOSURE = os.path.expanduser("~/.config/dicecam/exposure.json")

os.makedirs(WORK, exist_ok=True)
app = Flask(__name__)


class Camera:
    """Owns the Picamera2 instance and a background frame grabber."""

    def __init__(self):
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_video_configuration(
            main={"size": MAIN_SIZE, "format": "RGB888"},
            lores={"size": LORES_SIZE, "format": "YUV420"},
        ))
        self.lock = threading.Lock()
        self.latest_main = None
        self.latest_lores = None
        self.meta = {}
        self.locked = False
        self.locked_controls = {}
        self.lock_lux = None
        self.running = True
        self.picam2.start()
        time.sleep(2)                      # let AE/AWB settle before first frame
        threading.Thread(target=self._loop, daemon=True).start()
        self._restore_lock()

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
            ctrl = dict(saved["controls"])
            if isinstance(ctrl.get("ColourGains"), list):
                ctrl["ColourGains"] = tuple(ctrl["ColourGains"])
            time.sleep(1.0)                 # let the pipeline accept controls
            self.picam2.set_controls(ctrl)
            self.locked = True
            self.locked_controls = saved["controls"]
            self.lock_lux = saved.get("lux")
            print(f"[camera] restored exposure lock from {EXPOSURE}: "
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
            "lock_lux": round(float(self.lock_lux), 1) if self.lock_lux else None,
            "lock_stale": self._lock_stale(m.get("Lux")),
        }

    def _lock_stale(self, lux_now):
        """Has the room's light changed materially since we froze the settings?"""
        if not self.locked or not self.lock_lux or not lux_now:
            return None
        ratio = float(lux_now) / float(self.lock_lux)
        if ratio > 1.5 or ratio < 0.67:
            return (f"Lighting changed since the lock ({self.lock_lux:.0f} -> "
                    f"{float(lux_now):.0f} lux). The frozen exposure and white "
                    f"balance are for the OLD light -- expect a colour cast and "
                    f"wrong brightness. Unlock, let it re-converge, and re-lock.")
        return None

    def unlock_exposure(self):
        self.picam2.set_controls({"AeEnable": True, "AwbEnable": True})
        self.locked = False
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
        self.locked_controls = {k: (list(v) if isinstance(v, tuple) else v)
                                for k, v in ctrl.items()}
        # Remember how bright the room was when we froze. A lock is only valid
        # for the light it was taken under: locking under purple LEDs and then
        # switching on a lamp leaves the camera applying a 3.28x blue gain to a
        # scene that no longer needs it, and the result is a heavy false colour
        # cast that looks like a camera fault rather than a stale setting.
        self.lock_lux = m.get("Lux")
        try:
            os.makedirs(os.path.dirname(EXPOSURE), exist_ok=True)
            json.dump({"controls": self.locked_controls, "lux": self.lock_lux,
                       "saved": time.strftime("%Y-%m-%dT%H:%M:%S")},
                      open(EXPOSURE, "w"), indent=2)
        except OSError as e:
            print(f"[camera] could not persist exposure lock: {e}", flush=True)
        return self.locked_controls


cam = Camera()


# ------------------------------------------------------------- calibration ---

def load_quad():
    if not os.path.exists(CONFIG):
        return None, None
    cfg = json.load(open(CONFIG))
    return np.array(cfg["quad"], np.float32).reshape(4, 2), cfg.get("frame")


@app.get("/calibration")
def get_calibration():
    quad, frame = load_quad()
    if quad is None:
        return jsonify({"calibrated": False, "quad": None, "frame": None})
    return jsonify({"calibrated": True, "quad": quad.astype(int).tolist(),
                    "frame": frame})


@app.post("/calibration")
def set_calibration():
    body = request.get_json(force=True)
    quad = body.get("quad")
    if not quad or len(quad) != 4 or any(len(p) != 2 for p in quad):
        return jsonify({"error": "quad must be 4 [x,y] pairs"}), 400
    frame = body.get("frame") or list(MAIN_SIZE)
    os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
    json.dump({"quad": [[int(p[0]), int(p[1])] for p in quad],
               "frame": [int(frame[0]), int(frame[1])]},
              open(CONFIG, "w"), indent=2)
    return jsonify({"saved": True, "path": CONFIG, "quad": quad, "frame": frame})


# ------------------------------------------------------------------ frames ---

@app.get("/health")
def health():
    f = cam.frame()
    return jsonify({
        "ok": f is not None,
        "main_size": list(MAIN_SIZE),
        "lores_size": list(LORES_SIZE),
        "tuning_file": INHERITED_TUNING or "(libcamera default -- imx219.json)",
        "noir_tuning_active": INHERITED_TUNING == NOIR_TUNING,
        "tuning_warning": (None if INHERITED_TUNING == NOIR_TUNING else
                           "Running on the IR-cut tuning. Captures are usable but "
                           "lose ~0.14 Otsu separability. Restart via "
                           "run_camera_service.sh."),
        "controls": cam.controls(),
        "calibrated": os.path.exists(CONFIG),
    })


@app.get("/stream")
def stream():
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
    f = cam.frame()
    if f is None:
        return jsonify({"error": "no frame yet"}), 503
    ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.post("/capture")
def capture():
    f = cam.frame()
    if f is None:
        return jsonify({"error": "no frame yet"}), 503
    cid = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(WORK, f"{cid}.jpg")
    cv2.imwrite(path, f, [cv2.IMWRITE_JPEG_QUALITY, 95])
    json.dump(cam.controls(), open(os.path.join(WORK, f"{cid}_controls.json"), "w"))
    return jsonify({"id": cid, "file": f"{cid}.jpg",
                    "width": f.shape[1], "height": f.shape[0],
                    "controls": cam.controls()})


@app.post("/lock")
def do_lock():
    c = cam.lock_exposure()
    return (jsonify({"locked": True, "controls": c}) if c
            else (jsonify({"error": "no metadata yet"}), 503))


@app.post("/unlock")
def do_unlock():
    cam.unlock_exposure()
    return jsonify({"locked": False})


# ---------------------------------------------------------------- analysis ---

def _resolve_frame(body):
    """Load the requested capture, or grab a live frame if none specified."""
    cid = body.get("id")
    if cid:
        p = os.path.join(WORK, f"{cid}.jpg")
        if not os.path.exists(p):
            return None, None, f"no such capture: {cid}"
        return cv2.imread(p), cid, None
    f = cam.frame()
    return (f, None, None) if f is not None else (None, None, "no frame yet")


@app.post("/analyze")
def analyze():
    body = request.get_json(force=True) or {}
    img, cid, err = _resolve_frame(body)
    if err:
        return jsonify({"error": err}), 503

    quad, qframe = load_quad()
    if quad is None:
        return jsonify({"error": "not calibrated -- set the tray quad first"}), 400
    try:
        quad, _ = fit_quad_to_frame(quad, qframe or list(MAIN_SIZE), img.shape)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    gray, channel_note = to_gray(img, quad, body.get("channel", "auto"))

    dets, rejected, info = detect_all(
        gray, quad,
        min_frac=float(body.get("min_frac", 0.002)),
        max_frac=float(body.get("max_frac", 0.45)),
        window=(int(body["window"]) if body.get("window") else None),
        split=bool(body.get("split", True)),
    )
    ref = reference_geometry(dets)
    for d in dets:
        d["suspect_reasons"] = flag(d, ref)
        d["suspect"] = bool(d["suspect_reasons"])

    ctx = float(body.get("context", 1.5))
    stamp = cid or time.strftime("%Y%m%d-%H%M%S")
    overlay = img.copy()
    cv2.polylines(overlay, [quad.astype(np.int32)], True, (0, 180, 0), 2)
    crops = []
    for d in dets:
        x, y, w, h = d["bbox"]
        col = (0, 0, 255) if d["suspect"] else (0, 255, 0)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), col, 3)
        cv2.putText(overlay, f"#{d['id']}", (x, max(14, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)

        # Context multiplier is a request parameter, not a constant, because the
        # bbox is not always the die. Under weak or coloured lighting the die
        # body sinks to the same level as the tray floor and only the painted
        # numerals carry local variance, so the detection bounds the DIGITS and
        # a 1.5x crop shows a numeral with no die around it. A vision model then
        # cannot tell which face is up -- measured: whole-die crops read
        # correctly, numeral-only crops did not. Widening the context recovers
        # the die. It treats the symptom; lighting is the cure.
        side = int(max(w, h) * ctx)
        cx, cy = x + w // 2, y + h // 2
        x0, y0 = cx - side // 2, cy - side // 2
        pl, pt = max(0, -x0), max(0, -y0)
        pr = max(0, x0 + side - img.shape[1])
        pb = max(0, y0 + side - img.shape[0])
        sub = img[max(0, y0):min(img.shape[0], y0 + side),
                  max(0, x0):min(img.shape[1], x0 + side)]
        if sub.size == 0:
            continue
        if pl or pt or pr or pb:
            sub = cv2.copyMakeBorder(sub, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
        name = f"{stamp}_d{d['id']:02d}.png"
        cv2.imwrite(os.path.join(WORK, name),
                    cv2.resize(sub, (128, 128), interpolation=cv2.INTER_AREA))
        crops.append({"id": d["id"], "file": name, "suspect": d["suspect"],
                      "reasons": d["suspect_reasons"]})

    oname = f"{stamp}_overlay.jpg"
    cv2.imwrite(os.path.join(WORK, oname), overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])

    return jsonify({
        "id": stamp, "overlay": oname, "detections": dets, "crops": crops,
        "rejected": rejected, "split": info.get("split"),
        "channel": channel_note,
        "clean": sum(1 for d in dets if not d["suspect"]),
        "suspect": sum(1 for d in dets if d["suspect"]),
        "controls": cam.controls(),
    })


@app.post("/roll")
def roll():
    """Capture, crop to the tray, and count dice. Reading happens remotely.

    The Pi deliberately does NOT try to read values or even to separate touching
    dice -- both were measured unreliable here, and the remote model is better at
    them. What it contributes is an INDEPENDENT COUNT.

    That matters more, not less, now that any number of dice of any type may be
    rolled: with no known set to check against, this count is the only signal
    that can catch the reader silently omitting a die. Measured: the plain
    reading prompt returned 6 values for a 7-dice pile and asserted it
    confidently.

    Counting is cheap and does not require the split to work -- a clump
    contributes its distance-peak count rather than 1, so a pile of seven still
    counts as roughly seven.
    """
    body = request.get_json(silent=True) or {}
    img = cam.frame()
    if img is None:
        return jsonify({"error": "no frame yet"}), 503

    quad, qframe = load_quad()
    if quad is None:
        return jsonify({"error": "not calibrated -- set the tray quad first"}), 400
    try:
        quad, _ = fit_quad_to_frame(quad, qframe or list(MAIN_SIZE), img.shape)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    gray, channel_note = to_gray(img, quad, body.get("channel", "auto"))
    dets, rejected, info = detect_all(
        gray, quad,
        min_frac=float(body.get("min_frac", 0.002)),
        max_frac=float(body.get("max_frac", 0.45)),
        split=bool(body.get("split", True)))

    ref = reference_geometry(dets)
    blobs = []
    count = 0
    for d in dets:
        peaks = max(1, int(d.get("n_peaks", 1) or 1))
        # A blob only counts as multiple dice if it is genuinely clump-shaped.
        # An elongated single d6 reads as two peaks; trusting peaks alone would
        # inflate the count on ordinary rolls.
        n = peaks if is_clump(d, ref) else 1
        count += n
        blobs.append({"id": d["id"], "bbox": d["bbox"], "area_px": d["area_px"],
                      "n_peaks": peaks, "counts_as": n,
                      "suspect_reasons": flag(d, ref)})

    # Crop to the tray's bounding box. Everything outside is desk clutter that
    # only gives the reader more chances to hallucinate a die.
    h, w = gray.shape
    xs, ys = quad[:, 0], quad[:, 1]
    x0, y0 = max(0, int(xs.min())), max(0, int(ys.min()))
    x1, y1 = min(w, int(xs.max())), min(h, int(ys.max()))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return jsonify({"error": "tray crop is empty -- check calibration"}), 400

    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}_tray.png"          # lossless: JPEG ringing sits exactly on
    cv2.imwrite(os.path.join(WORK, name), crop)   # the numeral edges we care about

    return jsonify({
        "id": stamp,
        "tray_image": name,
        "tray_box": [x0, y0, x1, y1],
        "size": [crop.shape[1], crop.shape[0]],
        "die_count_estimate": count,
        "blob_count": len(dets),
        "blobs": blobs,
        "channel": channel_note,
        "split": info.get("split"),
        "controls": cam.controls(),
    })


@app.get("/framing")
def framing():
    """Brightness + sharpness uniformity across the tray, for mount tuning."""
    img = cam.frame()
    if img is None:
        return jsonify({"error": "no frame yet"}), 503
    quad, qframe = load_quad()
    if quad is None:
        return jsonify({"error": "not calibrated"}), 400
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    quad, _ = fit_quad_to_frame(quad, qframe or list(MAIN_SIZE), gray.shape)

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
    print(f"tuning: {os.environ.get('LIBCAMERA_RPI_TUNING_FILE')}", flush=True)
    print(f"work dir: {WORK}", flush=True)
    app.run(host="0.0.0.0", port=8081, threaded=True)
