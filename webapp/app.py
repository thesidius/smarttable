#!/usr/bin/env python3
"""
Dice tower control panel -- runs on TheBeast (10.0.0.5).

Three machines, three jobs:
    Pi 10.0.0.23   camera head; owns capture/detect/crop  (camera_service.py)
    TheBeast .5    this app, plus Ollama (RTX 4090)
    G5 .81         currently unassigned

The browser talks to BOTH this app and the Pi. Control calls are proxied
through here so there is one origin for the API and no CORS dance, but the
MJPEG preview is deliberately loaded straight from the Pi by the <img> tag:
proxying a live video stream through this process would add a hop and a buffer
on the exact path where latency is felt -- while you are physically adjusting
the mount and watching the picture move.

Run:  python app.py     then open http://10.0.0.5:5000
"""

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import time

import requests
from flask import Flask, Response, jsonify, render_template, request

import settings
from roll_reader import (read_roll, read_segmented, read_segmented_batch,
                         _anthropic, _claude_code, _parse)

PI = os.environ.get("DICECAM_PI", "http://10.0.0.23:8081")
OLLAMA = os.environ.get("DICECAM_OLLAMA", "http://10.0.0.5:11434")
VISION_MODEL = os.environ.get("DICECAM_VISION_MODEL", "gemma3:27b")
SEG = os.environ.get("DICECAM_SEG", "http://127.0.0.1:8090")

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")
LABELS = os.path.join(DATA, "labels.jsonl")
ROLLS = os.path.join(DATA, "rolls.jsonl")
CROPS = os.path.join(DATA, "crops")
os.makedirs(CROPS, exist_ok=True)

SINGLE_DIE_PROMPT = (
    "This is a photo of a single polyhedral tabletop RPG die viewed from above. "
    "Reply ONLY with a JSON object {\"dice\":[{\"type\":\"dN\",\"value\":<int>}]} "
    "for that one die: its type, and the number on the face pointing UP at the "
    "camera. Numbers on slanted side faces are not the answer."
)

VISION_PROMPT = (
    "This is a photo of a single polyhedral tabletop RPG die, viewed from above. "
    "Which number is on the face pointing UP toward the camera? Other faces are "
    "visible at an angle - ignore those. Reply with just the number."
)

app = Flask(__name__)


# ------------------------------------------------------------------- status ---

@app.get("/api/status")
def status():
    out = {"pi": None, "pi_error": None, "ollama": None, "ollama_error": None,
           "pi_base": PI, "vision_model": VISION_MODEL}
    try:
        out["pi"] = requests.get(f"{PI}/health", timeout=8).json()
    except Exception as e:
        out["pi_error"] = str(e)
    try:
        tags = requests.get(f"{OLLAMA}/api/tags", timeout=8).json()
        names = [m["name"] for m in tags.get("models", [])]
        out["ollama"] = {"models": names, "vision_available": VISION_MODEL in names}
    except Exception as e:
        out["ollama_error"] = str(e)
    out["anthropic"] = settings.public()
    return jsonify(out)


# -------------------------------------------------------------- Pi proxying ---

@app.route("/api/pi/<path:endpoint>", methods=["GET", "POST"])
def pi_proxy(endpoint):
    url = f"{PI}/{endpoint}"
    try:
        if request.method == "POST":
            r = requests.post(url, json=request.get_json(silent=True) or {}, timeout=120)
        else:
            r = requests.get(url, params=request.args, timeout=120)
    except Exception as e:
        return jsonify({"error": f"Pi unreachable: {e}"}), 502
    ct = r.headers.get("Content-Type", "")
    if "application/json" in ct:
        return jsonify(r.json()), r.status_code
    return Response(r.content, status=r.status_code, mimetype=ct)


# ---------------------------------------------------------------- labelling ---

def read_labels():
    if not os.path.exists(LABELS):
        return {}
    out = {}
    with open(LABELS) as f:
        for line in f:
            try:
                r = json.loads(line)
                out[r["crop"]] = r          # later lines win: last edit stands
            except ValueError:
                continue
    return out


@app.get("/api/labels")
def get_labels():
    return jsonify(read_labels())


@app.get("/crops/<path:name>")
def get_crop(name):
    """Serve a saved crop so a label can be checked against the image later."""
    path = os.path.join(CROPS, os.path.basename(name))
    if not os.path.exists(path):
        return jsonify({"error": "no such crop"}), 404
    with open(path, "rb") as fh:
        return Response(fh.read(), mimetype="image/png")


@app.post("/api/label")
def put_label():
    """Append-only label log.

    Append rather than rewrite: labelling is the expensive, unrepeatable part of
    this project, and an append-only log cannot lose earlier work to a crash or
    a bad edit halfway through a rewrite. read_labels() collapses to last-write-
    wins, so corrections are just another append.
    """
    b = request.get_json(force=True)
    crop = b.get("crop")
    if not crop:
        return jsonify({"error": "crop required"}), 400

    # Validate at the door. This is the training-data log, and a bad entry is
    # worse than a missing one because it looks like a labelled example.
    #
    # Blank is the obvious case. The less obvious one, observed live: browsers
    # and password managers autofill the last lone text input on a page, and a
    # saved username landed in a die-value box. Nothing about "pbahnmiller" is
    # detectable later as an accident -- it would just be a die whose recorded
    # face was a person's name. A die face is one to three digits; anything
    # else is rejected and named in the error, so the operator can see what was
    # actually in the box.
    label = (b.get("label") or "").strip()
    if not label:
        return jsonify({"error": "label is empty -- nothing to record"}), 400
    if not re.fullmatch(r"\d{1,3}", label) or int(label) > 100:
        return jsonify({"error": f"{label!r} is not a die face. Expected 1-3 "
                                 f"digits, 0-100. If you did not type this, it "
                                 f"is browser autofill -- clear the box and "
                                 f"re-enter the value."}), 400

    # SAM2 crops are composited in memory and never written to the Pi, so a
    # label referring to one would point at nothing. Persist the image here,
    # under the dataset, the first time it is labelled: a label log whose crops
    # cannot be opened again is useless as training data, which is the only
    # reason to be labelling at all.
    inline = b.get("image")
    if inline:
        path = os.path.join(CROPS, os.path.basename(crop))
        if not path.lower().endswith(".png"):
            path += ".png"
        try:
            with open(path, "wb") as fh:
                fh.write(base64.b64decode(inline))
        except Exception as e:
            return jsonify({"error": f"could not save crop: {e}"}), 500
        crop = os.path.relpath(path, DATA).replace("\\", "/")

    rec = {
        "crop": crop,
        "label": label,
        "die_type": b.get("die_type"),
        "source": b.get("source"),
        "suggested": b.get("suggested"),
        # Which crop geometry this label was made against. Without it, a change
        # to the crop definition silently invalidates every earlier label and
        # there is no way to tell afterwards which ones. It has already changed
        # once (v1 -> v2, stray-pixel cleanup, which re-framed every crop).
        "crop_version": b.get("crop_version"),
        "confirmed_by": "human",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(LABELS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return jsonify({"saved": True, "record": rec})


@app.post("/api/suggest")
def suggest():
    """Ask the vision model to read one crop. Pre-labelling, not ground truth."""
    b = request.get_json(force=True)
    crop = b.get("crop")
    inline = b.get("image")
    # Accept an inline base64 crop as well as a filename on the Pi. SAM2 crops
    # are composited in memory on the workstation and never written to the Pi,
    # so a filename cannot reach them -- and those are now the crops the reader
    # actually sees, which is what labelling should be based on.
    if inline:
        try:
            img = base64.b64decode(inline)
        except Exception as e:
            return jsonify({"error": f"bad inline image: {e}"}), 400
    elif crop:
        try:
            img = requests.get(f"{PI}/file/{crop}", timeout=20).content
        except Exception as e:
            return jsonify({"error": f"could not fetch crop from Pi: {e}"}), 502
    else:
        return jsonify({"error": "crop filename or inline image required"}), 400

    # Same provider choice as a roll. The Label tab was hard-wired to Ollama,
    # which meant pre-labelling used the weakest reader available while the
    # rolls it is meant to produce training data FOR went through Claude Code.
    provider = b.get("provider") or (
        "claude-code" if settings.code_token() else
        "anthropic" if settings.key() else "ollama")

    t0 = time.time()
    try:
        if provider == "claude-code":
            raw = _claude_code(SINGLE_DIE_PROMPT, img, settings.code_model(),
                               settings.code_token(), 300)
            used = settings.code_model()
        elif provider == "anthropic":
            raw = _anthropic(SINGLE_DIE_PROMPT, img, settings.model(),
                             settings.key(), 300, settings.effort())
            used = settings.model()
        else:
            r = requests.post(f"{OLLAMA}/api/generate", timeout=300, json={
                "model": VISION_MODEL,
                "prompt": VISION_PROMPT,
                "images": [base64.b64encode(img).decode()],
                "stream": False,
            }).json()
            raw = (r.get("response") or "")
            used = VISION_MODEL
    except Exception as e:
        return jsonify({"error": f"{provider} call failed: {e}"}), 502

    raw = (raw or "").strip()

    # Two reply shapes, because the two prompts ask for different things: the
    # Claude prompts request JSON, the Ollama one asks for "just the number".
    #
    # Do NOT fall back to scraping every digit. That was the original code, and
    # on {"type":"d20","value":4} it concatenated to 204 -- pre-filling the
    # label box with a wrong value for a human to rubber-stamp into the
    # training set, which is the worst possible place for a silent error. A
    # reply that is nothing but an integer is unambiguous; anything else has to
    # parse as JSON or it is reported as unread.
    dice = _parse(raw) or []
    die = dice[0] if dice else None
    value = die["value"] if die else (int(raw) if raw.isdigit() else None)
    return jsonify({
        "crop": crop,
        "raw": raw,
        "value": value,
        "die_type": die.get("type") if die else None,
        "n_parsed": len(dice),
        "seconds": round(time.time() - t0, 2),
        "provider": provider,
        "model": used,
    })


@app.post("/api/suggest_batch")
def suggest_batch():
    crops = (request.get_json(force=True) or {}).get("crops") or []
    results = []
    for c in crops:
        with app.test_request_context(json={"crop": c}):
            resp = suggest()
        results.append(resp.get_json() if hasattr(resp, "get_json") else resp[0].get_json())
    return jsonify({"results": results})


@app.get("/api/settings")
def get_settings():
    return jsonify(settings.public())


@app.post("/api/settings")
def post_settings():
    """Save operator settings. The response never contains the key."""
    return jsonify(settings.update(request.get_json(force=True) or {}))


@app.get("/api/settings/models")
def list_models():
    """Models this key can actually reach, from the API rather than a hard-coded list.

    A baked-in list goes stale the moment a model is released or retired, and
    then offers something the key cannot use -- a failure that surfaces as a
    404 mid-roll rather than at the moment of choosing. Asking the API means
    the dropdown can only ever offer what is really available.
    """
    k = settings.key()
    if not k:
        return jsonify({"ok": False, "error": "no key configured", "models": []}), 400
    try:
        r = requests.get("https://api.anthropic.com/v1/models",
                         timeout=30, params={"limit": 100},
                         headers={"x-api-key": k, "anthropic-version": "2023-06-01"})
    except Exception as e:
        return jsonify({"ok": False, "error": f"network: {e}", "models": []}), 502
    if r.status_code != 200:
        detail = ""
        try:
            detail = r.json().get("error", {}).get("message", "")
        except ValueError:
            pass
        return jsonify({"ok": False, "status": r.status_code,
                        "error": detail or "request failed", "models": []}), 200

    models = [{"id": m.get("id"), "name": m.get("display_name") or m.get("id"),
               "created": m.get("created_at", "")}
              for m in (r.json().get("data") or []) if m.get("id")]
    # Newest first: the list arrives roughly chronological and the useful
    # default is almost always a recent model, not whatever sorts first
    # alphabetically.
    models.sort(key=lambda m: m["created"], reverse=True)
    return jsonify({"ok": True, "models": models, "current": settings.model()})


@app.post("/api/settings/code_test")
def test_code_token():
    """Verify the Claude Code token AND the chosen model, together.

    There is no way to enumerate models for this path -- `claude` has no models
    subcommand, only `--model` taking an alias or a full name. So instead of
    offering a list that might be wrong, actually run the thing: a trivial
    prompt with no tools exercises auth, the model name and the subprocess in
    one go, which is what a list would only have guessed at.
    """
    tok = settings.code_token()
    if not tok:
        return jsonify({"ok": False, "error": "no Claude Code token configured"}), 400
    model = (request.get_json(silent=True) or {}).get("model") or settings.code_model()

    env = dict(os.environ)
    env["CLAUDE_CODE_OAUTH_TOKEN"] = tok
    env.pop("ANTHROPIC_API_KEY", None)
    work = tempfile.mkdtemp(prefix="dicecam_test_")
    t0 = time.time()
    try:
        r = subprocess.run(
            ["claude", "-p", "Reply with the single word OK.",
             "--model", model, "--output-format", "json",
             "--allowedTools", "",
             "--disallowedTools", "Bash Read Write Edit Glob Grep WebFetch WebSearch Task NotebookEdit",
             "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}'],
            env=env, cwd=work, capture_output=True, text=True, timeout=180)
    except FileNotFoundError:
        return jsonify({"ok": False, "error": "the `claude` CLI is not on PATH"}), 200
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "error": "timed out after 180s"}), 200
    finally:
        shutil.rmtree(work, ignore_errors=True)

    out = (r.stdout or "").strip()
    if not out:
        return jsonify({"ok": False, "error": (r.stderr or "no output").strip()[:300]}), 200
    try:
        env_json = json.loads(out)
    except ValueError:
        return jsonify({"ok": False, "error": out[:300]}), 200
    if env_json.get("is_error"):
        return jsonify({"ok": False, "error": str(env_json.get("result"))[:300]}), 200
    return jsonify({"ok": True, "model": model,
                    "seconds": round(time.time() - t0, 1),
                    "reply": str(env_json.get("result", ""))[:80]})


@app.post("/api/settings/test")
def test_settings():
    """Prove the key works, without revealing it.

    A one-token request is the cheapest question that still exercises auth, the
    model name and the network path -- the three things that are actually in
    doubt after typing a key into a box.
    """
    k = settings.key()
    if not k:
        return jsonify({"ok": False, "error": "no key configured"}), 400
    try:
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=60,
                          headers={"x-api-key": k,
                                   "anthropic-version": "2023-06-01",
                                   "content-type": "application/json"},
                          json={"model": settings.model(), "max_tokens": 1,
                                "messages": [{"role": "user", "content": "hi"}]})
    except Exception as e:
        return jsonify({"ok": False, "error": f"network: {e}"}), 502
    if r.status_code == 200:
        return jsonify({"ok": True, "model": settings.model()})
    detail = ""
    try:
        detail = r.json().get("error", {}).get("message", "")
    except ValueError:
        pass
    # Report the status and the API's own message. Neither contains the key.
    return jsonify({"ok": False, "status": r.status_code, "error": detail}), 200


@app.post("/api/roll2")
def roll_phase1():
    """Phase 1: SAM2 separates, the VLM reads each isolated die.

    Replaces the whole-tray read. The difference is not subtle -- measured, the
    reader misread a d20 as 14 inside a pile and read it correctly at 20 once
    isolated, and SAM2 separates 7/7 of a tight pile where the distance-transform
    method reported 20 dice for 8.

    The mask COUNT stays as the independent check. It is the only signal that
    catches the reader silently omitting a die, and now it comes from something
    that has been measured to count correctly.
    """
    t0 = time.time()
    body = request.get_json(silent=True) or {}

    try:
        pi = requests.post(f"{PI}/roll", json=body, timeout=180).json()
    except Exception as e:
        return jsonify({"error": f"Pi unreachable: {e}"}), 502
    if pi.get("error"):
        return jsonify({"error": pi["error"], "stage": "capture"}), 400
    try:
        img = requests.get(f"{PI}/file/{pi['tray_image']}", timeout=60).content
    except Exception as e:
        return jsonify({"error": f"could not fetch tray image: {e}"}), 502

    try:
        seg = requests.post(f"{SEG}/segment", timeout=300,
                            json={"image": base64.b64encode(img).decode()}).json()
    except Exception as e:
        return jsonify({"error": f"segmentation service unreachable at {SEG}: {e}. "
                                 f"Start it with .venv-ml\\Scripts\\python.exe "
                                 f"webapp\\seg_service.py", "stage": "segment"}), 502
    if seg.get("error"):
        return jsonify({"error": seg["error"], "stage": "segment"}), 500

    provider = body.get("provider") or (
        "claude-code" if settings.code_token() else
        "anthropic" if settings.key() else "ollama")

    # Batched by default: one call per prompt variant covering every die,
    # rather than one call per die. Per-die was 16 subprocess launches for 8
    # dice and ~295 s of a ~298 s roll.
    dice, raw = read_segmented_batch(
        seg.get("dice", []), provider,
        code_token=settings.code_token(), code_model=settings.code_model(),
        anthropic_key=settings.key(), anthropic_model=settings.model(),
        anthropic_effort=settings.effort(),
        ollama=OLLAMA, ollama_model=VISION_MODEL,
        variants=int(body.get("variants") or 2))

    # The Pi's classical count used to be cross-checked here. It is gone: on an
    # 8-dice tray it reported 20, 27, 36 and 73, so every roll raised a
    # mismatch and the warning stopped meaning anything. SAM2's count stands
    # alone, and disagreement between prompts is the confidence signal now.
    issues = []
    low = [d for d in dice if d["confidence"] == "low"]
    if low:
        issues.append(f"{len(low)} die/dice read differently by different prompts")

    result = {
        "provider": provider,
        "model": (settings.code_model() if provider == "claude-code"
                  else settings.model() if provider == "anthropic" else VISION_MODEL),
        "dice": dice,
        "values": [d["value"] for d in dice if d["value"] is not None],
        "count": seg.get("count"),
        "raw_masks": seg.get("raw_masks"),
        "high_confidence": sum(1 for d in dice if d["confidence"] == "high"),
        "issues": issues,
        "seg_seconds": seg.get("seconds"),
        "raw": raw,
        "seconds": round(time.time() - t0, 2),
        "capture": {"id": pi.get("id"), "tray_image": pi.get("tray_image"),
                    "remetered": pi.get("remetered"),
                    "controls": pi.get("controls")},
    }
    with open(ROLLS, "a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "id": pi.get("id"), "pipeline": "phase1",
                            "values": result["values"],
                            "types": [d["type"] for d in dice],
                            "issues": issues,
                            "seg_count": seg.get("count")}) + "\n")
    return jsonify(result)


@app.post("/api/segment")
def segment_only():
    """Capture and segment, WITHOUT reading. The Detect tab's job now.

    Reading costs minutes; segmentation costs three seconds. Separating them
    means the question "did segmentation get this right?" can be answered
    without paying for a read -- and that is the question worth asking first
    when a roll comes back wrong, because a bad mask and a bad read produce the
    same symptom.
    """
    t0 = time.time()
    try:
        pi = requests.post(f"{PI}/roll", json={}, timeout=180).json()
    except Exception as e:
        return jsonify({"error": f"Pi unreachable: {e}"}), 502
    if pi.get("error"):
        return jsonify({"error": pi["error"], "stage": "capture"}), 400
    try:
        img = requests.get(f"{PI}/file/{pi['tray_image']}", timeout=60).content
    except Exception as e:
        return jsonify({"error": f"could not fetch tray image: {e}"}), 502
    try:
        seg = requests.post(f"{SEG}/segment", timeout=300,
                            json={"image": base64.b64encode(img).decode(),
                                  "overlay": True}).json()
    except Exception as e:
        return jsonify({"error": f"segmentation service unreachable at {SEG}: {e}. "
                                 f"Start it with .venv-ml\\Scripts\\python.exe "
                                 f"webapp\\seg_service.py", "stage": "segment"}), 502
    if seg.get("error"):
        return jsonify({"error": seg["error"], "stage": "segment"}), 500

    return jsonify({
        "count": seg.get("count"), "raw_masks": seg.get("raw_masks"),
        "rejected": seg.get("rejected"), "overlay": seg.get("overlay"),
        "crops": [{"id": d["id"], "bbox": d["bbox"], "area": d["area"],
                   "crop": d["crop"]} for d in seg.get("dice", [])],
        "seg_seconds": seg.get("seconds"), "model": seg.get("model"),
        "crop_version": seg.get("crop_version"),
        "seconds": round(time.time() - t0, 2),
        "capture": {"id": pi.get("id"), "tray_image": pi.get("tray_image"),
                    "size": pi.get("size"), "controls": pi.get("controls"),
                    # Surfaced, not swallowed: a re-meter means the exposure
                    # changed under this capture, which invalidates anything
                    # keyed to the previous settings -- templates especially.
                    "remetered": pi.get("remetered"),
                    "work_source": pi.get("controls", {}).get("work_source")},
    })


@app.get("/api/seg/health")
def seg_health():
    try:
        return jsonify(requests.get(f"{SEG}/health", timeout=8).json())
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 502


@app.post("/api/roll")
def read_roll_endpoint():
    """Phase 0: the Pi captures the whole tray, the reader reads it in one shot.

    Kept as the no-segmentation baseline to measure Phase 1 against. It is not
    the pipeline -- /api/roll2 is -- and it is the weaker one: on a pile it
    returned 6 values for 7 dice and asserted it confidently.
    """
    t0 = time.time()
    body = request.get_json(silent=True) or {}
    try:
        pi = requests.post(f"{PI}/roll", json=body, timeout=180).json()
    except Exception as e:
        return jsonify({"error": f"Pi unreachable: {e}"}), 502
    if pi.get("error"):
        return jsonify({"error": pi["error"], "stage": "capture"}), 400

    try:
        img = requests.get(f"{PI}/file/{pi['tray_image']}", timeout=60).content
    except Exception as e:
        return jsonify({"error": f"could not fetch tray image: {e}"}), 502

    result = read_roll(img,
                       ollama=OLLAMA, model=VISION_MODEL,
                       provider=body.get("provider"),
                       anthropic_key=settings.key(),
                       anthropic_model=settings.model(),
                       anthropic_effort=settings.effort(),
                       code_token=settings.code_token(),
                       code_model=settings.code_model(),
                       variants=body.get("variants"))
    result["capture"] = {
        "id": pi.get("id"), "tray_image": pi.get("tray_image"),
        "size": pi.get("size"), "controls": pi.get("controls"),
    }
    result["seconds"] = round(time.time() - t0, 2)

    # Append-only roll log. This is the record a downstream game system consumes,
    # and it keeps the low-confidence cases rather than discarding them -- an
    # uncertain roll that is marked uncertain is usable; one silently dropped or
    # silently guessed is not.
    with open(ROLLS, "a") as f:
        f.write(json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "id": pi.get("id"),
            "values": result.get("values"),
            "agreement": result.get("agreement"),
            "issues": result.get("issues"),
        }) + "\n")
    return jsonify(result)


@app.get("/api/rolls")
def recent_rolls():
    if not os.path.exists(ROLLS):
        return jsonify([])
    with open(ROLLS) as f:
        lines = f.readlines()[-40:]
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return jsonify(out[::-1])


@app.get("/")
def index():
    return render_template("index.html", pi_base=PI, vision_model=VISION_MODEL)


if __name__ == "__main__":
    print(f"Pi      : {PI}")
    print(f"Ollama  : {OLLAMA}  model {VISION_MODEL}")
    print(f"Labels  : {os.path.abspath(LABELS)}")
    app.run(host="0.0.0.0", port=5000, threaded=True)
