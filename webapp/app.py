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
import shutil
import subprocess
import tempfile
import time

import requests
from flask import Flask, Response, jsonify, render_template, request

import settings
from roll_reader import read_roll, _anthropic, _claude_code

PI = os.environ.get("DICECAM_PI", "http://10.0.0.23:8081")
OLLAMA = os.environ.get("DICECAM_OLLAMA", "http://10.0.0.5:11434")
VISION_MODEL = os.environ.get("DICECAM_VISION_MODEL", "gemma3:27b")

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")
LABELS = os.path.join(DATA, "labels.jsonl")
ROLLS = os.path.join(DATA, "rolls.jsonl")
os.makedirs(DATA, exist_ok=True)

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
    rec = {
        "crop": crop,
        "label": b.get("label"),
        "die_type": b.get("die_type"),
        "source": b.get("source"),
        "suggested": b.get("suggested"),
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
    if not crop:
        return jsonify({"error": "crop required"}), 400
    try:
        img = requests.get(f"{PI}/file/{crop}", timeout=20).content
    except Exception as e:
        return jsonify({"error": f"could not fetch crop from Pi: {e}"}), 502

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
    digits = "".join(c for c in raw if c.isdigit())
    return jsonify({
        "crop": crop,
        "raw": raw,
        "value": int(digits) if digits else None,
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


@app.post("/api/roll")
def read_roll_endpoint():
    """Full roll pipeline: Pi captures + counts, Ollama reads, we reconcile."""
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

    result = read_roll(img, pi_count=pi.get("die_count_estimate"),
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
        "size": pi.get("size"), "blob_count": pi.get("blob_count"),
        "die_count_estimate": pi.get("die_count_estimate"),
        "channel": pi.get("channel"), "controls": pi.get("controls"),
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
            "pi_count": pi.get("die_count_estimate"),
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
