#!/usr/bin/env python3
"""
roll_reader.py -- read a tray image into dice values, with honest confidence.

Division of labour (see docs/system-architecture.md): the Pi captures, crops to
the tray, and counts. This module does the reading, remotely, via Ollama.

The whole design exists because of one measured fact: **the model's own
confidence is worthless.** Asked to flag dice it could not clearly see, it
returned `"confidence": "high"` with `"Clear view of the '9'"` for a die whose
up-face is not recoverable from the image at all. Meanwhile its value for that
die moved across prompt phrasings -- 8, then 6, then 9.

So confidence is derived from two signals the model does not control:

  1. AGREEMENT ACROSS GENUINELY DIFFERENT PROMPTS. Not repetitions -- at
     temperature 0 the same prompt returns byte-identical output every time, so
     repeating it catches nothing. Different framings disagree exactly where the
     reading is unreliable: on one test pile the d20 flipped between 14 and 20
     across formulations while every unambiguous die stayed put.

  2. THE PI'S INDEPENDENT COUNT. With any number of dice of any type in play
     there is no known set to validate against, so this is the only thing that
     can catch the reader silently omitting a die -- measured: the plain prompt
     returned 6 values for a 7-dice pile.

No prior about how many dice or which types. A set-composition prior was tested
and made things WORSE: told to expect one of each type, the model produced
structurally flawless output (no duplicates, no missing types, every value in
range) while calling a d20 showing 20 a "14" and inventing a d100 to hold the
20. Structural validation cannot catch semantic error.
"""

import base64
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter

import requests
from PIL import Image

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

# Long edge cap for images sent to the API. Above this the API downscales
# anyway, so sending more is paying to transmit detail that gets thrown away --
# and a full-sensor tray crop is 1980x1900.
MAX_IMAGE_EDGE = 1568

# The reply shape, enforced at the API rather than parsed out of prose. This is
# the main reason the Anthropic path is preferable to Ollama here: a reply
# either matches the shape or is rejected before it reaches us, which removes
# the whole class of "model wrote a sentence instead of JSON" failures that the
# regex parser below exists to survive.
DICE_SCHEMA = {
    "type": "object",
    "properties": {
        "dice": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string",
                             "enum": ["d4", "d6", "d8", "d10", "d12", "d20", "unknown"]},
                    "value": {"type": "integer"},
                },
                "required": ["type", "value"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["dice"],
    "additionalProperties": False,
}


def _shrink(image_bytes):
    """Cap the long edge. Returns (bytes, media_type)."""
    try:
        im = Image.open(io.BytesIO(image_bytes))
    except Exception:
        return image_bytes, "image/png"
    if max(im.size) <= MAX_IMAGE_EDGE:
        return image_bytes, "image/png"
    s = MAX_IMAGE_EDGE / float(max(im.size))
    im = im.convert("RGB").resize(
        (max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), "image/jpeg"


def _anthropic(prompt, image_bytes, model, key, timeout, effort=None):
    """One call, following the pattern in swadeledger's ai_lib.js."""
    data, media = _shrink(image_bytes)

    # Image BEFORE the text. That is the documented ordering for a single
    # image, and the honest one: everything the instructions refer to is in the
    # picture, so putting the picture second asks the model to hold a set of
    # rules in mind for something it has not seen yet.
    content = [
        {"type": "image",
         "source": {"type": "base64", "media_type": media,
                    "data": base64.b64encode(data).decode()}},
        {"type": "text", "text": prompt},
    ]

    body = {
        "model": model,
        "max_tokens": 2000,
        # The instruction block is identical for every roll, so mark it cached:
        # repetition becomes something paid for once rather than every roll.
        "system": [{"type": "text",
                    "text": "You read tabletop RPG dice from photographs. Be literal: "
                            "report only what is visible, and never invent a die.",
                    "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": content}],
        "output_config": {"format": {"type": "json_schema", "schema": DICE_SCHEMA}},
    }
    if "haiku" not in str(model):
        body["output_config"]["effort"] = effort or "low"

    r = requests.post(ANTHROPIC_URL, timeout=timeout, json=body, headers={
        "x-api-key": key,                      # the one and only use of the key
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    })
    if r.status_code != 200:
        detail = ""
        try:
            detail = r.json().get("error", {}).get("message", "")
        except ValueError:
            pass
        raise RuntimeError(f"Anthropic returned {r.status_code}" + (f": {detail}" if detail else ""))

    b = r.json()
    # A classifier refusal arrives as an ordinary 200 with empty content;
    # reading content[0] without checking turns that into a confusing crash.
    if b.get("stop_reason") == "refusal":
        raise RuntimeError("the model declined this image")
    if b.get("stop_reason") == "max_tokens":
        raise RuntimeError("reply truncated by max_tokens")
    return "".join(blk.get("text", "") for blk in b.get("content", [])
                   if blk.get("type") == "text")

def _claude_code(prompt, image_bytes, model, token, timeout, effort=None):
    """Read via the Claude Code CLI, using a CLAUDE_CODE_OAUTH_TOKEN.

    swadeledger's ai_lib.js marks claude-code as unable to see images, and for
    its workload that is correct: it passes page data inline and `claude -p` has
    no flag for an image. Here the image is a FILE, and Claude Code can open a
    file with Read. Measured on an 8-dice tray, this is the only configuration
    that got the d12/d20 split right (4 and 4); Ollama called every d12 a d10.

    Flags follow swadeledger, with one deliberate difference. It strips every
    tool including Read, because raw PDF text lands in the instruction position
    of an agent holding credentials. We cannot strip Read -- it is how the agent
    sees the picture at all -- so the blast radius is contained a different way:
    the working directory is a fresh temp dir containing ONLY the image, MCP is
    pinned to an empty config so nothing declared elsewhere can attach, and
    every other tool is refused by name. A photograph is a much narrower
    injection surface than arbitrary book text, but it is not zero -- someone
    could photograph written instructions -- and Read is scoped accordingly.
    """
    schema = DICE_SCHEMA
    work = tempfile.mkdtemp(prefix="dicecam_")
    try:
        img_path = os.path.join(work, "tray.png")
        with open(img_path, "wb") as f:
            f.write(image_bytes)

        env = dict(os.environ)
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token      # the one use of the token
        env.pop("ANTHROPIC_API_KEY", None)          # do not let a key shadow it

        full = ("Read the image file tray.png in the current directory. " + prompt)
        cmd = [
            "claude", "-p", full,
            "--json-schema", json.dumps(schema),
            "--model", model,
            "--output-format", "json",
            "--allowedTools", "Read",
            "--disallowedTools",
            "Bash Write Edit Glob Grep WebFetch WebSearch Task NotebookEdit",
            "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
        ]
        r = subprocess.run(cmd, env=env, cwd=work, capture_output=True,
                           text=True, timeout=timeout)
        out = (r.stdout or "").strip()
        if not out:
            raise RuntimeError((r.stderr or "claude produced no output").strip()[:300])
        env_json = json.loads(out)
        if env_json.get("is_error"):
            raise RuntimeError(str(env_json.get("result"))[:300])
        return env_json.get("result") or ""
    finally:
        shutil.rmtree(work, ignore_errors=True)


# Three genuinely different framings, not reworded twins. Enumeration order,
# spatial framing, and whether type is asked for all change how the model
# attends to the image; that is what makes disagreement informative.
PROMPTS = [
    ("typed",
     "Top-down photo of polyhedral RPG dice in a tray. There may be any number of "
     "dice of any types. Identify every die you can see: its type (by face count, "
     "e.g. d4 d6 d8 d10 d12 d20) and the number on the face pointing UP at the "
     "camera. Do not assume how many dice there are or which types are present. "
     'Reply ONLY with a JSON array of {"type":"<dN>","value":<int>}.'),

    ("spatial",
     "This photo shows dice on a tray, seen from directly above. Working from the "
     "top-left of the image to the bottom-right, go through the dice one at a time. "
     "For each, report only the number on its uppermost face -- the face aimed at "
     "the camera. Numbers on slanted side faces are not the result. "
     'Reply ONLY with a JSON array of {"value":<int>}.'),

    ("physical",
     "Photo of dice resting on a tray, viewed from above. Each die is a separate "
     "solid object. A single die shows several numbers at once - one on its top "
     "face and others on its slanted sides - but only ONE number per die counts: "
     "the one on the face aimed straight up at the camera. Report one number per "
     "physical die, not one per number you can see. "
     'Reply ONLY with a JSON array of {"value":<int>}.'),
]


def _parse(text):
    """Pull dice out of a model reply, tolerating fences, prose and both shapes.

    The Anthropic path returns {"dice": [...]} enforced by json_schema and needs
    none of this leniency; Ollama returns whatever it feels like, which is what
    the fence-stripping and bracket-hunting are for.
    """
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    if t.startswith("{"):
        try:
            obj = json.loads(t)
            if isinstance(obj, dict) and isinstance(obj.get("dice"), list):
                t = json.dumps(obj["dice"])
        except ValueError:
            pass
    if not t.startswith("["):
        m = re.search(r"\[.*\]", t, re.S)
        if not m:
            return None
        t = m.group(0)
    try:
        data = json.loads(t)
    except ValueError:
        return None
    out = []
    for e in data if isinstance(data, list) else []:
        if isinstance(e, dict):
            v, ty = e.get("value"), e.get("type")
        else:
            v, ty = e, None
        try:
            out.append({"value": int(v), "type": ty})
        except (TypeError, ValueError):
            continue
    return out


def _ask(ollama, model, prompt, b64, timeout):
    r = requests.post(f"{ollama}/api/generate", timeout=timeout, json={
        "model": model, "prompt": prompt, "images": [b64], "stream": False,
        # Deterministic so a rerun of the same variant is reproducible. Note this
        # is exactly why repeating one prompt proves nothing -- diversity has to
        # come from the prompts, not from sampling.
        "options": {"temperature": 0},
    }).json()
    return r.get("response", "")


def reconcile(readings, pi_count):
    """Turn several readings into one result plus a per-value confidence.

    Positional matching is not possible -- the variants enumerate dice in
    different orders by design -- so values are compared as MULTISETS. A value
    the variants agree on, with the same multiplicity, is solid; one that appears
    in some readings and not others is exactly the case that was wrong in
    testing.
    """
    counters = [Counter(d["value"] for d in v) for v in readings if v is not None]
    n = len(counters)
    if not n:
        return {"dice": [], "agreement": 0.0, "issues": ["no variant returned usable JSON"]}

    counts = sorted(sum(c.values()) for c in counters)
    expected = counts[len(counts) // 2]          # median, not union

    # CONSENSUS ONLY. Taking the union of what the variants said produced 12
    # dice from a 6-die pile, because one variant over-enumerated by listing
    # side faces as separate dice and every one of its extras was then reported
    # as a die. A reader that inflates the roll is worse than one that admits
    # it is unsure, so a value is only emitted at the multiplicity EVERY variant
    # supports.
    dice, issues = [], []
    for value in sorted(set().union(*[set(c) for c in counters])):
        agreed = min(c[value] for c in counters)
        for _ in range(agreed):
            dice.append({"value": value, "confidence": "high",
                         "seen_in": f"{n}/{n} readings", "note": None})

    # Anything the variants only partly support is reported as an unresolved
    # slot with its candidates, never as a value in the roll.
    disputed = {}
    for value in sorted(set().union(*[set(c) for c in counters])):
        extra = max(c[value] for c in counters) - min(c[value] for c in counters)
        if extra > 0:
            disputed[value] = sum(1 for c in counters if c[value] > 0)

    unresolved = max(0, expected - len(dice))
    if unresolved:
        issues.append(
            f"{unresolved} die/dice could not be read reliably -- candidates: " +
            ", ".join(f"{v} (in {k}/{n} readings)" for v, k in sorted(disputed.items())))

    if len(set(counts)) > 1:
        issues.append(f"readings disagree on how many dice: {counts}")
    if pi_count is not None and pi_count != expected:
        issues.append(
            f"camera counted {pi_count} dice, readings suggest {expected} -- a die "
            f"may have been missed or invented")

    return {
        "dice": dice,
        "values": [d["value"] for d in dice],
        "high_confidence": len(dice),
        "expected_dice": expected,
        "unresolved": unresolved,
        "disputed": disputed,
        "total": len(dice),
        "agreement": round(len(dice) / expected, 2) if expected else 0.0,
        "issues": issues,
        "per_variant": [[d["value"] for d in v] if v else None for v in readings],
    }


def read_roll(image_bytes, pi_count=None, ollama="http://10.0.0.5:11434",
              model="gemma3:27b", timeout=300, provider=None,
              anthropic_key=None, anthropic_model=None, anthropic_effort=None,
              code_token=None, code_model=None, variants=None):
    """Read a tray image. provider: 'claude-code' | 'anthropic' | 'ollama' | None.

    Auto order is claude-code, then anthropic, then ollama. Claude Code first
    because it is the only provider measured to get the d12/d20 split right, and
    it costs nothing per roll under a subscription. Ollama last as the fallback
    that keeps the table working when the network is down -- which matters for
    something that has to work at a table.
    """
    # Caller supplies these (see settings.py, which owns the env-then-stored
    # precedence). Falling back to the environment here keeps read_roll usable
    # standalone without duplicating that precedence in two places.
    key = anthropic_key if anthropic_key is not None else os.environ.get("ANTHROPIC_API_KEY", "")
    a_model = anthropic_model or os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
    if provider is None:
        # Claude Code first when a token is present: measured, it is the only
        # provider that got the d12/d20 split right, and it costs nothing per
        # roll under a subscription.
        provider = ("claude-code" if code_token else
                    "anthropic" if key else "ollama")
    if provider == "anthropic" and not key:
        return {"dice": [], "values": [], "total": 0, "agreement": 0.0,
                "issues": ["ANTHROPIC_API_KEY is not set"], "provider": "anthropic",
                "per_variant": [], "variant_names": []}

    b64 = base64.b64encode(image_bytes).decode()
    # Claude Code is an agent, not a completion call: ~105 s per opus call
    # against ~1.5 s for Ollama. Three of those is five minutes, so the
    # ensemble defaults to ONE prompt there. The Pi's independent count is
    # still cross-checked, so a missed die is still caught -- what is lost is
    # per-die disagreement, which is the cheaper of the two signals.
    prompts = PROMPTS[:1] if provider == "claude-code" else PROMPTS
    if variants:
        prompts = PROMPTS[:max(1, int(variants))]
    results, raw = [], {}
    for name, prompt in prompts:
        try:
            if provider == "claude-code":
                text = _claude_code(prompt, image_bytes,
                                    code_model or "claude-opus-5",
                                    code_token, timeout)
            elif provider == "anthropic":
                text = _anthropic(prompt, image_bytes, a_model, key, timeout,
                                  anthropic_effort)
            else:
                text = _ask(ollama, model, prompt, b64, timeout)
            raw[name] = text.strip()[:400]
            results.append(_parse(text))
        except Exception as e:
            raw[name] = f"ERROR: {e}"
            results.append(None)

    result = reconcile(results, pi_count)
    result["provider"] = provider
    result["model"] = ({"claude-code": code_model or "claude-opus-5",
                        "anthropic": a_model}.get(provider, model))
    result["variant_names"] = [n for n, _ in prompts]
    result["raw"] = raw
    result["pi_count"] = pi_count
    return result
