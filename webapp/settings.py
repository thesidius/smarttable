#!/usr/bin/env python3
"""
settings.py -- operator settings, including the Anthropic key.

Precedence, copied from swadeledger's ai_lib.js and for the same reason:

    environment  ->  stored value  ->  default

For the key specifically that ordering is doing real work rather than being a
tidiness preference. ANTHROPIC_API_KEY is the kill switch: if a key leaks, the
fix has to be "set the environment variable, restart", and that fix only works
if the environment cannot be silently outranked by something the UI can write.

Stored OUTSIDE the project directory (~/.dicecam/settings.json), not next to
the code. A secret living in the working tree is one `git add .` away from
being published, and this project's tree is full of things that do get shared.

The key is read in exactly one place -- key() -- whose return value is used at
exactly one call site, the x-api-key header. It is never logged, never put in
an error message, and never returned by an endpoint. Where it came from is
reportable; what it is, is not.
"""

import json
import os
import stat

PATH = os.path.expanduser("~/.dicecam/settings.json")

DEFAULTS = {
    "anthropic_model": "claude-opus-5",
    "anthropic_effort": "low",
    # Opus: measured, it is the only configuration that got the d12/d20 split
    # right on an 8-dice tray. Haiku is ~10x faster and got two of four wrong.
    "claude_code_model": "claude-opus-5",
}


def _load():
    if not os.path.exists(PATH):
        return {}
    try:
        with open(PATH) as f:
            return json.load(f)
    except (ValueError, OSError):
        return {}


def _save(data):
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    with open(PATH, "w") as f:
        json.dump(data, f, indent=2)
    try:
        # Best effort: owner-only. On Windows this is close to a no-op, so it
        # is a defence-in-depth gesture rather than the protection itself --
        # the real protection is that the file is not in the project tree.
        os.chmod(PATH, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def get(env_name, field, fallback=""):
    from_env = os.environ.get(env_name, "")
    if from_env:
        return from_env
    return _load().get(field) or fallback


def classify(cred):
    """Which door does this credential open? Prefix only -- never the value.

    These are two different credentials for two different services and they are
    trivially confusable: both start `sk-ant-`, both come from Anthropic, and
    pasting the wrong one produces `401 invalid x-api-key`, which reads like a
    bad key rather than the right key in the wrong field. That happened here.
    """
    c = (cred or "").strip()
    if c.startswith("sk-ant-oat"):
        return "claude-code"      # OAuth token -> `claude -p` subprocess
    if c.startswith("sk-ant-api"):
        return "anthropic"        # API key -> x-api-key on the Messages API
    if c.startswith("sk-ant-"):
        return "unknown-anthropic"
    return "unknown"


def _migrate():
    """Move a misfiled credential to the field that matches what it is.

    Runs once at import. A Claude Code OAuth token saved into the API-key field
    is not a bad key -- it is the right credential behind the wrong door, and it
    fails as `401 invalid x-api-key`, which reads like the former. Rather than
    ask for it to be pasted again, put it where it works.
    """
    data = _load()
    k = data.get("anthropic_key")
    if k and classify(k) == "claude-code" and not data.get("claude_code_token"):
        data["claude_code_token"] = k
        data.pop("anthropic_key", None)
        _save(data)
        print("[settings] moved a Claude Code OAuth token out of the API-key "
              "field and into claude_code_token", flush=True)


def key():
    """The Anthropic API key. The ONLY place it is read."""
    return get("ANTHROPIC_API_KEY", "anthropic_key", "")


def code_token():
    """The Claude Code OAuth token. The ONLY place it is read."""
    return get("CLAUDE_CODE_OAUTH_TOKEN", "claude_code_token", "")


def code_token_source():
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", ""):
        return "environment"
    return "app" if _load().get("claude_code_token") else None


def code_model():
    return get("CLAUDE_CODE_MODEL", "claude_code_model",
               DEFAULTS["claude_code_model"])


def key_source():
    """Where the key came from -- never what it is."""
    if os.environ.get("ANTHROPIC_API_KEY", ""):
        return "environment"
    return "app" if _load().get("anthropic_key") else None


def model():
    return get("ANTHROPIC_MODEL", "anthropic_model", DEFAULTS["anthropic_model"])


def effort():
    return get("ANTHROPIC_EFFORT", "anthropic_effort", DEFAULTS["anthropic_effort"])


def public():
    """Everything the UI is allowed to know. Deliberately excludes both secrets."""
    return {
        "key_present": bool(key()),
        "key_source": key_source(),
        "env_overrides_key": bool(os.environ.get("ANTHROPIC_API_KEY", "")),
        "anthropic_model": model(),
        "anthropic_effort": effort(),
        "code_token_present": bool(code_token()),
        "code_token_source": code_token_source(),
        "claude_code_model": code_model(),
        "path": PATH,
    }


def update(body):
    """Write settings. Returns public() -- never the key."""
    data = _load()

    # A credential is filed by WHAT IT IS, not by which box it was typed into.
    # The two are trivially confusable and the failure is misleading, so route
    # it correctly and report where it went rather than storing it somewhere it
    # can only produce a 401.
    routed = None
    for field in ("anthropic_key", "claude_code_token"):
        if field not in body:
            continue
        v = (body.get(field) or "").strip()
        if v == "":
            data.pop(field, None)                # explicit clear
            continue
        kind = classify(v)
        target = ("claude_code_token" if kind == "claude-code" else
                  "anthropic_key" if kind == "anthropic" else field)
        data[target] = v
        routed = {"entered_as": field, "stored_as": target, "kind": kind}

    for field, env_name in (("anthropic_model", "ANTHROPIC_MODEL"),
                            ("anthropic_effort", "ANTHROPIC_EFFORT"),
                            ("claude_code_model", "CLAUDE_CODE_MODEL")):
        if field in body:
            v = (body.get(field) or "").strip()
            if v:
                data[field] = v
            else:
                data.pop(field, None)

    _save(data)
    out = public()
    if routed:
        out["routed"] = routed
    return out


_migrate()
