# Control panel

Two services. Start both.

```bash
ssh paul@10.0.0.23 '~/pi-tower/run_camera_service.sh'
```

```bash
cd C:/Claude/smart-table/webapp && python app.py
```

Then open **http://10.0.0.5:5000**.

| | |
|---|---|
| `pi-tower/camera_service.py` | Flask on the Pi, port 8081. Owns the camera. |
| `pi-tower/run_camera_service.sh` | **Use this to start it** — see below. |
| `webapp/app.py` | Flask on TheBeast, port 5000. UI, Ollama, labels. |
| `dataset/labels.jsonl` | Append-only label log. |

## Always launch the Pi service via the shell script

`camera_service.py` tries to set `LIBCAMERA_RPI_TUNING_FILE` itself and **that
does not reliably work**. libcamera resolves the tuning path early enough that
an `os.environ` assignment at the top of the script can lose the race — verified
here: the service came up on `imx219.json` with that code in place, while a
plain `export` before launch gets `imx219_noir.json` every time.

The failure is silent. Wrong tuning does not error, it just costs ~0.14 Otsu
separability on every frame the service captures.

`/health` therefore reports the tuning **inherited from the launcher**, not
`os.environ` after the script has written to it — reporting the latter would
claim success in exactly the case that fails. The header pill shows
`NoIR tuning` (green) or `WRONG tuning` (amber). Trust the pill.

### Killing the service

`pkill -f camera_service.py` **kills your own SSH session** — the remote command
line contains that string, so pkill matches it. The bracket trick does not save
you either if the name also appears literally later in the command. Kill by port:

```bash
ssh paul@10.0.0.23 'fuser -k 8081/tcp'
```

## One camera configuration, never switched

The Pi runs the camera continuously in a video configuration with two streams:
`main` 1640×1232 RGB888 for capture/detect, `lores` 640×480 YUV420 for the MJPEG
preview. The ISP produces both from the same frame, so preview costs almost
nothing and never fights capture. Reconfiguring between video and still modes
per request is slow on a Pi 3 and a reliable source of hangs.

1640×1232 is not a compromise: the detection pipeline was tuned at that
resolution and a die is ~114 px across.

## The preview is not proxied

The `<img>` tag loads `http://10.0.0.23:8081/stream` directly. Control calls go
through TheBeast (one origin, no CORS), but routing live video through a second
Flask process adds a hop and a buffer on exactly the path where latency is felt
— while you are physically adjusting the mount and watching the picture move.

## Tabs

**Mount** — live preview, click-to-calibrate, framing/uniformity, exposure
lock. Click the four tray-*floor* corners in any order; clicks are in displayed
pixels and get scaled to sensor pixels, stored with the frame size so the quad
can be rescaled if applied at another resolution. Corners with no detail report
"flat — nothing to resolve" instead of a meaningless focus number.

**Detect** — capture, parameter sliders, overlay, crops. Re-detect runs on the
*last capture* without re-capturing, so tuning does not disturb the scene.

**Label** — gemma3 pre-labelling then human confirmation. Suggestions are
pre-labels, **not ground truth** — the model is verified on two crops. Labels
append; `read_labels()` collapses to last-write-wins, so corrections are just
another append. Append-only because labelling is the expensive, unrepeatable
work and a partial rewrite could lose it.

**Runtime** — reads a roll end to end. Pi captures, crops to the tray and
counts; Ollama reads; the app reconciles. ~12–18 s. Settle detection is still
explicitly "not implemented" (rolls are read on demand), and the pipeline
reality-check table stays, because a panel that implies something works when it
does not is worse than one that says nothing.

## The roll pipeline

```
Pi /roll ──► capture ─► tray crop (lossless PNG) ─► independent die count
                                    │
TheBeast /api/roll ◄────────────────┘
   └─► 3 DIFFERENT prompts to gemma3:27b ─► reconcile ─► values + confidence
                                                     └─► dataset/rolls.jsonl
```

**Three different prompts, not three repeats.** At temperature 0 the same
prompt returns byte-identical output every time, so repetition catches nothing —
it is consistently wrong when wrong. Diversity has to come from the framing.

**Confidence never comes from the model.** Asked to self-report, it returned
`"confidence": "high"` with `"Clear view of the '9'"` for a die whose up-face
is not recoverable from the image. Confidence here is derived from two things
it does not control: agreement across the prompt variants, and the camera's
independent count.

**Consensus is conservative, never the union.** An early version reported 12
dice from a 6-die pile because one variant over-enumerated — listing side faces
as separate dice — and every extra was emitted as a die. A value is now only
reported at the multiplicity *every* variant supports; anything partial becomes
an explicit unresolved slot listing its candidates. Over-reporting a roll is
worse than admitting uncertainty.

Which variant over-enumerates moves around: `typed` was clean in one run and
returned 10 values in the next. That is exactly why the consensus is taken
across variants rather than trusting any one.

Measured on a 7-dice pile: **[2, 2, 4, 5, 6] confident, 1 unresolved**, with
all three cross-checks firing (variants disagreed on count, camera count
differed from the readings). Notably the d20 — which a single prompt had
confidently misread as 14 — lands in the unresolved candidates instead of being
reported wrong.

`dataset/rolls.jsonl` is append-only and keeps the flagged rolls: an uncertain
roll marked uncertain is usable by a downstream game system; one silently
guessed is not.

## Settings tab — the Anthropic key

Enter it in the app: **Settings → Anthropic API key → save → test it**.

Precedence, copied from swadeledger's `ai_lib.js` and for the same reason:

    environment  ->  value saved in the app  ->  default

`ANTHROPIC_API_KEY` in the environment is the operator's kill switch. If a key
leaks the fix has to be "set the variable, restart", and that only works if the
environment cannot be silently outranked by something the UI can write. When
the environment supplies the key the Settings page says so, and saving there
has no effect until it is unset.

Handling:

- Stored at `~/.dicecam/settings.json`, **outside the project directory** — a
  secret in the working tree is one `git add .` from being published, and this
  tree gets shared.
- Read in exactly one place (`settings.key()`), used at exactly one call site
  (the `x-api-key` header). Never logged, never in an error message, never in a
  response.
- `GET`/`POST /api/settings` return only whether a key exists and where it came
  from. Verified: saving a key returns a body that does not contain it.
- The input is `type=password` and is cleared after saving; the stored value is
  never rendered back into the DOM.
- **Test it** makes a 1-token request — the cheapest question that still
  exercises auth, model name and network path. A bad key reports
  `401 invalid x-api-key` and nothing else; verified the failure message does
  not echo the key.

### Model list comes from the API, not a hard-coded list

`GET /api/settings/models` asks `https://api.anthropic.com/v1/models` with the
configured key and feeds a `<datalist>`. A baked-in list goes stale the moment a
model ships or retires, and then offers something the key cannot reach — which
surfaces as a 404 mid-roll rather than at the moment of choosing. Asking the API
means the dropdown can only offer what is genuinely available *to this key*.

A `datalist` rather than a `select` on purpose: it gives the dropdown while
still allowing a model name that is not in the list. If the current value is not
among the returned models the note says so explicitly, since that is exactly the
typo that would otherwise fail later.

Sorted newest first — the list arrives roughly chronological and the useful
default is a recent model, not whatever sorts first alphabetically.

The list loads automatically once a key is present, and can be refreshed with
**load available models**. Without a key it refuses cleanly
(`{"ok":false,"error":"no key configured"}`) and typing still works.

Without a key, reading falls back to Ollama automatically. That fallback is
deliberate: it is local and free, and it keeps the table working when the
network or the API is down.

## Tray boundary: saved calibration ONLY, never auto-detected

`camera_service.load_quad()` reads `~/.config/dicecam/tray.json` and returns
`None` if it is absent. `/analyze`, `/roll` and `/framing` all error out rather
than guess. The service never calls `find_tray_quad()`.

Auto-detection still exists in `tray_framing_check.py` as a standalone
convenience, and it is deliberately not in the service path: it found the LEGO
baseplate instead of the tray, then leaked into the desk, because a dark matte
tray on a dark desk spans ~14 grey levels. The camera is fixed once mounted, so
this is a one-time measurement, not a per-frame problem.

## The exposure lock survives restarts

It did not, and that quietly corrupted a run of measurements: every service
restart built a fresh Camera with AE/AWB free, so the segmentation threshold
drifted with the exposure and detection quality wandered for no visible reason.

Now: `/lock` writes `~/.config/dicecam/exposure.json`, startup re-applies it,
and `/unlock` deletes it so an explicit release is not silently undone on the
next restart. Verified across a restart — same exposure, same gains, `lock_lux`
preserved, staleness check still armed.

A restored lock can of course be stale; `_lock_stale()` still flags that. A
stale lock is better than none, because it is at least reproducible.

## An exposure lock is only valid for the light it was taken under

This bit us. Exposure and white balance were locked under purple LED lighting,
freezing ColourGains at (1.149, **3.276**) — a heavy blue boost. When a lamp
was added, those gains stayed frozen, and the result looked like a stubborn
purple camera fault. It was not: it was a correct lock for lighting that no
longer existed.

Unlocking and re-converging changed the gains to (1.302, 1.881) and the tray
floor from R 46 / G 5.5 / B 191 to R 103 / G 69 / B 102 — **green recovered
12.5×**.

`/health` now records the lux at lock time and reports `lock_stale` when the
current reading drifts more than ±50%. The control panel shows it as a red
banner. Locking is still right — AE drift breaks capture consistency and will
throw false positives in settle detection — but **re-lock after any lighting
change**.

Residual note: a slight warm/magenta cast remains after re-converging and is
expected. The module is NoIR with no IR-cut filter, so infrared lifts red and
blue relative to green. The NoIR tuning rebalances it; nothing short of a
physical filter removes it.

## Honest warnings, shown not buried

The header pills and banners surface: Pi unreachable (with the start command),
wrong colour tuning, no tray calibration, and **exposure not locked**. The last
one matters — AE drift makes captures inconsistent and will generate false
positives in settle detection later.
