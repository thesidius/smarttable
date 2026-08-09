# Control panel

Three services. The camera one starts itself.

```bash
ssh paul@10.0.0.23 'systemctl status dicecam'
```

```bash
cd C:/Claude/smart-table/webapp && python app.py
```

Then open **http://10.0.0.5:5000**.

| | |
|---|---|
| `pi-tower/camera_service.py` | Flask on the Pi, port 8081. Owns the camera. |
| `pi-tower/dicecam.service` | systemd unit — starts the camera at boot. |
| `pi-tower/run_camera_service.sh` | The unit's ExecStart; also how to run it by hand. |
| `webapp/app.py` | Flask on TheBeast, port 5000. UI, Ollama, labels. |
| `dataset/labels.jsonl` | Append-only label log. |

## The camera starts at boot

`pi-tower/dicecam.service`, enabled. Install it with:

```bash
sudo install -m 644 dicecam.service /etc/systemd/system/ && sudo systemctl enable --now dicecam
```

Two ordering facts it depends on, both learned the hard way from a real reboot:

**The USB stick must be in `/etc/fstab`.** Left to udisks2 it mounts only when a
desktop session logs in — far too late, and not guaranteed. `camera_service.py`
picks its work directory once, at import, so a service that beats the mount
falls back to the 7.4 GB SD card, which has already filled to 100% once and made
captures vanish with "no such capture". The unit orders itself after
`media-paul-pi_storage.mount`; `After=` without `Requires=`, so a pulled stick
degrades to the SD card and says so in `work_source` rather than refusing to
start.

**The camera can lose a race for contiguous memory.** Full-sensor RGB888 plus
the RAW stream want ~100 MB of CMA, and the desktop's KMS framebuffers take
nearly all of it while booting: measured 3 MB free of 256 MB at 49 s into a
boot, and 217 MB free ninety seconds later. The first boot crashed with
`OSError: [Errno 12] Cannot allocate memory` and only recovered via
`Restart=always`, 191 s after the reboot. `Camera._open()` now retries with
backoff instead — 62 s, zero restarts. Reserving more CMA would permanently
cost a 1 GB Pi general-purpose RAM to fix a problem that lasts a minute.

## Focus (Camera Module 3 only)

The V2 was fixed-focus. The CM3 has a voice-coil lens that defaults to
**continuous** autofocus, which is wrong here in a way that does not announce
itself: it re-hunts whenever the dice change, so two captures of the same tray
are focused differently. That breaks comparability between rolls, and it would
quietly destroy the template matching in `docs/geometric-face-reading.md`,
which assumes a fixed optical path. The camera and tray never move, so focus is
a constant — find it once, pin it, persist it.

`POST /autofocus` sweeps lens position and keeps the sharpest, measured **on
the tray**, then pins it. `POST /focus {"lens_position": 4.1}` sets it directly
(dioptres, i.e. 1/metres). `{"mode": "hardware"}` uses the sensor's own AF.

**Why not just use the sensor's autofocus.** Same reason `/autoexpose` meters
the dice rather than the frame: the hardware optimises for a subject it chooses
itself. Measured, its AF cycle settled on 6.79 dioptres (~0.15 m) for a tray
that was ~0.24 m away — plausible-looking, and soft where it mattered. The
sweep found 4.09 and took central sharpness from **3.5 to 109.7**.

**Measure sharpness on real frames, not on a timer.** The first sweep waited a
fixed 0.35 s after each lens move. At 4608x2592 on a Pi 3 the grabber turns
over slower than that, so samples were contaminated by the previous position:
the peak beat its neighbours by 5%, inside the noise, and the same position
measured twice as sharp once genuinely settled. It now counts two whole frames
after the lens reports arrival. The margin went to 32%, and the answer moved a
step — the bug had been choosing the wrong position, not merely reporting it
imprecisely.

`sharpness_at_pin` is recorded so drift is detectable, rather than discovered by
eye on a blurry capture.

## Exposure: auto, locked, or manual

Four buttons on the Mount tab.

| | |
|---|---|
| **auto (meter the dice)** | Find the brightest exposure that does not blow out the dice faces, and lock there. This is the one to use. |
| **apply manual** | Set exposure, gain and white balance explicitly. |
| **lock where AE lands** | Freeze whatever ordinary auto-exposure converged to. |
| **release to AE** | Free-running AE/AWB. Drifts, so captures are not comparable. |

### Highlight-priority metering: `POST /autoexpose`

Ordinary AE meters the whole frame. The tray floor is most of that frame and it
is dark, so AE raises brightness until the *floor* is mid-grey — and the dice,
far brighter, saturate. A clipped numeral is unreadable at any resolution, so
this is the one exposure error that directly costs reads.

`/autoexpose` bisects exposure against the 99.5th percentile of the pixels
*inside the tray boundary*, targeting 245, then locks. Measured on the same
scene:

| | clipped | p99.9 |
|---|---|---|
| plain AE | 0.233% | 254 |
| highlight metering (117 ms @ gain 1.0) | **0.022%** | **233** |

Ten times less clipping, and it landed within 7% of the value found by hand.

Two implementation notes that matter:

**Gain is fixed during the search, not inherited.** Inheriting made the result
depend on history — the same tray metered to 55 ms because a gain of 4.0 was
left over from an earlier lock, where from a clean state it chose 117 ms. Both
are correct exposures; only one is reproducible. `analogue_gain` is a request
parameter defaulting to 1.0. It is not a quality knob (see the equivalence
below); it buys preview frame rate.

**The metric is a plain percentile, not "pixels brighter than the floor."** The
selective version looked more precise and failed: the first bisection step lands
on a deliberately extreme exposure, the frame saturates, every pixel equals the
floor, and "brighter than the floor" selects nothing — so the probe meant to
report *far too bright* reported *no dice here* and the whole run aborted. The
dice are far more than 0.5% of the tray, so the top 0.5% of it is dice at any
exposure, and at saturation it correctly reads 255.

`POST /exposure {"exposure_us": 110000, "analogue_gain": 1.0, "colour_gains": [0.82, 1.49]}`
— any subset; the rest is left alone. `GET /exposure` returns the current values
plus the sensor's ranges, so the UI bounds its sliders from the camera rather
than from a guess.

All three persist to `~/.config/dicecam/exposure.json` and are re-applied on
restart, along with the frame duration — without which a restored long exposure
would silently clamp back.

### The response reports what the sensor DID, not what you asked

A request for 227 ms came back as 47 ms with no error, because **exposure time
is capped by frame duration**: the video mode was running ~21 fps, and a frame
cannot expose for longer than it lasts. `camera_controls` cheerfully advertises
an ExposureTime maximum of **11.77 seconds** — the sensor's limit, not the
reachable one.

`set_manual()` now raises `FrameDurationLimits` to fit the requested exposure,
and reports the preview frame-rate cost rather than hiding it. It also reads the
applied values back from sensor metadata and flags any remaining gap. The
sliders snap to what was actually applied — leaving them on the request would
show a number the camera never used.

### Exposure time and analogue gain are interchangeable here

The intuitive theory — dice are stationary, so motion blur is free, so trade
gain for time and get a cleaner image — is **false at this light level**.
Measured on the same scene:

| | floor | noise | clipped |
|---|---|---|---|
| 47 ms @ gain 4.8 | 131.6 | 3.26 | 0.23% |
| 227 ms @ gain 1.0 | 132.1 | 3.33 | 0.25% |

Identical. The noise is photon-limited, and the same total light arrives either
way. Matching floor means confirm the two exposures really were equivalent, so
this is a measurement rather than a coincidence.

### What is worth tuning is total brightness, and the limit is clipping

At gain 1.0, sweeping exposure:

| exposure | floor | clipped | p99 |
|---|---|---|---|
| 45 ms | 82.0 | 0.000% | 118 |
| 70 ms | 110.8 | 0.000% | 156 |
| **110 ms** | **145.7** | **0.000%** | **194** |
| 160 ms | 175.9 | 0.059% | 221 |
| 227 ms | 201.9 | 0.326% | 240 |

**110 ms at gain 1.0 is the operating point**: the brightest setting that clips
nothing. A clipped numeral is unreadable at any resolution, and auto-exposure
walks straight into that — it meters the whole tray, most of which is dark
floor, and pushes brightness until the floor is mid-grey.

A caveat on how far to trust this: a synthetic numeral-vs-body contrast score
kept improving all the way to 227 ms, i.e. it disagrees with the clipping
argument at the top end. The only metric that settles it is whether the reader
gets the values right, and that has not been measured across this sweep. 110 ms
is the conservative choice, not a proven optimum.

## The NoIR tuning: the environment variable does NOT work

Setting `LIBCAMERA_RPI_TUNING_FILE` — from a launcher script, from `.bashrc`, or
from Python — is **silently discarded**. picamera2 manages the tuning file
itself and, constructed without a `tuning=` argument, does:

```
os.environ.pop("LIBCAMERA_RPI_TUNING_FILE", None)  # Use default tuning
                                    -- picamera2.py:337, v0.7.1+rpt20260609
```

It pops the variable *before* libcamera reads it, so the export loses every
time regardless of who wins the race. `camera_service.py` passes the path
through `Picamera2(tuning=...)` instead, which is the supported API.

This went unnoticed because the check was watching the wrong thing. `/health`
reported the environment variable, which *we* had set, so it read
`noir_tuning_active: true` while libcamera's own log said
`Using tuning file .../imx219.json`. It is now read back from picamera2 after
construction, where it reflects what libcamera was actually handed. To confirm
independently, ask libcamera:

```bash
ssh paul@10.0.0.23 "journalctl -u dicecam -b | grep 'Using tuning file'"
```

The failure is silent either way: wrong tuning does not error, it costs ~0.14
Otsu separability and 46 grey levels of numeral contrast on every frame.

### Killing the service

`pkill -f camera_service.py` **kills your own SSH session** — the remote command
line contains that string, so pkill matches it. The bracket trick does not save
you either if the name also appears literally later in the command. Kill by port:

```bash
ssh paul@10.0.0.23 'fuser -k 8081/tcp'
```

## One camera configuration, never switched

The Pi runs the camera continuously in a video configuration with two streams:
`main` 3280×2464 RGB888 for capture, `lores` 640×480 YUV420 for the MJPEG
preview. The ISP produces both from the same frame, so preview costs almost
nothing and never fights capture. Reconfiguring between video and still modes
per request is slow on a Pi 3 and a reliable source of hangs.

Full sensor because it buys accuracy *in the reader*, not in any CV silhouette:
at 1640×1232 the model typed d20s unstably; at 3280×2464 it got 4/4 right.

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

**Segment** (was Detect) — capture and segment with SAM2, **no reading**. Shows
the tinted mask overlay and the isolated crops the reader will receive. ~6.5 s,
against minutes for a read.

That split is the whole point of the tab. When a roll comes back wrong, "a mask
merged two dice" and "the reader misread a clean crop" produce the identical
symptom, and only one of them is worth paying a read to investigate. If each
die has its own outline in the overlay, segmentation was fine.

The old sliders — variance window, min/max blob area, crop context, watershed
toggle — are gone with the classical detector they tuned. Keeping them would
have invited re-tuning a method already measured at 20-to-73 dice for 8.

**Label** — pre-labelling then human confirmation, on the same SAM2 crops the
reader sees. Provider is selectable (Claude Code / Anthropic / Ollama); it used
to be hard-wired to Ollama, which meant pre-labelling used the weakest reader
available while the rolls it produces training data *for* went through Claude
Code. Suggestions are pre-labels, **not ground truth**.

Saving writes the crop PNG under `dataset/crops/`, keyed by capture id and mask
index, alongside the label. SAM2 crops are composited in memory and never exist
on the Pi, so a label naming a Pi filename would point at nothing — and a label
log whose crops cannot be reopened is useless as training data. Labels append;
`read_labels()` collapses to last-write-wins, so corrections are just another
append. Append-only because labelling is the expensive, unrepeatable work and a
partial rewrite could lose it.

**Runtime** — reads a roll end to end, via `/api/roll2`: Pi captures and crops,
SAM2 separates the dice, the reader reads each isolated crop, the app
reconciles. Settle detection is still explicitly "not implemented" (rolls are
read on demand), and the pipeline reality-check table stays, because a panel
that implies something works when it does not is worse than one that says
nothing.

## The roll pipeline

```
Pi /roll ──► capture ─► tray crop (lossless PNG)          1.6 s
                              │
TheBeast /api/roll2 ◄─────────┘
   ├─► SAM2 ─► one mask per die ─► crop on neutral grey    3.5 s
   └─► N crops ─► 2 DIFFERENT prompts ─► reconcile ─► values + confidence
                                                  └─► dataset/rolls.jsonl
```

**The Pi no longer counts.** It used to return an independent
distance-transform count as a cross-check. On an 8-dice tray that count
reported 20, 27, 36 and 73 — so it fired a mismatch warning on every roll, and
a warning that always fires is a warning nobody reads. It also cost 67 s of Pi
3 CPU per capture, which was the entire reason a capture was not near-instant.
Removing it took `/roll` to 1.6 s. SAM2's mask count stands alone.

**Three different prompts, not three repeats.** At temperature 0 the same
prompt returns byte-identical output every time, so repetition catches nothing —
it is consistently wrong when wrong. Diversity has to come from the framing.

**Confidence never comes from the model.** Asked to self-report, it returned
`"confidence": "high"` with `"Clear view of the '9'"` for a die whose up-face
is not recoverable from the image. Confidence is derived from something it does
not control: agreement across the prompt variants.

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
`None` if it is absent. `/roll` and `/framing` both error out rather than
guess. The service never calls `find_tray_quad()`.

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
