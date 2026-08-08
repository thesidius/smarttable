# smarttable — dice-reading camera tower

A camera tower over a TTRPG dice tray that reads what was rolled, so the values
can be fed to a game system without anyone typing them in.

## Hardware

| | |
|---|---|
| Camera head | Raspberry Pi 3 B + Camera Module V2 **NoIR** (IMX219), full sensor 3280×2464 |
| Control panel / inference | Windows workstation, RTX 4090, Ollama + Claude Code |
| Tray | 3D-printed, 45° inside edges, matte dark grey |

## Layout

```
pi-tower/     runs ON the Pi
  camera_service.py      HTTP camera head: capture + tray crop
  run_camera_service.sh  launcher: the unit's ExecStart, and how to run it by hand
  dicecam.service        systemd unit -- starts the camera at boot
  tray_geometry.py       rescale a tray calibration between resolutions
  tray_framing_check.py  mount aid: framing, uniformity, calibration
  camera_check.py        first-light / exposure characterisation
  noir_contrast_test.py  IR vs visible numeral-contrast test

webapp/       runs on the workstation
  app.py            control panel + roll pipeline
  seg_service.py    SAM2 instance segmentation, port 8090, own venv (see below)
  roll_reader.py    reading providers (Claude Code / Anthropic API / Ollama)
  settings.py       credentials, stored OUTSIDE this tree

docs/         findings, with the measurements behind them
```

## Running it

The camera starts itself at boot (`dicecam.service`). To check on it:

```bash
ssh paul@10.0.0.23 'systemctl status dicecam'
```

```bash
cd webapp && python app.py     # then http://<workstation>:5000
```

```bash
.venv-ml/Scripts/python.exe webapp/seg_service.py
```

**Three services, and the third needs its own interpreter.** The control panel
runs on Python 3.14, for which no CUDA torch wheel exists; SAM2 needs 3.13 +
torch-cu126. Hence `.venv-ml` and a separate process — which it wants to be
anyway, because the model takes ~20 s to load and would otherwise reload on
every roll. The header shows a `sam2` pill; if it is red, this is what is not
running.

**The NoIR tuning cannot be set by environment variable.** libcamera defaults to
the IR-cut tuning, and `LIBCAMERA_RPI_TUNING_FILE` is silently discarded —
picamera2 pops it in its own constructor unless handed a `tuning=` argument. The
path goes through that argument now. Wrong tuning does not error; it quietly
costs ~0.14 Otsu separability on every frame.

## Credentials

Never in this tree. `~/.dicecam/settings.json`, set via the Settings tab.

`sk-ant-oat…` is a **Claude Code OAuth token** (drives `claude -p` as a
subprocess). `sk-ant-api…` is an **API key** (`x-api-key` on the Messages API).
Both start `sk-ant-` and swapping them returns `401 invalid x-api-key`, which
reads like a bad key rather than the right credential at the wrong door — so
credentials are filed by what they are, not by which box they were typed into.

## What works, and what does not

Honest status, because a tool that overstates itself is worse than one that says
nothing.

**Works.** Capture with locked exposure that survives restarts; tray
calibration; **separating touching dice** (SAM2: 8/8 scattered, 7/7 in a tight
pile); reading values and die types via a vision model on isolated crops.
Capture-and-segment is 6.5 s end to end.

**Does not.** Reliable die-type identification — d12 vs d20 is still the weak
spot, and no configuration is yet both fast and accurate.

**Deleted 2026-08-07.** The classical detector (`dice_detect.py`) and the Pi's
independent die count. The distance-transform peak count is a proxy for "how
many dice" that only holds when blobs are round and equal-sized, which a mixed
set violates — it reported 20, 27, 36 and 73 dice for 8, while costing 67 s of
Pi 3 CPU per capture. SAM2 does that job now, on the workstation's GPU, in 3.5 s.

**Measured, so nobody re-derives it:**

- The NoIR tuning file is worth ~0.14 Otsu separability. libcamera never selects
  it automatically, and `LIBCAMERA_RPI_TUNING_FILE` does not work — picamera2
  pops it. Pass `Picamera2(tuning=...)`. Verify against libcamera's own log, not
  against the variable you set.
- A health check that reports your own input is not a health check. `/health`
  said the NoIR tuning was active while libcamera logged `imx219.json`, because
  it was reading the environment variable rather than the outcome.
- Exposure time and analogue gain are **interchangeable** on this rig: 47 ms at
  gain 4.8 and 227 ms at gain 1.0 gave the same brightness, noise and clipping.
  The noise is photon-limited. Tune total brightness, not the split.
- `ExposureTime` is capped by frame duration, not by the sensor. `camera_controls`
  advertises 11.77 s; a request for 227 ms silently became 47 ms until
  `FrameDurationLimits` was raised to match. Always read back what was applied.
- Auto-exposure overshoots, because it meters the dark tray floor rather than the
  numerals: plain AE clipped 0.233% of pixels, essentially all on dice faces.
  Metering the tray's top 0.5% instead and bisecting for it clipped **0.022%**.
  `POST /autoexpose` does this and locks; it is what the Mount tab's auto button
  runs.
- Fix the gain during that search rather than inheriting it, or the answer
  depends on history: the same tray metered to 55 ms with a stale gain of 4.0 and
  117 ms from a clean state. Both correct; only one reproducible.
- Under coloured light, OpenCV's fixed `BGR2GRAY` weights can hand 59% of the
  signal to a dead channel. Pick the channel per frame.
- Opposite faces of a d20 sum to 21, so the visible hemisphere holds exactly one
  of each complementary pair. Useful as a validity check.
- A "step 6" pattern in d20 neighbours is a **coincidence** — disproved on the
  second die.
- Self-reported model confidence is worthless: `"high"` with `"clear view"` on a
  die whose face is not in the image. Cross-prompt disagreement is the signal
  that actually tracks error.

See `docs/` for the numbers behind each of these.
