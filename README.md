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
  camera_service.py      HTTP camera head: capture, tray crop, detect, count
  run_camera_service.sh  launcher -- USE THIS, see below
  dice_detect.py         segmentation, clump detection, watershed split
  crop_pipeline.py       canonical die crops + manifest for training data
  tray_framing_check.py  mount aid: framing, uniformity, calibration
  camera_check.py        first-light / exposure characterisation
  noir_contrast_test.py  IR vs visible numeral-contrast test

webapp/       runs on the workstation
  app.py            control panel + roll pipeline
  roll_reader.py    reading providers (Claude Code / Anthropic API / Ollama)
  settings.py       credentials, stored OUTSIDE this tree

docs/         findings, with the measurements behind them
```

## Running it

```bash
ssh paul@10.0.0.23 '~/pi-tower/run_camera_service.sh'
```

```bash
cd webapp && python app.py     # then http://<workstation>:5000
```

**Always launch the Pi service via `run_camera_service.sh`.** libcamera cannot
tell a NoIR module from a standard V2 and defaults to the IR-cut tuning; setting
the environment variable from inside Python loses the race. The wrong tuning does
not error, it just quietly costs ~0.14 Otsu separability on every frame.
`/health` reports which tuning actually won.

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

**Works.** Capture with locked exposure that survives restarts; tray calibration;
detecting that dice are present; reading values and die types via a vision model;
cross-checking the reader against an independent count.

**Does not.** Separating dice that touch. The distance-transform peak count is a
proxy for "how many dice" that only holds when blobs are round and equal-sized,
which a mixed dice set violates — on one frame it reported 20 dice for 8. Asking
the vision model for bounding boxes instead returned one box per die including a
four-die cluster, and is the direction this is heading.

**Measured, so nobody re-derives it:**

- The NoIR tuning file is worth ~0.14 Otsu separability. libcamera never selects
  it automatically.
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
