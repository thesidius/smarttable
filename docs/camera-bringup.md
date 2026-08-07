# Camera bring-up — prototype tower #1

Running log + reference for getting the dice-reading camera verified and
characterized. First prototype, single non-modular tower.

## Hardware

| Item | Detail |
|---|---|
| SBC | Raspberry Pi 3 Model B v1.2 |
| Camera | Raspberry Pi Camera Module V2 **NoIR** (IMX219, 8MP) |
| Case/display | SmartiPi Touch 1, 7" official touchscreen |
| OS | Raspberry Pi OS 64-bit Desktop, **Trixie (Debian 13)**, Python 3.13.5 |
| Host | `dicecam.local` → **10.0.0.23**, user `paul` |

Not part of this Pi's setup: GMKtec G5 mini PC (N97, 12GB, Win11) — later
inference/orchestration.

### CSI port gotcha

On the Pi 3 B the camera and display CSI ports look identical. **The camera
port is the one nearer the HDMI end of the board.** This already cost time
once during display bring-up. Ribbon orientation: blue stiffener toward the
ethernet/USB side, bare contacts toward the HDMI side. Always power down
before reseating — hot-plugging CSI can kill the module.

Ribbon length: the stock cable may be too short depending on final mount
distance. Plan for a **300–500 mm** CSI cable once the geometry below is
physically built.

## Status

- [x] OS flashed, SSH + VNC working
      (early hiccups: display ribbon port confusion; VNC defaulting to the
      small physical panel resolution — both resolved)
- [x] SSH key auth from this workstation — `paul@10.0.0.23`
- [x] Camera detected — imx219, all 8 modes enumerate
- [x] Deps installed — **already present on the image, nothing to do**
- [x] `camera_check.py` clean: well exposed, locked capture holds
- [ ] NoIR contrast test — needs an IR illuminator
- [ ] Mounted to target geometry, full tray + 4-corner dice in frame

## First light — 2026-08-05

Handheld/unmounted over the tray, ambient warm room light. Both captures in
`test-data/firstlight/`.

**Camera is healthy.** Ribbon is in the correct port, sensor enumerates as
`imx219` at 3280×2464, 4 resolutions × {10-bit, 8-bit}. `ExposureTime`
75 µs – 11.77 s, `AnalogueGain` 1.0 – 10.67.

**Exposure lock holds.** AE converged to 29999 µs / gain 3.4 / ColourGains
(1.967, 1.543) at 367 lux; on read-back after freezing: 29980 µs, gain and
colour gains exact. 19 µs of drift is pipeline quantisation, not AE hunting.

**Focus is good** — this was nearly a false alarm. Frame-wide Laplacian
variance read 28, which looks like a badly defocused lens; the die face itself
measures 135 and individual glitter specks resolve crisply. The flat tray,
desk and backdrop dominate the frame and crush the average. `camera_check.py`
now tiles the frame and reports the *sharpest* tile, which is the honest
question — "does anything in this scene resolve" — rather than "is the average
pixel busy". Same scene now reads 95 best-tile vs 28 frame-wide.

**Baseline numeral readability** (ROI on one d6 face, ambient light,
**correct NoIR tuning** — see below):

| channel | otsu_separability | class_gap |
|---|---|---|
| **R** | **0.845** | **149.8** |
| gray | 0.806 | 128.3 |
| G | 0.796 | 125.9 |
| B | 0.774 | 92.7 |

Threshold for GOOD is 0.55, so there's real headroom. That's what the IR test
has to beat, or at least not collapse against.

## The tuning file — read this before trusting any colour

**libcamera loads the wrong tuning file for this camera by default.**

The NoIR module is electrically identical to the standard V2 and reports the
same sensor id, so libcamera cannot tell them apart and falls back to
`imx219.json` — the tuning calibrated for the IR-cut-filtered version.
`imx219_noir.json` ships alongside it for exactly this case but is never
selected automatically.

The symptom is the heavy magenta cast on the first-light frames. Measured
whole-frame channel means:

| tuning | R | G | B | R/G |
|---|---|---|---|---|
| `imx219.json` (default) | 206.9 | 32.2 | 85.7 | **6.43** |
| `imx219_noir.json` | 121.0 | 118.3 | 121.7 | **1.02** |

The wrong tuning's AWB pushed red gain *up* to 1.967; the correct one pulls it
*down* to 0.822 — a 2.4× swing — because it knows the red channel is carrying
an IR load.

**This is not cosmetic.** On the same die face, same scene, same moment:

| | otsu_separability | class_gap |
|---|---|---|
| `imx219.json` | 0.668 | 82.3 |
| `imx219_noir.json` | **0.806** | **128.3** |

Wrong tuning was costing ~0.14 separability and 46 grey levels of
numeral-to-body distance. Both scripts now set
`LIBCAMERA_RPI_TUNING_FILE=/usr/share/libcamera/ipa/rpi/vc4/imx219_noir.json`
before picamera2 initialises (it has no effect if set later). Export the
variable yourself to override.

**What this does not do:** it does not remove IR. It cannot — there is no
filter. It rebalances the channels so the output looks neutral, but the IR
energy is still mixed into every channel. Under a dedicated IR illuminator
colour will still behave oddly, and that is expected.

Anything using the camera outside these scripts — `rpicam-hello`, `rpicam-jpeg`,
future capture daemons — needs the same environment variable, or it gets the
magenta version.

Set in `~/.bashrc` on the Pi (line 114) as of 2026-08-05. Verified scope:

| context | inherits it |
|---|---|
| interactive terminal (VNC, touchscreen, SSH login session) | yes |
| non-interactive SSH — `ssh paul@… "rpicam-hello …"` | **no** |
| login shell, GUI-menu-launched apps | **no** |
| systemd service | **no** |

Debian's `.bashrc` returns early for non-interactive shells, so that is the
expected ceiling, not a misconfiguration. Remote one-shot commands must pass
the variable inline. **When the capture daemon is written it will need an
explicit `Environment=` line in its unit file** — it will not inherit this,
and the failure mode is silent: correct-looking captures with ~0.14 less
separability. `/etc/environment` is the system-wide fix if that gets
tiresome.

### Open items from first light

- **Gain 3.3–3.4 is high.** 367 lux is dim, so AE is paying in noise. Adding
  controlled illumination should let gain drop toward 1.0, which will sharpen
  numeral edges more than any algorithm change will.
- **`camera_properties["Rotation"]` reports 180** and the frame appears
  inverted. Resolve deliberately at mount time with a known-orientation
  target (write TOP on a card) rather than guessing from die numerals —
  numerals sit at arbitrary angles and prove nothing.
- **1.5% blown highlights**, all in the bright reflection at the top of the
  frame, outside the tray. Harmless now; it resolves once the camera is
  mounted and the tray fills the frame. Worth re-checking then.
- **These dice are a hard case** — dark translucent bodies with glitter.
  Specular sparkle produces bright point highlights that will read as false
  edges. Good news if it works; don't generalise a failure here to plain dice.

## Target mounting geometry

Derived previously to keep a 7.5 × 5" dice tray fully in frame with margin on
both FOV axes. **Do not re-derive — build to these numbers.**

| Parameter | Value |
|---|---|
| Camera height above tray floor | 8.14" |
| Tilt from vertical | 21.3° |
| Orientation | portrait — camera module physically rotated 90° |
| Long FOV axis (62.2° on IMX219 V2) | runs along the tray's **long** axis |
| Tray | 7.5 × 5" |

IMX219 V2 FOV is 62.2° × 48.8°. "Portrait" here means the module is rotated in
its mount; libcamera cannot rotate 90° in the pipeline, so any 90° rotation in
software is a `cv2.rotate()` on the array and **does not change the FOV** —
`camera_check.py --rot 90` only affects the saved image.

## The NoIR question (test early)

The module has no IR-cut filter. Two outcomes:

- **Numerals hold contrast under IR** → IR illumination is invisible to
  players and immune to ambient room light. Big win.
- **Numerals wash out under IR** → many pigments are effectively transparent
  in near-IR, so painted numerals can vanish into the die body. Then: visible
  light + an IR-cut filter.

~20 minute test, two photos, dice untouched between them:

```bash
python3 noir_contrast_test.py capture --label visible   # room lights on, IR off
python3 noir_contrast_test.py capture --label ir        # room lights OFF, IR on
python3 noir_contrast_test.py compare ~/dicecam-captures/*_visible.jpg \
    ~/dicecam-captures/*_ir.jpg --roi X,Y,W,H --montage /tmp/noir.png
```

The ROI must tightly frame **one die face**, same pixels in both images.
Without it the tray, background and shadows dominate the statistics.

**`otsu_separability` is the number that matters** — how cleanly the numeral
splits from the die body by brightness. ≥0.55 with a class gap ≥60 is good;
below ~0.35 the paint isn't separable and thresholding will be fragile.

Worth testing more than one die material/colour — results vary a lot by
pigment. Black-on-white and white-on-black behave very differently in IR.

### Sourcing the illuminator (nothing on hand as of 2026-08-05)

**Get 850 nm, not 940 nm.** IMX219 quantum efficiency falls off steeply past
~850 nm — roughly a third the response at 940 nm — so 940 nm buys true
invisibility at the cost of needing substantially more emitter power or
longer exposure, and exposure is already at 30 ms. 850 nm puts a faint red
glow on the emitter itself but does not visibly light the tray. Start there;
only move to 940 nm if the glow actually bothers players at the table.

A 12 V CCTV illuminator board is the cheap way in. Whatever the source:

- **Diffuse it.** These dice are glossy with glitter. A bare LED array will
  punch specular hotspots across the faces and those read as false edges.
  Diffusion film, or bounce it.
- **Light at a grazing angle, not down the lens axis.** Coaxial light
  maximises glare straight back into the sensor; a low angle makes painted
  and engraved numerals throw micro-shadows, which *adds* contrast. This
  matters more than raw brightness.
- **Aim to drop AnalogueGain to ~1.0.** Current 3.3 is costing real noise.

**The fallback is cheap.** If IR fails, it's visible light plus an IR-cut
filter — either a stick-on filter over the NoIR module or just a standard
(non-NoIR) V2 module. Not a design dead-end, so don't over-invest in
de-risking this before testing it.

### Sequencing

Mount first, then test. Running the NoIR comparison through the final
geometry measures the lighting setup that will actually ship; a handheld test
at an arbitrary distance measures one that won't. Both halves of the test
must also be captured back-to-back without the dice moving, which is only
practical once the rig is fixed.

## Why exposure gets locked

Auto-exposure hunts. Drifting frame brightness will make dice-face reading
inconsistent shot to shot, and will throw false positives in frame-difference
motion detection later. So: let AE/AWB converge once, read the values back,
freeze them, and bake those numbers into the capture config.
`camera_check.py` prints the exact `set_controls()` block to paste.

If the locked exposure drifts on read-back, AE hasn't fully disengaged — rerun
with `--pin-framerate` to pin `FrameDurationLimits` too (exposure can't exceed
frame duration, so an unpinned frame rate can silently shorten it).

## Scripts

| File | Purpose |
|---|---|
| `pi-tower/install_camera_deps.sh` | apt deps + detection check with a troubleshooting ladder |
| `pi-tower/camera_check.py` | enumerate, report modes, auto capture + exposure verdict, locked capture |
| `pi-tower/noir_contrast_test.py` | IR vs visible numeral-readability capture + compare |

Deps go in via **apt, not pip** — picamera2 binds to the system libcamera
build, and Trixie blocks pip into the system environment (PEP 668) anyway.
As it turns out this image already ships all of them: python3-picamera2
0.3.36, python3-libcamera 0.7.1, python3-opencv 4.10.0, python3-numpy 2.2.4,
rpicam-apps 1.12.0. The script is kept for reflashes.

Note `picamera2` exposes no `__version__` attribute — ask dpkg instead.

## Focus note

The V2 module is manual-focus and ships set near infinity. It reads sharp at
roughly hand-held tray distance today, but **re-check after mounting at
8.14"** — that's a different working distance.

Read `laplacian_var_best_tile`, not the frame-wide figure. On a sparse scene
like a mostly-empty tray the frame-wide number is meaningless: first light
scored 28 frame-wide while the actual subject measured 135.
