# Instance segmentation — implementation plan

Replaces the classical-CV detection path in `pi-tower/dice_detect.py`.

Written 2026-08-06. API details verified against **ultralytics 8.4.115**.

---

## Why this replaces the current approach

The current path segments on local variance, then tries to split merged blobs
with a watershed on the distance transform. `docs/dice-reading.md` records, with
measurements, that this cannot work:

- Touching dice merge into one variance blob.
- The watershed guard **correctly rejects every split** — distance-transform
  topography loses 27–52% of the pile and produces pieces of 5 241–32 645 px
  against real single dice of 7 833–22 902 px.
- Overlapping dice have **no waist** for a distance transform to cut at. Their
  union is convex. This is a property of the method, not a tuning failure.
- The peak count is unstable: "reported 7, then 6, then 5 across consecutive
  frames as segmentation noise moved the maxima."
- Circularity **inverts** for d12 vs d20 (measured 0.678 for a d12 against 0.790
  and 0.825 for two d20s) because mask noise exceeds the 0.907-vs-0.970 ideal gap.

Separating touching instances of the same object class is **instance
segmentation**. It is a learned-perception problem. No kernel size, threshold, or
seed-separation value fixes it.

### Two requirements, one model

Both blockers are solved by the same output:

1. **Separate touching and overlapping dice.**
2. **Identify die type per instance.** Non-negotiable — a d20 and a d12 rolled
   together can both show 6, and the game needs to know which is which.

An instance-segmentation model returns a mask **and** a class per detection. That
is exactly the two things needed, from one pass.

### This does not require changing the tray

Threshold-based segmentation failed because the tray floor (86–167 grey) sits
inside the dice range (near-black ~20–40 to pale ~180+), so no polarity works for
both ends. A learned model does not threshold on level. It keys on shape,
texture, and edges — the same cues a person uses to see eight dice in the photo.

**The green tray idea is shelved, not cancelled.** Revisit only if the model
underperforms after the camera change.

---

## Constraints

| | |
|---|---|
| Dice per roll | any number, any types — no fixed set, no count prior |
| Classes | `d4, d6, d8, d10, d100, d12, d20` (7) |
| Camera | Module 3 standard (IR-cut) replacing the V2 NoIR — **see sequencing** |
| Camera pose | fixed: 8.14″ above tray floor, 21.3° tilt, portrait |
| Tray | existing 3D print, 45° inside edges, matte dark grey — unchanged |
| Segmentation runs on | workstation (RTX 4090), not the Pi |
| Pi's job | capture, crop to tray quad, ship the frame |

### Sequencing constraint — read this before collecting anything

**Do not collect training data until the Module 3 is mounted and exposure is
locked.**

`docs/dice-reading.md` already records that the same die through `imx219.json`
vs `imx219_noir.json` differs by ~6× in R/G — "visibly a different image."
Swapping the sensor is a larger change than that. Every frame captured with the
NoIR is a different domain and will poison a training set.

Existing `test-data/` frames are fine for **Phase 0 validation**. They are not
training data.

Keep the per-crop provenance stamping (`ExposureTime`, `AnalogueGain`,
`ColourGains`) already in `crop_pipeline.py`. Add a `camera_model` field so NoIR
and Module 3 captures are separable after the fact.

---

## Phase 0 — validate SAM2 zero-shot  ⛔ GATE

**Purpose:** find out whether learned segmentation actually separates your dice
before investing in a dataset. Half a day. Do this first.

Run against the existing `test-data/` frames — the 7-dice pile and the 8-dice
scattered frame (4× d12, 4× d20).

```python
from ultralytics import SAM

model = SAM("sam2.1_b.pt")          # b=base. Try sam2.1_l.pt if b underperforms
results = model("test-data/<frame>.jpg")   # no prompts = automatic mask generation
```

Automatic mask generation parameters (verified signature,
`ultralytics.models.sam.predict.Predictor.generate`):

```
crop_n_layers=0, crop_overlap_ratio=0.341, crop_downscale_factor=1,
point_grids=None, points_stride=32, points_batch_size=64,
conf_thres=0.88, stability_score_thresh=0.95,
stability_score_offset=0.95, crop_nms_thresh=0.7
```

`points_stride` is the sampling grid density. At 32, a 1640×1232 frame gets a
32×32 grid — roughly 51 px spacing, so a ~200 px die receives ~16 sample points.
Adequate. Raise it if small dice are missed.

### SAM2 returns a mask hierarchy — you must filter

This is the part that will look broken if skipped. AMG returns masks at every
scale: the whole tray, each die, each die's top facet, each painted numeral,
wall seams, shadows. Expect 50–200 raw masks for 8 dice.

Filter, in this order:

1. **Inside the calibration quad.** Reuse `fit_quad_to_frame()` — reject any mask
   whose centroid is outside. This kills wall and floor masks immediately.
2. **Area band.** Measured single-die areas at 1640×1232 are 7 833 (d6) to
   22 902 (d20) px — a 2.9× spread. Scale with `quad_scale()`; do not hard-code.
   Allow generous margins: `0.5× min` to `1.5× max`.
3. **Nesting.** Drop any mask contained >80% within a larger kept mask. This
   removes facets and numerals, which are the most numerous false positives.
4. **Stability.** Keep SAM's own `stability_score_thresh` as a coarse filter.

### Gate criteria

Measure on at least three frames with known ground truth:

| metric | pass |
|---|---|
| 8 scattered dice → 8 masks | ≥ 7/8, no merges |
| **7-dice pile → separate masks** | **≥ 5/7 separated** |
| False positives after filtering | ≤ 1 per frame |
| Latency on the 4090 | < 5 s per frame |

**The pile row is the one that matters.** It is the case classical CV cannot do
at all. If SAM2 separates 5+ of 7, proceed to Phase 1. If it separates 2–3, try
`sam2.1_l.pt` and a higher `points_stride` before concluding.

If SAM2 fails the gate even at large: stop and reconsider. The fallback is a
second camera angle, not more software.

**Record the numbers in `docs/dice-reading.md` either way.** A negative result
here is as valuable as a positive one.

---

## Phase 1 — SAM2 as stage 1, ship it

If Phase 0 passes, this is a working system with **no training at all**.

```
frame → SAM2 AMG → filter → per-die masks → crop each → stage 2 (type + value)
```

- **Segmentation:** SAM2, workstation.
- **Type and value:** the existing VLM path in `webapp/roll_reader.py`. It
  already reads values correctly on clean single-die crops (4/4 on an isolated
  d20; correct on peak-centred crops from a pile). The reason it previously
  failed was clutter in the crop, which clean masks remove.
- **Independent count:** the number of surviving masks. Keep this. `README.md`
  records the VLM silently omitting a die while reporting high confidence — mask
  count is the only signal that catches it.

**Crop from the mask, not the bounding box.** Use the mask to composite the die
onto a neutral background before cropping. That is the whole point — it removes
the neighbouring dice that were causing the misreads. Keep `crop_pipeline.py`'s
square/context/PNG/no-rotation conventions and **bump `CROP_VERSION`**.

Ship this. It is a working table. Phases 2–3 are optimisation.

---

## Phase 2 — bootstrap the dataset

SAM2 is slow and general. YOLO is fast and specific to your dice, your tray, your
camera. Getting there needs labels, and SAM2 generates them.

### Collection

- **After** the Module 3 is mounted and exposure locked.
- Target **200–400 frames**. That sounds low for instance segmentation and is
  fine here: the camera pose is fixed, the tray is fixed, and the lighting is
  controlled. The domain is extremely narrow. At ~5 dice per frame that is
  1 000–2 000 instances.
- Deliberately over-sample the hard cases: touching pairs, tight piles, dice
  against walls, cocked dice, glossy and pale dice, glitter dice.
- Include **every die in the collection**, not just the test set. Colour and
  finish variation is what the model needs to generalise over.
- Vary lighting a little within what the final rig will actually do. Do not vary
  camera pose.

### Labelling

Run SAM2 over each frame, apply the Phase 0 filter, and present each mask for a
human to assign a class. Labelling becomes a 7-way dropdown, not polygon drawing.

Budget: at ~1 500 instances and 2 s each, roughly one evening. Build a minimal
review UI — keyboard `1`–`7` for class, `x` to reject a bad mask, `space` to skip
a frame. The bad-mask reject is important: SAM2 will produce some wrong masks and
they must not enter the training set.

### Export

YOLO segmentation format — one `.txt` per image, one line per instance:

```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>      # normalised polygon, 0-1
```

Split 70/20/10 train/val/test. **Split by frame, never by instance** — two dice
from the same frame in different splits leaks.

`dice.yaml`:
```yaml
path: ../datasets/dice
train: images/train
val: images/val
test: images/test
names:
  0: d4
  1: d6
  2: d8
  3: d10
  4: d100
  5: d12
  6: d20
```

---

## Phase 3 — train YOLO11-seg

```bash
yolo segment train \
  data=dice.yaml \
  model=yolo11m-seg.pt \
  epochs=150 \
  imgsz=1024 \
  batch=8 \
  patience=30
```

Start at `m`. Drop to `s` if latency matters, go to `l` only if `m` plateaus
below target.

### Augmentation — the important part

The camera pose is **fixed**, and that changes what is safe:

| setting | value | why |
|---|---|---|
| `degrees` | **0** | Rotating the frame breaks the fixed foreshortening direction. Dice already land at every rotation naturally. |
| `perspective` | **0** | Same reason. The perspective is a real, constant property of the rig. |
| `scale` | 0.1 | Small only. Absolute die size is a genuine type cue here. |
| `translate` | 0.1 | Dice do land anywhere in the tray. |
| `fliplr` / `flipud` | **0** | A flip mirrors numerals and inverts the near/far gradient. |
| `hsv_v`, `hsv_s` | 0.4 / 0.5 | Lighting variation is the real-world nuisance. Push these. |
| `mosaic` | 0.0 | Composites four images; destroys the fixed geometry the model should exploit. |

**Resist the urge to turn on the defaults.** Ultralytics' defaults assume a
free-moving camera. Yours does not move, and that constraint is free accuracy if
you do not augment it away.

### Success criteria

| metric | target |
|---|---|
| mask mAP50 | > 0.90 |
| **d12 vs d20 confusion** | **< 5%** |
| Touching-pair separation | > 90% |
| Inference on 4090 | < 100 ms |

The d12/d20 row is the one to watch. It is the pair that defeated circularity,
the VLM, and direct side-counting. Report a full confusion matrix, not just mAP.

If d12/d20 is the only weak class, that is a data problem — collect more of both,
in more orientations, before touching the model.

---

## Phase 4 — value reading

Type comes from YOLO. Value is a separate problem and stays that way.

Do **not** fold value into the YOLO classes. 7 types × up to 20 faces ≈ 70
classes, which `docs/dice-reading.md` already identifies as data-hungry for no
benefit.

Options, in order of effort:

1. **VLM on the clean crop** — works today. Keep it.
2. **Per-type classifier** — a small CNN per die type, ≤20 classes each, trained
   on crops the YOLO model produces. Cheap once the pipeline exists.

Constraints to preserve from existing findings:

- **Never trust self-reported model confidence.** Measured: `"high"` confidence
  with `"clear view"` on a die whose face is not determinable from the image.
- **Derive confidence from cross-prompt disagreement.** Values that move between
  prompt phrasings are the ones that are wrong.
- **Do not supply set-composition priors.** Measured: forcing "one of each type"
  produced structurally perfect, semantically wrong output — it invented a d100
  to satisfy the constraint.
- **Opposite d20 faces sum to 21.** Keep as a hard validity check: any reading
  claiming two visible faces summing to 21 is impossible.

---

## Evaluation harness — build this before Phase 2

A held-out set of frames with hand-verified ground truth: per-die type, value,
and position. **20–30 frames is enough**, but they must include the hard cases.

Every approach — SAM2, YOLO, VLM — is measured on the identical frames, or the
comparison measures the pipeline instead of the approach. `crop_pipeline.py`
already exists for exactly this reason; keep that discipline.

Report:

- Instance detection: precision, recall, and **count error distribution**
- Type: confusion matrix, 7×7
- Value: accuracy, split by whether the die was touching another
- End-to-end: fraction of rolls where every die is correct

The last one is the number that matters at a table. Per-die accuracy of 95%
means a 7-die roll is fully correct only 70% of the time.

---

## What to delete

Once Phase 1 passes, this becomes dead code. Delete it rather than leaving it —
it represents a method that has been measured and rejected, and keeping it
invites re-litigation.

In `pi-tower/dice_detect.py`:

- `watershed_split()` and its guards
- distance-transform peak counting as a **control signal** (keep it, if at all,
  only as a cheap independent count for cross-checking)
- solidity-based clump flagging — measured useless: a single d6 scored 0.815
  while a merged d12+d10 scored 0.849
- circularity-based type identification — measured to **invert** for d12 vs d20
- the area-vs-median size heuristic
- top-face polygon detection by brightness — the numerals are brighter than the
  facets, so it traces digits

**Keep:** tray quad calibration and `fit_quad_to_frame()`, `quad_scale()`, the
per-frame channel selection in `to_gray()` (still useful for preview and
diagnostics), `crop_pipeline.py`'s crop conventions and provenance stamping,
`REF_TRAY_W` and the resolution-scaling discipline around it.

The two silent resolution bugs already documented — quad applied at the wrong
resolution, and kernel sizes not scaling — are exactly the class of bug that will
recur when frames start moving between the Pi and the training pipeline. Keep
the guards.

---

## Open questions

1. **Which SAM2 size?** `b` vs `l` — measure in Phase 0, do not assume.
2. **Does the Module 3 change the answer?** More pixels and no IR contamination
   should improve masks. Re-run Phase 0 after the swap and compare; that is a
   cheap, informative A/B.
3. **Full-sensor vs binned capture.** `docs/dice-reading.md` notes full-sensor
   stills require stopping the service because it holds a 1640×1232 video config,
   and that mode switching is "slow and hang-prone on a Pi 3." Decide: a
   dedicated stills path, or a permanently higher-resolution main stream. Measure
   whether 3280×2464 actually improves mask quality before paying for it.
4. **Genuinely occluded dice.** Some piles cannot be read from one viewpoint by
   any model — the up-face is physically hidden. This is an information limit,
   not a software problem. The answer is a UX that asks for a nudge, or a second
   camera. Decide which, and make the system say so rather than guess.
5. **Where does YOLO inference run long-term?** The 4090 is fine now. If the
   table must work with the workstation off, that changes the model size budget.
