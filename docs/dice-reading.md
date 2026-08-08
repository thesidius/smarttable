# Dice reading pipeline

> **The classical path described in Stages 1–1.5 was deleted on 2026-08-07.**
> SAM2 replaced it; see *The camera count is retired* below for what went and
> why. Those sections are kept as the record of what was tried and measured —
> the local-variance finding is still true, it just was not enough — but
> `dice_detect.py` no longer exists and none of it runs.

## Stage 1 — detection (superseded, kept as history)

`pi-tower/dice_detect.py`. Turns a tray frame into per-die crops plus a JSON
manifest.

**Segment on local variance, not brightness.** The tray floor measures mean
95.9 / std 1.1 — essentially featureless. A die is a near-black body carrying
bright numerals, so any window covering one contains both extremes at once.
Local standard deviation separates them by more than an order of magnitude
(floor ~1, die ~40+), while absolute brightness is close to useless: die body
and tray floor are similar shades and both drift with lighting.

Useful consequence: this is indifferent to the magenta cast, to the exposure
lock, and to whether the scene is lit by IR or visible light. It keys on local
structure, not on level or colour.

Result on the first real frame — 7 dice present, found as 5 blobs:

| # | verdict | truth |
|---|---|---|
| 0 | clean | d10 showing 10 |
| 1 | suspect — low solidity, oversized | d20 + d4 touching |
| 2 | suspect — fragment | sliver of the small d4 |
| 3 | suspect — low solidity, aspect 1.78 | d8 + d4 touching |
| 4 | clean | d6 showing 5 |

Every flag was correct and there were no detections outside the tray. Touching
dice merge into one variance blob; the script flags them via solidity and size
rather than silently emitting a bad crop. Splitting them needs a watershed
pass — worth adding once we know how often it happens in real rolls rather
than in a hand-placed pile.

## Stage 1.5 — canonical crop pipeline (working)

`pi-tower/crop_pipeline.py`. All three stage-2 approaches consume its output,
so the crop is defined once here and everything downstream reads the manifest.
Otherwise the comparison measures the crop pipeline instead of the approaches.

Crop definition, and why each choice: **square** (dice are square-ish in bbox;
no approach should ever see an aspect-distorted die), **fixed context ratio**
1.5× the larger bbox side (normalises scale, so a classifier cannot cheat by
learning "big blob = d20" — which breaks the moment camera height changes),
**lossless PNG** (JPEG ringing around numerals is exactly the signal we care
about), **no rotation normalisation** (orientation in the tray is genuinely
random; straightening here hides the problem until deployment), and
**BORDER_REPLICATE** padding for dice against a wall (a black border is a
strong artificial edge a classifier will latch onto).

`CROP_VERSION` is stamped into every record. Bump it when crop geometry
changes — that invalidates every label collected against the old version.

### Two resolution bugs, both silent

Worth recording because both produced *plausible* output rather than errors:

1. **Quad is resolution-specific.** A 1640×1232 calibration applied to a
   3280×2464 frame masks a quarter of the tray. No error, just quietly wrong
   crops. `fit_quad_to_frame()` now rescales explicitly, and *refuses* when the
   aspect ratio differs — on IMX219, 1640×1232 and 3280×2464 are the same FOV
   at different binning, but 1920×1080 is a sensor crop with a narrower FOV, so
   scaling into it would point the mask at the wrong part of the scene.
2. **Kernel sizes are resolution-specific.** The variance window and morphology
   kernels are physical sizes ("about a fifth of a die") expressed in pixels.
   Hard-coded, they halve in relative terms at 2× resolution, so dice fragment
   instead of closing into solid blobs — every detection came back flagged.
   `quad_scale()` now scales them with the tray.

### Measured behaviour

On three test frames (7 dice each, hand-placed in a pile): 6 blobs per frame,
2 clean and 4 suspect, consistent across all three.

- **Precision on "clean" is 6/6** — every unflagged crop is a well-centred
  single die.
- **Recall is poor** — 5 of 7 dice were touching and merged into clumps. Real
  rolls scatter more, but this is the limitation to beat.
- False positives on tray-wall seams *are* being caught by the aspect check, so
  they land in "suspect" rather than polluting the clean set.

### Watershed split — built, guarded, and currently declining to fire

`watershed_split()` + `detect_all()` in `dice_detect.py`. Distance transform →
local maxima as seeds → watershed on the *inverted* distance transform, so
basins flood outward from each die centre and meet along the waist between two
touching dice. Seed separation is keyed to expected die size rather than a
fraction of the global maximum, because a global fraction loses the small d4
entirely as soon as dice differ in size.

Two implementation traps worth remembering:

- **Read the watershed LABELS, never a rebuilt binary.** Watershed separates
  basins with a 1-pixel line, and 8-connected component analysis leaks
  diagonally across a 1-pixel gap — the regions silently re-merge and the split
  appears to do nothing.
- **"More blobs" is not the success metric.** Watershed over-segments happily,
  carving slivers with solidity 0.4 out of a ragged blob. The guard now
  measures *clean single dice* before and after and keeps the unsplit version
  unless that number improves, so the split can never make output worse.

**On current data the guard rejects every split** (clean dice 2 → 0 and 2 → 1).
That is the correct behaviour, not a bug, and it points at the real blocker:
the variance silhouettes are too ragged for a distance transform to find one
maximum per die.

Why they are ragged — measured, correcting two earlier guesses of mine:

| region | grey level |
|---|---|
| tray floor | 83–95 |
| die faces | 100–160 |
| tray floor MAD across whole tray | **9.0** (vs std 1.1 in a small patch) |

So brightness alone cannot segment dice (the ranges overlap), *and* there is a
brightness gradient across the floor that defeats any global threshold. Getting
solid die silhouettes needs local background estimation — but that should be
tuned against mounted-rig frames, not against one handheld shot of a hand-made
pile. Tuning it here is fitting to a scene that will not exist.

### A scaling constant that must stay a measurement

`REF_TRAY_W` is the tray width in px of the frame the kernels were tuned
against (805 px). It was briefly a guessed 750, which inflated every kernel by
~7% — enough for the morphological close to bridge adjacent dice and merge
detections that had been separate. If kernels are ever retuned, update this to
the tray width of the frame they were tuned on.

## Mixed die types — the size heuristic rewrite (2026-08-06)

The original `flag()` compared each blob's area to the median and called
outliers clumps or fragments. That is wrong for a polyhedral set, and measuring
it showed why:

| blob | area px | inradius |
|---|---|---|
| d6 | 7 833 | 32.0 |
| d8 | 10 376 | 40.0 |
| d4 | 14 729 | 48.3 |
| d10 | 16 350 | 53.2 |
| d20 | 22 902 | 69.0 |
| **d12+d10 merged** | **44 075** | **71.4** |

Real single dice span **2.9× in area**. Any area-vs-median rule either flags the
legitimate d4 and d20 or misses real clumps — and the median is itself computed
over a population containing clumps, which inflates it and makes small real dice
look like fragments.

**Solidity is no better, which was a surprise.** Measured: a single d6 scored
**0.815** while the merged d12+d10 scored **0.849**. Two dice side by side form
a perfectly convex-looking blob. Convexity does not distinguish them, so it was
dropped as a criterion (still reported for diagnostics).

### What replaced it

Count die-sized lobes via the distance transform, and size against the
**inradius** rather than area — the largest inscribed circle does not double
when two dice merge, whereas area does.

Two tests, both required, because each alone gives false positives:

- **Peak count alone** flags elongated single dice. A d6 photographed at an
  angle shows a top face and a side face joined at a narrow waist and reads as
  two maxima — while being the *smallest* blob in frame, so it cannot hold two
  dice.
- **Area alone** cannot work, per the 2.9× spread above.

Together: a clump has multiple lobes AND is substantially larger than a lone
die. Result on the same frame — **5 of 6 blobs clean, no false flags on the d4
or d20**, all five single dice read correctly by gemma3.

### Touching dice: flagged correctly, still not split

The user-visible complaint — a d12 and d10 in one box — is now **correctly
flagged** ("contains 3 dice"), but not separated.

The watershed keeps producing a sliver rather than two dice, because these two
dice **overlap substantially in projection** rather than merely touching. Their
union is a convex blob with no waist for the distance transform to cut at.
Separating genuinely overlapping dice from a single 2D view is a much harder
problem than separating adjacent ones.

Two guards now stop that from being papered over:

1. **Split per clump, never over the whole binary.** The earlier version
   re-carved good single dice too, turning whole-die boxes back into
   numeral-sized fragments — and it was *accepted*, because the guard
   recomputes its size reference on the split output, so when everything shrinks
   uniformly each piece still looks proportionate.
2. **Reject a split wholesale if any piece is not die-sized.** Otherwise the
   watershed "succeeds" by shaving a sliver off a clump: two components, clean
   count up, guard satisfied — while the two dice are still merged and now
   carry one peak each, so nothing flags them. That launders a known problem
   into a clean-looking result, which is worse than not splitting at all.

Also fixed here: `blob_shape` returned an inradius of **65533.8** (2^16-1) when
a crop window contained no background pixel — `distanceTransform` measures
distance to the nearest zero, and with no zero the result is unbounded. It then
poisoned the median every other test depends on. A zero border fixes it.

Options for the overlap case, none yet chosen: a tray floor texture or slope
that discourages dice settling on each other; accepting that overlapping dice
stay flagged and get nudged or re-rolled; or a second camera angle.

## Tight groups (2026-08-06) — detected and counted, not split

All seven dice pushed into one pile. This is a normal roll outcome, and the
tray already has 45° inside edges, so there is no physical fix available.

**Critical bug fixed first: a tight group produced ZERO detections.** The
merged blob exceeded `max_frac` (0.06 of tray area) and was silently discarded
— total silence, the worst possible response. A 7-dice pile is legitimately
~20% of the tray. Oversized blobs are now kept, flagged, and passed to the
splitter; `max_frac` default raised 0.06 → 0.45.

**What works now:** the pile is detected as one blob and correctly reported as
*"contains N dice — touching, needs splitting"*. The distance-transform peak
count lands in the right region (5–7 for 7 dice) but is **not stable** — the
same physical pile reported 7, then 6, then 5 across consecutive frames as
segmentation noise moved the maxima. Useful as a sanity check, not as a control
signal.

**What does not work: splitting it.** Measured, on the same pile:

| topography | pieces | coverage of clump |
|---|---|---|
| distance transform | 7–9 | 48–73% |
| **real image gradient** | **2** | **8%** |

Flooding over real pixels was tried and is far worse — the background marker
wins territory through weak floor-to-die gradients while strong internal edges
from numerals stall the flood inside each die. Reverted to the distance
transform.

Even at its best the distance-transform split loses roughly half the pile and
yields pieces of 5 241–32 645 px against real single dice of 7 833–22 902. The
guards (no sliver relative to siblings, ≥50% coverage retained) correctly
refuse it rather than emitting crops that would read wrong.

Separating a dense pile of *overlapping* polyhedral dice from one top-down view
is the genuinely hard case. Classical CV is the wrong tool; this is normally
solved with trained instance segmentation or a second viewpoint.

### The architecture already has a better answer

Reading is remote (see `system-architecture.md`), so the crop does not have to
contain exactly one die. Feeding the **clump region as a single image** to
`gemma3:27b`:

| input | reply | time |
|---|---|---|
| whole tray | `[4, 5, 2, 8, 20, 2]` | 1.8 s |
| clump crop | `[4, 14, 5, 2, 8, 2]` | 2.4 s |

Six of seven dice, no segmentation at all. Not perfect — it missed one and read
a d20's side face as 14 where the up-face is 20 — but far better than any split
produced, and it costs one request.

**Proposed design, not yet built:** split what splits cleanly, and for anything
still flagged as a clump send the region whole and ask for a list, using the
peak count as an expected-count sanity check. That plays to the remote reader
instead of fighting an unsolved segmentation problem on a Pi 3.

### Can it read all 7? No — and it will not tell you which one it missed

Measured on the same pile, `gemma3:27b` at temperature 0, four trials per
prompt (every trial within a prompt was byte-identical):

| prompt | result |
|---|---|
| no count hint | `[4, 14, 5, 2, 8, 2]` — only 6 values |
| **"exactly 7 dice"** | `[4, 20, 5, 2, 8, 6, 2]` — 7 values |
| ask for per-die confidence | `[4, 20, 5, 2, 9, 8, 2]`, **all seven "high confidence"** |

**Telling it the count genuinely helps.** It went from 6 values to 7 *and*
corrected the d20 from 14 to 20 — being forced to account for every die made it
re-examine rather than settle for the first plausible face. The hint is free:
the dice set is known.

**But self-reported confidence is worthless.** Asked to flag dice it could not
clearly see, it returned `"confidence": "high"` with `"Clear view of the '9'"`
for a die whose up-face is *not determinable from that image at all* — an
octahedron at an oblique angle, foreshortened and partly occluded by
neighbours. Independent inspection could not read it either.

Worse, its value for that die moved across prompt phrasings — **8, then 6, then
9** — while claiming a clear view each time. So the count hint trades a
*detectable* failure (a short list) for an *undetectable* one (a confabulated
value delivered with confidence).

**Determinism is not agreement.** Four identical trials means repetition catches
nothing: it is consistently wrong when wrong. Voting requires genuinely
different inputs, not repeated ones.

### What actually signals uncertainty

Cross-prompt disagreement does. The four unambiguous dice (4, 20, 5, 2) were
stable under every formulation; only the occluded ones moved. So:

- **Use the count hint** — it measurably improves accuracy.
- **Never trust the model's own confidence field.**
- **Derive confidence from agreement** across 2–3 genuinely different prompts
  or crops, and from the detector's independent peak count.
- **Surface disagreements for a nudge or re-roll** rather than guessing.

### Priors: the count helps, the set composition hurts

Constraint from the user (2026-08-06): **any number of dice, any types.** No
fixed set. That rules out both priors tested — which is the right call, because
one of them was actively harmful.

Same pile, same model, temperature 0, all deterministic across 3–4 trials:

| prompt | d20 reads | count |
|---|---|---|
| "exactly 7 dice" | **20** ✓ | 7 |
| ask for per-die confidence | **20** ✓ | 7 |
| "one of each type" set prior | 14 ✗ | 7 |
| **no priors** (the shipping config) | 14 ✗ | 7 ✓ |

Ground truth verified by zooming the die: the up-face is unambiguously **20** —
large, upright, face-on, while 14/8/18/12/2/10 are foreshortened side facets.

**The set-composition prior produced structurally perfect, semantically wrong
output.** One of each type, no duplicates, every value in range for its type —
every structural check passed — and it called the d20 a 14 while assigning the
20 to a "d100" that is not in the pile. Forcing a type allocation made it invent
a die to satisfy the constraint. Structural validation cannot catch this.

Good news for the unpriored config: it gets the **count** right (7) with no hint
at all, deterministically, in ~3.3 s.

### Cross-prompt disagreement is the usable confidence signal

The d20 is exactly the die that flips between formulations. Self-reported
confidence said "high" every time; prompt-to-prompt disagreement tracked the
actual errors. It needs no priors, so it survives the any-number-any-type
constraint, and it is fully automatic — no player interaction, just 2–3 calls at
~3 s each.

Pair it with the Pi's own blob/peak count as an independent estimate of how many
dice are present. Under this constraint that count is **more** valuable, not
less: with no known dice count, it is the only signal that can catch the model
silently omitting a die.

## Phase 0 RESULT — SAM2 passes the gate (2026-08-06)

`pi-tower/sam2_phase0.py`, SAM2.1-base, automatic mask generation, RTX 4090.
Run per `docs/instance-segmentation-plan.md`.

| frame | truth | raw masks | after filter | time | separation |
|---|---|---|---|---|---|
| 8 dice, some touching | 8 | 18 | 9 | **2.4 s** | **8/8, no merges** |
| 8 scattered (4 d12, 4 d20) | 8 | 15 | 10 | **1.4 s** | **8/8** |
| **7 in a tight pile** | 7 | 11 | 8 | **1.4 s** | **7/7 separated** |

Every gate criterion met:

| criterion | target | measured |
|---|---|---|
| scattered dice | ≥7/8, no merges | 8/8 |
| **tight pile** | **≥5/7 separated** | **7/7** |
| false positives | ≤1/frame | +1, +2, +1 |
| latency | <5 s | 1.4–2.4 s |

**The pile row is the one that matters, and it is not close.** Every die in a
seven-die pile received its own mask. The watershed on the same class of frame
lost 27–52% of the pile and was correctly rejected by its own guard every time.
This is the case that is impossible for distance-transform methods and routine
for a learned model.

The residual +1/+2 are almost certainly tray floor or shadow surviving the area
band; the plan's quad filter (skipped here because these are already tray crops)
should remove them on full frames.

### Head-to-head against the VLM bounding-box result

| approach | separation | time | gives type? | needs |
|---|---|---|---|---|
| distance-transform peaks | fails; reported 20 dice for 8 | ~0 | no | — |
| VLM boxes (Claude Code, opus) | 8/8 incl. 4-die cluster | **263 s** | **yes** | API round-trip |
| **SAM2.1-base** | **8/8 and 7/7 pile** | **1.4–2.4 s** | **no** | local GPU |

SAM2 is ~110x faster and returns *masks* rather than boxes, which matters
because the plan crops from the mask — that is what removes the neighbouring
dice that caused the misreads. It is class-agnostic, so it does not answer "d12
or d20"; the VLM does. They are complementary, which is exactly the Phase 1
split: SAM2 separates, the VLM reads type and value from a clean crop.

### Toolchain note

The plan assumed a working GPU stack. There wasn't one: the default Python here
is **3.14, for which no CUDA torch wheel exists** — the installed torch was
CPU-only, and Ollama reaches the 4090 through its own bundled runtime rather
than this environment. Fixed with a separate Python **3.13** venv
(`.venv-ml/`, gitignored) carrying `torch 2.13.0+cu126` and
`ultralytics 8.4.116`.

One API correction: `points_stride` is a parameter of `Predictor.generate()`,
**not** a call kwarg. Passing it to `model(...)` fails validation with "not a
valid YOLO argument". The default is already 32, so it only needs the predictor
route when overriding.

## Phase 1 BUILT — SAM2 separates, the VLM reads isolated dice

```
Pi /roll -> tray image -> seg_service /segment -> N isolated die crops
                                                      |
                          per-crop VLM read (2 prompts) -> type + value + confidence
```

- `webapp/seg_service.py` — SAM2 on the 4090, port 8090, **its own interpreter**
  (`.venv-ml`, Python 3.13). The control panel is on 3.14, which has no CUDA
  torch wheel. A service rather than a subprocess so the model loads once.
- `webapp/app.py` `POST /api/roll2` — the Phase 1 pipeline.
- `read_segmented()` in `roll_reader.py` — per-crop reading.

**Crops are mask-composited onto neutral grey, not bounding-box cut-outs.** That
is the whole point: a box around a die in a pile still contains its neighbours,
and neighbours are what caused the misreads. Grey rather than black or white
because a hard black border is an artificial edge the reader latches onto, and
white blows out against pale dice; mid-grey sits between the darkest (~20) and
palest (~180+) dice measured here.

### Measured, live

| stage | result |
|---|---|
| segmentation | **8/8 dice, 0 false positives**, 17 raw masks filtered to 8, **3.09 s** |
| reading (haiku, 2 prompts each) | works; 6/10 low-confidence on an earlier run |
| **total** | **295 s** — segmentation is 1%, reading is the rest |

### The filters that matter

The first live run returned three false positives: a thin tray-edge sliver, a
**LEGO brick sitting outside the tray**, and an empty corner. All three touched
the frame border — and no die can, because the tray walls are inside the crop.
Adding border rejection plus a shape test (aspect 0.45–2.2, mask/bbox fill
≥0.45; a die fills ~0.7–0.8 of its box, a sliver ~0.1) took it to zero.

Note this is *because* the tray quad covers the walls rather than the floor —
the right call, since a die can rest against a wall, but it means the crop
contains wall and whatever is beyond it.

### Batched reads — measured, and it works

One call per prompt variant covering every die, instead of one call per die.
Isolated properly (same model, same rig, per-die vs batched):

| config | dice | total | segmentation | high-confidence |
|---|---|---|---|---|
| haiku, per-die | 10 | 295 s | ~3 s | 4/10 |
| **haiku, batched** | 8 | **76 s** | 8.3 s | 4/8 |
| opus, batched | 8 | **1035 s** | 2.6 s | **7/8** |
| ollama gemma3, batched | 8 | 442 s | 2.7 s | 5/8 |

**Batching is ~3x faster per die** (haiku 295 s → 76 s). Most of that was
`claude -p` subprocess and agent startup paid 16 times instead of twice.

**No configuration is yet both fast and accurate.** Opus gets the d12/d20 split
right — 4 and 4, 7/8 dice agreeing across prompts — and takes 17 minutes.
Haiku is 13x faster and only half its dice agree. Ollama manages to be *both*
slower than haiku and less accurate, calling 6 of 8 dice d20; local and free is
its only remaining argument.

An earlier note here compared 295 s against 1035 s as if it showed batching
made things worse. It did not — two variables changed at once (per-die→batched
AND haiku→opus). The haiku-batched row is what isolates it.

### The Anthropic API path is the untested cell

`claude -p` is an agent: it starts up, plans, and reads files as tool calls.
The Messages API sends the same images to the same model with none of that.
Opus's accuracy at a fraction of 1035 s is the plausible win, and it is the one
configuration in the matrix nobody has measured — it needs an `sk-ant-api` key,
which is not the OAuth token currently configured.

### The camera count is retired — and with it the whole classical path (2026-08-07)

Across these runs the Pi's distance-transform count reported **20**, **27**,
**36** and finally **73** dice for 8. A cross-check wrong by 9x does not catch
the reader omitting a die; it raises a mismatch on every single roll until the
warning means nothing. Removed from `/api/roll2`, from `reconcile()`, and from
both result tables.

Removing the count removed the only caller of the classical detector, so that
went too:

| removed | was |
|---|---|
| `pi-tower/dice_detect.py` | 42 KB: local-variance segmentation, watershed split, distance-transform counting, shape heuristics |
| `POST /analyze` on the Pi | the endpoint the Detect tab's sliders drove |
| `_resolve_frame()` | re-run detection on a stored capture |
| `pi-tower/crop_pipeline.py` | training-data CLI whose detection half was that module |
| Detect-tab controls | variance window, min/max blob area, crop context, watershed toggle |

What survived is `pi-tower/tray_geometry.py` — just `fit_quad_to_frame()`,
which was never about detection. It converts a tray calibration measured at one
resolution into another, and refuses when the aspect ratio differs. `/roll` and
`/framing` both still need it, and it is still the guard that turns a silent
quarter-of-the-tray mask into an error.

Three things the plan listed under **Keep** were not kept, deliberately:

- **`quad_scale()` and `REF_TRAY_W`.** They existed to scale morphology kernel
  sizes with resolution. There is no morphology left, so there is no kernel to
  scale — and a scaling helper with no consumer is precisely the dead code the
  plan warns about. The *discipline* the plan actually wanted is in
  `fit_quad_to_frame()`, which refuses rather than silently rescaling across a
  changed field of view. That is kept.
- **`to_gray()`'s per-frame channel selection.** Genuinely a good finding —
  under coloured light, OpenCV's fixed BGR2GRAY weights can hand 59% of the
  signal to a dead channel — but nothing calls it now. The finding is recorded
  here; the code was not worth keeping alive for a hypothetical caller.
- **`crop_pipeline.py`'s crop conventions.** Kept, but *moved* rather than
  retained: they are the docstring of `seg_service._crop()`, which is where
  cropping actually happens now. One convention was deliberately dropped —
  see below.

`crop_pipeline` normalised every crop to a fixed side so a classifier could not
cheat by learning "big blob = d20". `_crop()` does not, because these crops go
to a vision model and downsampling a 625 px die to a common size throws away
exactly the numeral resolution that decides the read. The PNGs are stored at
native size; normalise at training time if a classifier ever wants them.

**Provenance stamping was kept, and immediately earned its place.**
`CROP_VERSION` is recorded with every label. It went from 1 to 2 on the same
day the labelling flow was built, because the stray-pixel fix below re-framed
every crop — labels collected either side of that change are against different
images, and without the stamp there would be no way to tell which were which.

**The Pi's `/roll` went from 67 s to 1.6 s.** That classical pass was the entire
reason a capture was not near-instant; it was burning 67 s of Pi 3 CPU to
produce a number that was wrong by 9x. Capture-and-segment end to end is now
**6.5 s**, down from ~47 s.

The Detect tab was repurposed as **Segment**: capture and segment with no
reading, showing the tinted mask overlay and the isolated crops. Segmentation
costs seconds where reading costs minutes, and when a roll comes back wrong,
"a mask merged two dice" and "the reader misread a clean crop" produce the
identical symptom. Being able to rule out the first for 6 s is the point.

### Two defects the overlay and the crops exposed

**Stray-pixel masks.** SAM2 masks routinely carry a few pixels tens of percent
of the frame away from the die. Far too small to trip the area filter — but
min/max over the raw coordinates is decided by the single most distant pixel,
so one speck off-centres and inflates the crop. Measured: a d20 was pushed into
the right-hand third of its own 1045 px crop, throwing away most of the
resolution the reader gets. Keeping only the largest connected component fixed
it: same die, 625 px, filling the frame. It also cleaned up the shape filter,
which computes fill and aspect from the same bounding box.

**The overlay was unreadable.** Tinting each mask a random colour at 45% over
dice that already carry strong colour produced a wash — two adjacent dice got
near-identical tints, which is exactly the case the overlay exists to catch.
Now: background dimmed to 38% so an unsegmented die is visibly dark, a fixed
well-separated palette, and a hard outline per mask. Two dice inside one
outline reads instantly.

**Digit scraping in `/api/suggest`.** `"".join(c for c in raw if c.isdigit())`
over `{"type":"d20","value":4}` yields **204**. That pre-filled the Label tab's
input with a wrong value for a human to rubber-stamp into the training set —
the worst possible place for a silent error. Now parsed with `_parse()`, which
already existed for the roll path, and the die type is surfaced alongside.

### Labels now point at images that exist

SAM2 crops are composited in memory on the workstation and never written to the
Pi, so a label naming a Pi filename would point at nothing. `/api/label` takes
the inline image and writes it under `dataset/crops/`, keyed by capture id and
mask index. A label log whose crops cannot be reopened is useless as training
data, which is the only reason to be labelling.

### The bottleneck moved

Segmentation is no longer the problem; it is 1% of the time. The cost is now
**N dice × 2 prompts sequential VLM calls** — 16 subprocess launches for 8 dice.

The obvious fix is batching: `claude -p` can be handed several image files in
one call, turning 16 launches into one. Untested, but it is the difference
between ~295 s and something usable at a table.

## Die-type identification by SHAPE, not numbers

The useful invariant is topological, not numeric: **the top face is a polygon
whose side count is fixed per die type, and exactly that many faces border it.**
This is independent of the numbers, the colour, and the size — which is what
makes it worth building rules on.

| type | top face | faces bordering it | silhouette from above | ideal circularity |
|---|---|---|---|---|
| d4 (standard) | none — vertex up | 3 (all of them) | triangle | 0.605 |
| d6 | square | 4 | square | 0.785 |
| d8 | triangle | 3 | square / rhombus | 0.785 |
| d10 | kite (asymmetric quad) | 4 | hexagon with a point | ~0.88 |
| d12 | pentagon | 5 | decagon, nearly round | 0.970 |
| **d20** | **triangle** | **3** | **hexagon** | **0.907** |

d8 and d20 share a triangular top with 3 neighbours, so the **silhouette breaks
the tie**: square for a d8, hexagon for a d20. The d10's top is a kite rather
than a square, which separates it from the d6.

### Verified so far: d20 only

Four d20s, deliberately different colours, sizes and up-faces (7, 13, 10, 20):

- **Triangular top face bordered by exactly 3 faces — 4/4.** The user's claim
  holds across the set.
- **Hexagonal silhouette — 2/4 clean** (6 vertices, circularity 0.850 and
  0.843). The other two were damaged by segmentation, not by shape.

### Segmentation is the limiting factor, again

The two failures are instructive and both are surface-property problems:

- the **glossy pink** die's specular highlights punched holes in its mask
- the **pale white** die is too close to the tray floor in level, losing a notch
  from its outline

Circularity fell to 0.597 for the pale one — triangle-like, nothing to do with
its real shape. So shape-based typing is only as good as the mask, and the two
hardest surfaces are exactly gloss and pale.

### Top-face polygon detection: brightness does not work

Tried isolating the top facet as "the brightest large region near the centre",
then counting its sides. It fails, and instructively: **the painted numerals are
far brighter than any facet**, so the detector traces digits rather than the
face. Facet shading is real but the numerals overprint it.

Detecting the facet *edges* (the creases) rather than the regions is the
remaining option — Hough lines within the die mask — but that still needs a
clean mask first, which is the actual blocker.

### The real blocker is the tray floor, not the algorithm

Scattered d20s, exposure locked, four dice:

| die | circularity | verts | shape verdict |
|---|---|---|---|
| glossy pink | **0.822** | **6** | **hexagon → d20 ✓** |
| pale white | 0.265 | 9 | mask has two notches cut out |
| blue + black | 0.402 | 8 | still touching, one blob |

Locking exposure did not change this — it is not an AE-drift problem.

**Why it keeps happening:** the tray floor is mid-grey, measured 86–167. Dice
run from near-black (~20–40) to pale white (~180+). The floor tone sits *in the
middle of the die range*, so there is no threshold, and no single channel, that
separates both a black die and a white die from it. Dark dice were the problem
under purple light; pale dice are the problem now. Gloss adds specular holes on
top.

**A floor colour outside the dice gamut would make this nearly free.** A
saturated matte colour — the green-screen principle — separates by hue instead
of level, and hue does not care whether a die is black, white or glossy. That
one physical change would fix the pale-die notches, the specular holes, and the
channel-selection fragility in one move, and it would make the shape rules
usable rather than occasional.

Worth checking the dice gamut first: the current set includes blue-speckled and
pink dice, so green is the safer choice.

### Who fails at what — measured, 2026-08-06

The lamp was moved lower and to the side. It **halved the glare and changed
nothing about the masks**, which is the cleanest evidence that these are two
independent problems with two different owners.

| failure | owner | fixed by |
|---|---|---|
| glare destroying the up-face | lighting geometry | lamp angle — **done, worked** |
| pale/glossy masks, merged dice | Pi segmentation | still open |
| d12 vs d20 confusion | Ollama | not lighting-related |
| 6 vs 9 | inherent to dice | dot convention, needs resolution |

**Value reading is not the weak link.** Given a crop centred on one die, Ollama
read correctly every time tested — 4/4 on an isolated d20 and correct on
peak-centred crops pulled straight out of a pile. Its only value failure was on
the die whose face was physically destroyed by specular blowout.

**Type identification is unstable in Ollama**, and moving the lamp made it
*worse* for one die:

| die | lamp overhead | lamp low + side |
|---|---|---|
| white | d20 ✓ | **d12** ✗ |
| blue | d20 ✓ | d20 ✓ |
| black | d20 ✓ | d20 ✓ |
| pink | **d12** ✗ | **d12** ✗ |

The confusion is always d12↔d20 — precisely the pair the silhouette test
separates cleanly (hexagon 0.907 vs decagon 0.970).

**They fail on different dice, which makes them complementary.** The Pi's shape
method correctly typed the pink die (circularity 0.826, 6 vertices → d20) — the
exact die Ollama calls a d12 — because an outline survives glare even when the
surface does not. Conversely Ollama typed the white die correctly while its mask
was too broken to measure.

**Suggested division: Pi determines TYPE from shape, Ollama determines VALUE
from pixels.** Each on its strength, cross-checking the other.

### The mask fix that does not need a new tray

Segmentation currently keys on *level*, which is why it fails at both ends: the
tray floor (86–167) sits between black dice (~20–40) and pale ones (~180+), so
no threshold has the right polarity for both.

**A die's outline is a real edge regardless of which side of the floor tone it
sits on.** Edge-based silhouette extraction — Canny plus contour closing, rather
than thresholding on level — does not care about polarity and should recover
both the pale die and the glossy one. That is the next thing to try for masks,
and it needs no change to the tray.

## d12 vs d20 — three approaches tried, none works yet (2026-08-06)

Eight dice, four d12 and four d20, mixed colours and sizes, well separated,
**ambient light only**.

### Positive finding: ambient light beats the lamp for segmentation

8 blobs for 8 dice, against 3 blobs for 4 dice under the directional lamp.
Diffuse light produces fewer specular holes, and image p95 actually *rose*
(114 → 131) despite lower mean brightness — fewer blown highlights, not more.

### 1. Silhouette circularity — fails

Theory says d12 (decagon, 0.970) is rounder than d20 (hexagon, 0.907).
Measured, restricted to the three masks that were visually solid:

| die | true type | circularity |
|---|---|---|
| black | d20 | 0.790 |
| blue | d20 | 0.825 |
| blue | **d12** | **0.678** |

The d12 scores *lower* than both d20s. The rule does not merely fail, it
inverts. Suspecting the detector's morphology was rounding silhouettes, the
same measurement was repeated on a minimally-processed mask — it got **worse**
(d20 0.790 → 0.413), because the raw variance mask is ragged.

The real reason: the ideal gap is only 0.907 vs 0.970, and mask noise is far
larger than that. Circularity cannot resolve it at this segmentation quality.

### 2. Ollama naming the die type — unstable

3/4 in one lighting condition, and moving the lamp flipped a correct d20 to
d12. Errors are always the d12↔d20 pair.

### 3. Ollama counting the top face's sides — worse

The most promising framing (3 sides vs 5 is a huge difference, unlike a 7%
roundness gap) scored **3/8** — including calling a d20's triangular top face
"8". Asking a more concrete visual question did not help.

### 4. Full-sensor resolution — helps the model, not the CV

Captured at 3280x2464 (2x linear, ~4x pixels per die) with the same locked
exposure.

**Circularity is unchanged** — 0.789 vs 0.790, 0.861 vs 0.859, etc. Expected:
circularity is scale-invariant and the masks are limited by contrast, not
pixels. Resolution does not rescue the silhouette approach.

**But type identification improved and became systematic:**

| | at 1640x1232 | at 3280x2464 |
|---|---|---|
| d20s correct | unstable, flipped with lighting | **4/4** |
| d12s correct | — | **0/4** — three called d10, one d20 |

So "is this a d20?" is now reliable, and the residual error is a consistent
d12→d10 confusion rather than noise. Both are roundish from above; the real
difference is face shape (d10 has *kite* faces, d12 has *pentagons*), which is
the same discrimination that scored 3/8 when asked directly.

Cost: full-sensor stills need the service stopped, since it holds the camera in
a 1640x1232 video configuration. Mode switching per request was deliberately
avoided as slow and hang-prone on a Pi 3, so this needs a decision — either a
dedicated stills path or a permanently higher-resolution main stream.

### Where that leaves type identification

**Unsolved.** Not by shape, not by the model. The information is plainly there
— a human sees pentagons on the d12s and triangles on the d20s instantly — but
neither route extracts it reliably at ~200 px per die.

Untried options, in rough order of promise:

- **More pixels.** Dice are ~200 px here; the sensor supports 3280x2464, which
  would roughly double that. Facet creases may simply be below the resolution
  needed. Cheap to test.
- **Hough lines on facet creases** inside a die crop, rather than region-based
  segmentation. The creases are straight and the top face is the polygon
  enclosing the die's centre.
- **Do not identify type at all.** Often the value implies it (a 17 can only be
  a d20), and more importantly the *game system already knows what it asked
  for* — "roll 1d20" does not need the table to work out that the die is a d20.
  Type identification may be solving a problem the application does not have.

### Not yet verified: every other die type

A mixed-set measurement was attempted against an older frame and **discarded as
invalid** — that frame predates the tray shift, so the current calibration quad
does not match it. It produced circularity 0.48–0.72 and two pure wall
artifacts. Deriving rules from it would have been deriving rules from a broken
mask.

The table above is sound solid geometry, but only the d20 row is measured. Each
remaining type needs a fresh capture in the current rig.

## d20 numbering — what holds, what does not (2026-08-06)

### Confirmed: opposite faces sum to 21

So the visible top hemisphere contains **exactly one of each complementary
pair**, never both. Verified against two independent photos of the same die
(20 up, and 17 up) — 14 checks, zero violations.

| | visible faces | complements, which must be hidden |
|---|---|---|
| 20 up | 2, 8, 10, 12, 14, 18, 20 | 1, 3, 7, 9, 11, 13, 19 |
| 17 up | 3, 7, 8, 10, 12, 15, 17 | 4, 6, 9, 11, 13, 14, 18 |

Neat cross-check: `{2,14,18,20}` appear only in the first photo and
`{3,7,15,17}` only in the second — and 3, 7 are precisely the complements of
18, 14. The die was re-rolled between shots, flipping those pairs.

**Usable as a hard validity test:** any reading claiming two visible faces that
sum to 21 is impossible.

### Disproved: the "step 6" pattern

Around the 20, the three edge-adjacent faces are 14, 8, 2 — every gap exactly 6,
which looks like a rule. It predicts 17's neighbours would be `{11, 5, 19}`.
The actual ring around 17 is `{10, 3, 7, 15, 12, 8}`. **One-sample coincidence**,
recorded here so nobody re-derives it.

### The finding that actually matters: clutter, not geometry

Both photos show a **bright specular streak straight across the up-face** — it
is the face aimed at both the lamp and the lens, so it is the worst-lit numeral
on the die, while the tilted ring around it is cleaner. That is very likely why
the reader kept grabbing 14 off the rim instead of the 20 in the middle.

Measured, same die, same frame:

| input | d20 reads | correct? |
|---|---|---|
| whole pile image | 14 | ✗ |
| **isolated die crop** | **20** | ✓ (4/4 trials) |
| **peak-centred crop** | **20** | ✓ |

**The error is caused by surrounding clutter, not by the die.** Isolate it and
the reading is right.

### Consequence: locate, do not segment

This dissolves the problem that blocked the tight-group case. Reading
individually does **not** require the watershed split to work — it only requires
a *centre point* per die, and the distance-transform peaks already provide that
even when the split fails.

Crop a die-sized box around each peak and read each box:

```
peak centres (5 found): (906,423) (785,471) (944,571) (780,679) (950,756)
readings:                20        5         2         2         2
```

The d20 is correct here where the whole-pile read was wrong.

**Remaining gap:** only 5 peaks were found for 7 dice, so two were missed. That
is a tractable tuning problem (peak threshold and separation) and a much easier
one than partitioning overlapping silhouettes. Whole-pile reading stays useful
as an independent cross-check on the count.

### The information limit

Some piles cannot be read from one top-down view, by any model or algorithm,
because the up-face is physically occluded. That is not solvable by better
software. The real options are a second camera angle, or a UX that asks the
player to nudge the pile when the system cannot see every die — which is also
what a human does when dice land stacked.

### These particular frames are not a usable dataset

Two of the three were captured before the tuning-file fix. The same die through
`imx219.json` vs `imx219_noir.json` differs by ~6× in R/G — visibly a different
image. Mixing them would poison any colour-sensitive model. The manifest now
records capture provenance (`ExposureTime`, `AnalogueGain`, `ColourGains`) per
crop so a mixed dataset is *detectable* rather than silently harmful. Filter on
`ColourGains[0] < 1.0` to keep only correctly-tuned captures.

## Stage 2 — reading the value (not started)

**The hard part is not OCR, it is knowing which face is up.** A d20 in the
test frame legibly shows 14, 20, 8, 2, 10 and 12 *simultaneously*. Any approach
that just finds numerals and reads them will return six answers for one die.
A d6 has one obvious top face; a d20's top face is one small triangle among
many equally-legible ones.

Two ways to handle it:

1. **Geometric** — find facet polygons inside the die crop, pick the one whose
   normal points at the camera (least foreshortened, most central), read only
   that. Explicit and debuggable; needs reliable facet segmentation on glossy,
   glittery dice, which is where it will fight us.
2. **Learned, whole-crop** — feed the entire die crop to a classifier and
   predict the value directly, letting it learn "which face is up" implicitly.
   Much less code and no fragile geometry, at the cost of needing labelled
   examples across orientations.

3. **Vision model** — ship crops to a general vision model. No training data
   and quick to stand up. **Tested 2026-08-05 and it works** — see below.

**Decision (2026-08-05): prototype all three and compare** on identical
held-out frames rather than committing up front. The comparison is the
deliverable. This makes the shared crop pipeline the critical piece — all
three must consume exactly the same crops or the comparison means nothing.

A likely outcome worth anticipating: (3) is the fastest way to generate labels
for (1) and (2), so it may earn its place as tooling regardless of whether it
wins on latency.

### Approach 3 tested — gemma3:27b reads dice correctly at ~1s each

`gemma3:27b` on the Ollama host (10.0.0.5) reports `capabilities: completion,
vision`, 131k context. Fed 128×128 crops straight from `crop_pipeline.py` with
the prompt *"…which number is on the face pointing UP toward the camera? Other
faces are visible at an angle — ignore those."*:

| crop | quality | answer | truth |
|---|---|---|---|
| `..._d00` | clean | **10** | 10 ✓ |
| `..._d05` | clean | **5** | 5 ✓ |
| `..._d01` | suspect (d20+d4 clump) | 14 | ambiguous — two dice in frame |
| `..._d02` | suspect (clump + wall) | 3 | ambiguous |

**Both clean crops correct.** The two suspect crops contain more than one die,
so there is no single right answer and they prove nothing either way — they are
listed to avoid overstating the result.

Latency: **19.5 s cold model load, then ~1.0–1.3 s per die warm** (42 tok/s
generation). Seven dice ≈ 7 s per roll serially, less if batched. That is
usable at a table, so approach 3 is a plausible *product*, not just a labelling
assistant — which was not the expectation going in.

Caveats before treating this as settled: sample size is two verifiable crops,
from a handheld frame rather than the final geometry, on one set of dice. It
needs systematic evaluation on mounted-rig data before it beats a trained
classifier on merit. But it is now the cheapest path to a working system *and*
the cheapest path to labels for the other two approaches.

### Sequencing constraint

**Do not collect training data until the mount is final.** Camera height, tilt
and lighting all change the appearance of a die crop. Data gathered handheld
will not transfer to the mounted rig, and re-labelling is the expensive part.
Mount first, lock exposure, then collect.

### Open questions

- Does the value classifier need to know the die *type* (d6 vs d8 vs d20)
  first, or can one model cover all of them? A "20" only exists on a d20, but
  a "3" exists on nearly everything.
- Where does inference run — Pi or G5? The Pi 3 is slow; the G5 is the stated
  inference box. That argues for the Pi capturing and cropping, and shipping
  crops over the network.
- How are ties/occlusions handled when a die lands cocked against a wall?

## Mounted rig, first run — 2026-08-06

Camera mounted, tray calibrated (59.1% frame coverage, margins 15.5/14.3/3.7/7.5%,
no edge clipping, corner brightness 73–109% of centre), exposure locked.
Detection found **all 7 dice as 7 separate blobs** — scattered dice do not merge,
so the watershed correctly declined again.

**And the result is still not usable, because of lighting.**

The tray is lit by purple/blue LEDs. Measured: tray floor at **grey level 13 of
255**, analogue gain pegged at **9.85** against a 10.67 maximum, green channel
essentially **zero** (R 26 / G 1.7 / B 118). At that level the die *bodies* sit
at the same value as the tray floor and carry no local variance — only the
painted numerals do.

So detection bounds **the digits, not the dice**. The overlay makes it obvious:
boxes enclose painted numerals while the die bodies extend well outside them.

### This breaks reading, and crop size cannot fix it

Feeding the crops to gemma3:27b:

| crop context | outcome |
|---|---|
| 1.5× | numerals with no die visible → 3–4 of 7 wrong (a d6 showing 1 read as "16") |
| 3.5× | that d6 now reads "1" correctly, but crops #3–#6 each contain **two or three dice** |

Whole-die crops read correctly (d20 → 17, d12 → 10). Numeral-only crops fail,
exactly as predicted: a reader cannot identify the up-face without seeing the
die. But widening the context far enough to capture the die also captures its
neighbours, and then there is no single right answer either.

`context` is now a request parameter with a UI slider, because no fixed value is
right — **and needing to tune it at all is the symptom, not the disease.**

### The grayscale conversion was throwing the signal away

Before blaming the lighting entirely: `cv2.cvtColor(BGR2GRAY)` is fixed at
`0.299R + 0.587G + 0.114B`. Under this illumination that weighting is actively
destructive — it gives **59% weight to the channel with SNR 0.71 and 47% of its
pixels clipped to zero**, and 11% to the only channel carrying the scene.

Measured contrast within the tray:

| source | std | pixels at zero |
|---|---|---|
| luminance grey | 8.5 | 0% |
| **B channel** | **27.3** | 0% |
| R channel | 12.5 | 6% |
| G channel | 6.6 | 50% |

`to_gray()` now scores each candidate per frame — contrast, discounted by
clipping — and picks the best. It chooses per-frame rather than hard-coding
blue so it still does the right thing once the lighting is fixed and luminance
becomes the better source.

**Effect, same scene, same lighting:**

| | luminance | auto → B channel |
|---|---|---|
| blobs / clean | 7 / 2 | 6 / **4** |
| what boxes enclose | painted numerals | **whole dice** |
| gemma3 on clean crops | 3–4 of 7 wrong | **all correct** |

A d8 that read "6" now reads **3**. The genuinely-touching d12+d8 pair is
correctly flagged rather than silently cropped as one die.

Two false positives on the bright specular strip of the right tray wall were
fixed by tightening the calibration quad — the wall was inside it.

### Can the purple cast be white-balanced away? No.

Worth recording because it looks like it should work. White balance is
per-channel multiplication, and it cannot invent information that was never
captured. Neutralising this scene needs a green gain of **68.7×** applied to a
channel whose noise exceeds its signal, with half its pixels already at hard
zero. The result is a noise storm, not a corrected image
(`test-data/_wb_demo.png`, panel 2).

Choosing a better channel works; correcting a dead one does not.

### The blocker is light, not code

Under the earlier handheld white-light frames the same variance segmentation
produced solid whole-die blobs at solidity 0.9+. Nothing about the algorithm
changed; the illumination did. Any neutral broad-spectrum light on the tray
should restore die-body separation immediately — even a desk lamp, well before
the 850 nm illuminator arrives.

This is also the clearest possible argument for the original lighting plan:
controlled illumination that is immune to whatever the room's mood lighting is
doing. Purple LEDs are a worst case, but *any* ambient-dependent setup will
drift.

## Hardware notes

Glitter dice throw specular highlights that read as bright speckle inside the
die body (visible in every capture so far). They survived detection fine — the
variance approach does not care — but they will be noise for any numeral
classifier. Worth keeping a plain-coloured set around as a control.
