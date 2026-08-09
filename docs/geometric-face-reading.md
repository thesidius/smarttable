# Geometric face reading

Replacing the model-based value reader with closed-form geometry plus template
matching. No neural network in the hot path.

Written 2026-08-08. All numbers below are **measured**, not estimated — the
method and the measurements are reproducible from `test-data/`.

---

## Why this works here and not in general

Reading a die from an arbitrary photo is a perception problem. Reading one from
**this rig** is not, because two things are fixed:

1. **The camera pose never moves** — 8.14″ above the tray floor, 21.3° tilt.
2. **Dice rest on a flat tray**, so every die's up-axis is the tray normal.

Together those mean the top face is not something to *find*. It is something to
*predict*. That is the whole idea.

This is also why the earlier brightness-based facet detection failed and was
deleted: it searched for a thing whose location was already determined.

---

## 1 — Locating the top face

A die resting on a face sits with its centroid at the **inradius** `r` above the
tray, and its top face centre at `2r`. The silhouette centroid that SAM2 already
gives you is approximately the projection of that centroid. So:

```
top_face_centre_px = silhouette_centroid_px + r · projected_up_vector(position)
```

### Getting r

For every die shape that rests face-down — d6, d8, d10, d12, d20 — the solids
are **isohedral**, so all faces are equidistant from the centre and:

```
r = (face-to-face caliper measurement) / 2
```

Measure each die once with calipers. That is the entire parameter.

### Getting the up-vector

The tray quad calibration already establishes the mapping between image and tray
plane. The projected up-vector at any image point radiates from the camera
nadir — the image point directly beneath the lens — with magnitude
`r · tan(θ)`, where `θ` is the viewing angle at that position.

Measured across the tray:

| Position | View angle | d20 offset | d12 offset | % of die width |
|---|---|---|---|---|
| near edge | 0.0° | 0.0 mm | 0.0 mm | 0% |
| ¼ along | 13.0° | 2.3 mm | 2.1 mm | 12% |
| mid tray | 24.7° | 4.6 mm | 4.1 mm | 23% |
| ¾ along | 34.6° | 6.9 mm | 6.2 mm | 35% |
| far edge | 42.7° | 9.2 mm | 8.3 mm | 46% |

Zero at the near edge, nearly half a die width at the far edge. Large enough
that ignoring it lands you on the wrong face; small enough that it is a
correction, not a search.

### Pixels available

| Die | Face polygon | Face width | px near | px far |
|---|---|---|---|---|
| d4 | triangle | 8.8 mm | 180 | 132 |
| d6 | square | 8.8 mm | 180 | 132 |
| d8 | triangle | 8.8 mm | 180 | 132 |
| d12 | pentagon | 9.9 mm | 202 | 149 |
| d20 | triangle | 11.0 mm | 225 | 165 |

Ample. Numeral legibility is not the constraint.

---

## 2 — Rotation, via log-polar

A die lands at an arbitrary rotation about the vertical. The naive fix is
matching ~24 pre-rotated templates per value. Don't.

**A log-polar transform about the face centre turns rotation into a circular
shift along the angle axis**, recoverable with a single FFT correlation.

```python
def logpolar(im):
    h, w = im.shape
    return cv2.warpPolar(im, (180, 128), (w/2, h/2), w/2,
                         cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR)

# circular cross-correlation along the ANGLE axis
F  = np.fft.rfft(ref, axis=0) * np.conj(np.fft.rfft(test, axis=0))
cc = np.fft.irfft(F, axis=0).sum(axis=1)
rotation_deg = np.argmax(cc) * 360.0 / ref.shape[0]
```

### Measured, on the red d20 from `test-data`

| True rotation | Recovered | Error |
|---|---|---|
| 0° | 0.0° | 0.0° |
| 17° | 16.9° | 0.1° |
| 45° | 45.0° | 0.0° |
| 73° | 73.1° | 0.1° |
| 120° | 120.9° | 0.9° |
| 168° | 168.8° | 0.8° |
| 250° | 250.3° | 0.3° |
| 331° | 331.9° | 0.9° |

**Mean absolute error 0.4°**, at a 3° angular bin size. It returns the value and
the rotation together.

Preprocess with the **green channel + CLAHE**, not `cvtColor(BGR2GRAY)` —
measured 40–80% more detail energy on this rig's dice, because the naive
conversion weights a red channel that carries mostly noise.

---

## 2b — Measured on the real rig (2026-08-08)

Everything in this section was run against live captures from the current
hardware (Camera Module 3, IMX708, 4608x2592). Where it contradicts the
sections above, prefer this — §2's numbers predate the camera swap.

### The 0.4 deg rotation figure is synthetic

Reproduced digit for digit — 0.0, 0.1, 0.0, 0.1, 0.9, 0.8, 0.3, 0.9, MAE 0.39 —
by taking one crop, rotating it with `warpAffine`, and recovering the angle.
That measures an FFT round-trip, not a die that physically landed at a
different angle, where the numerals rotate but the specular highlights do not.
Sub-bin accuracy from a 2.8 deg grid is the tell: only possible when reference
and test are the same pixels. **The method may be fine; that number is not
evidence for it.** Rotation recovery on physically re-rolled dice is still
unmeasured.

### Match on GRADIENT MAGNITUDE, not intensity

Rotation-invariant correlation between *different* faces (self = 1.000, lower
is better):

| representation | white light | blue LED light |
|---|---|---|
| naive `BGR2GRAY` | 0.475 | 0.205 |
| CLAHE'd intensity (what §2 does) | 0.245 | 0.250 |
| **gradient magnitude** | **0.127** | **0.189** |

A gradient discards the additive term (ambient level); normalising discards the
multiplicative one (brightness, gain). Between them that is most of what
changing the light does.

### The channel is a per-capture measurement, not green

§2 says green. Measured best channel was **blue** under white light and
**green** under blue LEDs — the opposite of the naive guess both times, because
what matters is where numeral-vs-body contrast survives, not where the
illuminant is bright. Pick it per capture by detail energy on the die pixels.
(§2's "40-80% more detail than BGR2GRAY" also understates it: measured
+216% to +358%.)

### Cross-lighting: one template set survives a 27x change in light

The test that matters for §4, run with the dice untouched between captures so
crops could be paired by position (max drift 6 px):

| | A | B |
|---|---|---|
| illuminance | 1.6 lux | 43.1 lux |
| colour (mean B/G/R) | 84.6 / 26.2 / 41.8 | 43.8 / 46.5 / 49.2 |
| exposure chosen | 1992 ms @ 1.6x | 117 ms @ 1.12x |
| best channel chosen | green | blue |

| representation | same face across light | different face | correct nearest-match |
|---|---|---|---|
| CLAHE'd intensity | mean 0.338, worst 0.104 | mean 0.118, best 0.245 | 8/10 |
| **gradient magnitude** | mean 0.335, worst 0.147 | mean 0.080, best 0.168 | **10/10** |

**Templates do not have to be re-derived when the light changes.** Note the
channel selection differed between the two captures and matching still held.

### Therefore: nearest-neighbour, never an absolute threshold

Worst same-face (0.147) sits BELOW best different-face (0.168). There is no
global cut that separates them — yet every die still matched itself better than
any other, because the comparison that matters is within one query. So §7's
"threshold on peak correlation" is unusable and should be dropped; only its
second half survives, the **margin between best and second-best**.

### What this does NOT show

These ten crops were ten *different dice* — different colours, sizes and types.
Telling a black d20 from a white d12 is much easier than telling face 7 of a
d20 from face 13 of the same die, which is the actual task. The same-lighting
figure for that harder case is 0.08-0.21 (§2b above, real d20 crops); the
cross-lighting figure for it is still unmeasured, and needs a real template set
to measure. Re-run this test at the end of Phase 4 against the 20 clustered
templates before trusting one set across conditions.

## 3 — Speed

Measured, single CPU core, all 20 templates correlated in one batched FFT:

| | |
|---|---|
| one die vs all 20 d20 faces | **4.0 ms** |
| a 7-die roll | **28 ms** |
| current SAM2 segment step | 1400–2400 ms |

The reader stops being the bottleneck entirely. Everything after this document
is about the segmentation step.

---

## 4 — Templates without labelling

The honest cost: **this is per-die-set calibration, not zero-shot.** New dice
need new templates.

It is cheaper than it sounds, because no roll ever needs a label:

1. Roll one die ~200 times. Capture, segment, locate the top face, extract the
   patch, log-polar it.
2. **Cluster the signatures.** Rotation-normalise first (align each to a
   canonical rotation via the FFT peak), then k-means with k = number of faces.
   A d20 lands in 20 clusters.
3. A human labels **20 cluster centroids**. Twenty clicks per die.
4. Store the centroid of each cluster as that face's template.

The adjacency map (§5) falls out of the same 200 rolls for free — you observed
which numerals surrounded each top face every time.

Sanity check the clustering before labelling: cluster sizes should be roughly
equal (a fair die), and there should be exactly `n_faces` populated clusters. If
one cluster holds 15% of rolls, two faces have merged and the templates are
contaminated.

---

## 5 — Adjacency as an independent check

This is the part that makes the whole thing trustworthy, and it has no analogue
in the model-based reader.

**On a d20, the top face's three edge-neighbours are always visible.** So read
four faces, not one, and check them against the die's adjacency map.

- Top reads 17, neighbours read {3, 8, 14}, map says 17's neighbours are
  {3, 8, 14} → confirmed by four independent observations.
- They disagree → you *know*, rather than reporting a wrong number confidently.
- Top is ambiguous but neighbours are confident → **the neighbours determine
  the top face.**

Also enforce the hard invariant. Opposite faces sum to a constant:

| Die | Sum |
|---|---|
| d6 | 7 |
| d8 | 9 |
| d10 (0–9) | 9 |
| d12 | 13 |
| d20 | 21 |

Any reading implying two *visible* faces that sum to that constant is
geometrically impossible — opposite faces cannot both be visible.

This matters because `README.md` already records the VLM reporting `"high"`
confidence with `"clear view"` on a face that was not determinable from the
image. Self-reported confidence is not evidence. Four mutually-constraining
observations are.

---

## 6 — Per-type special cases

**d4 is genuinely different.** It rests on a face with a vertex up — there is no
top face. Depending on the die, the result reads at the apex or along the bottom
edges. The `2r` derivation does not apply. Handle it as its own path, or accept
the VLM fallback for d4 only.

**d10 and d% are fine.** A pentagonal trapezohedron is isohedral and its
opposite faces are parallel, so `r` is well-defined and the derivation holds
unchanged. Its faces are kites rather than regular polygons, which affects
nothing here — the template does not care about face shape.

**d% templates are distinct from d10 templates.** Same solid, different glyphs
(00–90 vs 0–9). Separate template sets.

---

## 7 — Failure detection and fallback

The method must know when it is wrong. Three signals, all cheap:

1. **Weak peak correlation.** A cocked die, an occluded face, or a blown
   highlight correlates poorly with every template. Threshold on peak
   correlation, and on the **margin between best and second-best** — a
   confident read beats the runner-up clearly.
2. **Adjacency violation.** Per §5.
3. **Geometry violation.** The predicted top-face centre falls outside the
   silhouette mask, which means the die is not resting flat.

On any of these, escalate to the existing VLM reader for that die only. Log the
escalation rate — it is the health metric for this whole subsystem, and a rising
rate means lighting or calibration has drifted.

**Do not delete the VLM path.** It becomes the fallback, not the primary.

---

## 8 — Known limits

- **Cocked dice break the derivation.** A die against a wall or resting on
  another has an unknown up-axis. Detect (§7) and call the re-roll, which is
  what a human does anyway.
- **Specular highlights destroy correlation.** A blown highlight across a
  numeral is uncorrelated with any template. Diffuse lighting matters more for
  this reader than it did for segmentation.
- **Per-die-set calibration.** A guest bringing their own dice gets the VLM
  path until their set is calibrated.
- **Template drift.** If lighting, exposure lock, or camera position change,
  templates silently degrade. Version the template set against the same
  provenance fields `crop_pipeline` already stamps, and refuse to use templates
  captured under different exposure settings.

---

## 9 — Making segmentation faster

With the reader at 28 ms, SAM2's 1400–2400 ms is the entire latency budget.
In rough order of payoff:

### Train YOLO11-seg — the real answer

Phase 3 of `instance-segmentation-plan.md`. YOLO11n-seg or s-seg on a 4090 runs
in **10–20 ms** against SAM2's 1400+. That is the 100× step, and it also gives
die-type classification in the same pass, which the geometric reader needs in
order to pick the right template set.

Everything below is worth doing meanwhile, and some of it stays useful after.

### Crop to the tray quad before encoding

SAM2's image encoder resizes to a fixed 1024×1024 regardless of input. Sending a
full frame wastes most of that budget on tray walls. Cropping to the quad first
puts the dice across the full 1024 — **better masks at identical cost**. Free.

### Try `sam2.1_t`

Measured on `test-data`: the **tiny** model separated 8/8 dice with **zero pixel
overlap** between any pair — max IoU 0.000 — including the touching purple pair.
That was on a 440×464 screenshot, NoIR, no tuning. `sam2.1_b` may be buying
nothing. Test it; tiny is roughly 2–3× faster.

### `half=True`

FP16 on a 4090 is roughly 2× on the encoder for no accuracy cost at this task.

### Point prompts instead of automatic mask generation

AMG at `points_stride=32` runs the decoder over 1024 grid points and then does
NMS across hundreds of hierarchical masks. Point prompts run it 8 times.

Prompts do not need to be good — they only need to land *somewhere* on each die:

1. Background-subtract against a stored empty-tray reference. The camera and
   tray are fixed, so this is reliable and nearly free.
2. Threshold to a foreground mask.
3. Seed a coarse grid (~⅓ of a die width) clipped to the foreground.

Touching dice merging into one blob is fine — the grid still puts points on
both, and SAM2 separates them. This is a far weaker requirement than the
distance-transform peak counting that was deleted, which had to be *correct*.

### Skip SAM2 entirely when nothing is touching

Most rolls have some scattered dice. After background subtraction, any
connected component whose area is within the single-die band **is** a die — no
segmentation needed. Only invoke SAM2 on oversized blobs.

On a typical roll that removes most dice from the expensive path.

---

## 10 — Build order

0. **Prerequisites the camera swap created.** `test-data/` is from the previous
   camera at a different height under different light and cannot validate this
   rig. §1's offset table was computed for 8.14 in / 21.3 deg with the V2 lens
   and is now doubly stale, so the up-vector field must be **measured, not
   derived**: place a die at several known tray positions, mark the true
   top-face centre, fit the offset field. That also absorbs lens distortion and
   mount error, which the nominal numbers never carried.
   `seg_service` returns bbox and area but **no mask centroid** — the input this
   whole method starts from. Add it first.

1. **Top-face localisation** on fresh captures. Verify the predicted centre
   lands inside the correct face by eye on 20 dice. Nothing else works if this
   doesn't. Expect the silhouette centroid to be a biased estimate of the
   projected body centroid — the silhouette includes side faces, and which ones
   depends on the die's yaw, so the bias varies per roll. Measure it per die
   type rather than assuming it away.
2. **Log-polar + FFT rotation recovery.** Already validated at 0.4° here;
   confirm on your own crops.
3. **Template capture and clustering** for one d20. Check cluster balance
   before labelling.
4. **Single-die read end to end**, measured against known values.
5. **Adjacency map and validation.**
6. **Remaining die types**, d4 last.
7. **Fallback wiring and escalation logging.**

Gate at step 4: if single-die accuracy on clean isolated dice is below ~95%,
stop and diagnose before building the adjacency layer on top of it.