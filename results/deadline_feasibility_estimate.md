# Reviewer-response work: cost/feasibility estimate (no runs started)

Generated 2026-08-08. Deadline: **Aug 15, 2026 → 7 days of runway from today.**
Hardware: single RTX 4050 Laptop GPU (confirmed available, `torch.cuda.is_available()==True`),
no second GPU — everything below is **strictly sequential**, nothing can overlap.

All per-epoch rates are derived from `LastWriteTime` deltas between this repo's own
`checkpoints/*_epoch_N.pth` files (checkpoints saved every 10 epochs), i.e. real observed
wall-clock from the actual training runs already on disk, not a generic guess. Full arithmetic:

| Script | Config | Observed rate | Basis |
|---|---|---|---|
| `distillation_v2.py` | teacher-in-loop, Wing+0.002·Bone, bs=32 | **26.64 min/epoch** (avg, ep10→100) | range 19.0–32.6 min/epoch across nine 10-epoch windows — noisy, likely shared-machine contention |
| `distillation_finetune.py` | teacher-in-loop, Wing+0.002·Bone, grad-clip, bs=32 | **17.49 min/epoch** (avg, ep110→150) | range 16.4–19.0 min/epoch across four windows — much cleaner/more consistent than the v2 run above, same per-step compute graph |
| `directsup_baseline.py` | **no** teacher, Wing only, bs=32 | **9.91 min/epoch** (full-run avg) / **8.00 min/epoch** (steady-state, ep50→100) | first 3 windows (ep10-40) ran 10.3–16.4 min/epoch, then settled to a flat 7.9–8.0 min/epoch for the rest — treat 8 min/epoch as the clean hardware-bound rate |
| `finetune_freihand.py` | no teacher, bs=32, FreiHAND | 6.20 min/epoch (avg, noisy 3.96–10.93) | context only, not used below |

Takeaway used throughout: **the teacher forward pass (HRNet inference + per-sample CPU heatmap
decode inside the training loop) is the dominant cost driver** — teacher-in-loop epochs run
~2–3× slower than teacher-free epochs. Any estimate below that involves the teacher uses the
17.5–26.6 min/epoch band; anything GT-only uses the 8–10 min/epoch band.

---

## 1. R2-05 — controlled Direct Supervision retrain

**Settings diff, `distillation_v2.py` vs `directsup_baseline.py`** (read in full):

| Setting | `distillation_v2.py` | `directsup_baseline.py` | Parity? |
|---|---|---|---|
| Init weights | `student_model/blazehand_landmark.pth` | same | ✅ already matched |
| Augmentation | Flip + RandomBBoxTransform(shift .1/.3, scale .75-1.25, rot 45°/.6) + PhotoDistort | identical pipeline, byte-for-byte | ✅ already matched |
| Optimizer / LR | Adam 1e-4, Cosine→1e-6, T_max=100 | identical | ✅ already matched |
| Batch size | 32 | 32 | ✅ already matched |
| Gradient clipping | none (Stage 1 has no clip; only `distillation_finetune.py` does) | none | ✅ already matched (both off) |
| Validation cadence | every 5 ep, 200 samples | every 5 ep, 200 samples | ✅ already matched |
| Epoch count | 100 | 100 | ✅ already matched |
| **Loss composition** | Wing(w=10,ε=2) **+ 0.002·Bone** | **Wing only, no bone term** | ❌ **real gap** — add `0.002 * bone_loss(...)` for true parity |
| **Target coordinate space** | teacher-decoded heatmap coords, `np.clip(...,0,255)` | `ds.gt_instances.keypoints` (train, L105) / `.keypoints[0]` (val, L149), clamped [0,255] | ❌ **this is a bug, not a design choice** — `gt_instances.keypoints` is *original, un-cropped image space* (mmpose's `TopdownAffine` never overwrites it); the crop-space field is `gt_instances.transformed_keypoints`, only populated if the pipeline uses `PackPoseInputs(pack_transformed=True)`. This exact bug was already found and fixed in the *eval* script (`directsup_gt_comprehensive_eval.py`) during the prior paper audit, but **the training script was never patched** — the checkpoints currently in `checkpoints/direct_sup_*.pth` were trained against the wrong-space target the whole time. |

Contrary to the framing "list every setting that differs" — most of the requested settings are
**already** at parity. The two real gaps are the missing bone-loss term and, far more importantly,
the coordinate-space bug in the training target itself. **A retrain that doesn't fix the second
item is not a controlled ablation — it just reproduces the same corrupted baseline the audit
already flagged**, so this fix is a hard prerequisite, not optional cleanup.

**Required pre-run code changes** (~20–30 min, no GPU time):
1. `directsup_baseline.py` train_pipeline: `PackPoseInputs()` → `PackPoseInputs(pack_transformed=True)`.
2. L105: `ds.gt_instances.keypoints` → `ds.gt_instances.transformed_keypoints`.
3. L149 (val): same substitution.
4. Add `loss = wing_loss(...) + 0.002 * bone_loss(student_coords, gt_targets)`.
5. Recommend a quick smoke test (run 1 epoch, sanity-check loss doesn't explode) before committing to the full run — ~10 min.

**Command:** `python directsup_baseline.py` (after the above edits), then
`python directsup_gt_comprehensive_eval.py` to score it.

**Wall-clock estimate:** target is GT, so the teacher is never invoked → use the **no-teacher
rate**, not the distillation rate. 100 epochs × 8–10 min/epoch = **800–1000 min ≈ 13.3–16.7
hours**, plus ~5 min for the final eval (based on this repo's own measured ~12 samples/sec
full-set eval throughput, `results/README.md`'s 684.9s/3-models rerun). **Total ≈ 14–17 hours
(~0.6–0.7 days).**

**Blocking dependency:** none external — RHD data, student init weights all present. The only
blocker is the mandatory code fix above.

---

## 2. R2-11 — ablations (50-epoch budget)

All six variants modify `distillation_v2.py`'s loss/target step, so all are **teacher-in-loop**
→ use the 17.5–26.6 min/epoch band (17.49 = this repo's cleanest teacher-in-loop measurement,
26.64 = the actual `distillation_v2.py` script's own historical average, which ran noisier;
reporting both rather than picking one).

Per 50-epoch run: 50 × 17.49–26.64 min = **875–1332 min ≈ 14.6–22.2 hours**.

| Variant | New code needed | Cost |
|---|---|---|
| (a) Bone off (λ=0) | parametrize the hardcoded `0.002` at L199 | 1 run: 14.6–22.2h |
| (b) λ sweep {0.0005, 0.002, 0.01} | same parametrization | 3 runs: 43.8–66.6h |
| (c) Wing→L1, Wing→MSE | swap `wing_loss(...)` for `F.l1_loss`/`F.mse_loss` | 2 runs: 29.2–44.4h |

**Full set (6 runs): 87.6–133.2 hours ≈ 3.65–5.55 days.**

**Cost-saving note:** λ=0.002 at 50 epochs is *already on disk* —
`checkpoints/distilled_v2_epoch_50.pth` is exactly that config/epoch from the existing 100-epoch
V2 run (saved 4/24, 06:00 AM). It doesn't need to be retrained, only (re-)evaluated (~2–5 min).
So **"cheapest useful subset" (Bone on/off + one λ alternative) only requires 2 *new* runs**
(bone-off λ=0, and one alternative e.g. λ=0.01), not 3:

**Cheapest useful subset: 2 runs = 29.2–44.4 hours ≈ 1.2–1.85 days**, plus ~5 min to re-score
the existing ep-50 checkpoint as the λ=0.002 reference point.

**Pre-run code changes** (~25–30 min): parametrize bone weight and loss-fn choice as CLI args or
per-variant script copies; **give each variant a distinct checkpoint output path** — as written,
every variant would overwrite `checkpoints/distilled_v2_epoch_*.pth` and clobber the others.

**Blocking dependency:** none beyond the code parametrization above.

---

## 3. R2-12 — seeds (3× {FT V2-150 full pipeline, parity-matched Direct Sup.}, full budget)

**FT V2-150 full pipeline per seed** = Stage 1 (100 ep, teacher-in-loop) + Stage 2 (up to 50 ep,
teacher-in-loop, early-stop patience = 5 val-checks = 25 epochs without PCK improvement — the
existing run never triggered it and went the full 50, so plan for the full 50 as the safe case):
- Stage 1: 100 × 17.49–26.64 min = 1749–2664 min = 29.2–44.4h
- Stage 2: 50 × 17.49 min (using the finetune script's own clean historical rate) = 874.5 min = 14.6h
- **Per seed: 43.8–59.0 hours**
- **3 seeds: 131.4–177.0 hours ≈ 5.5–7.4 days**

**Parity-matched Direct Sup. per seed** (R2-05's fixed script, no-teacher rate):
- 100 × 8–10 min = 800–1000 min = 13.3–16.7h
- **3 seeds: 40.0–50.0 hours ≈ 1.7–2.1 days**

**R2-12 total: 171.4–227.0 hours ≈ 7.1–9.5 days of continuous, uninterrupted single-GPU time.**

**This is the headline planning risk.** Today is Aug 8; the deadline is Aug 15 — **7 days of
runway total**. R2-12 alone, run back-to-back with zero interruption and zero contention from
anything else on the machine, consumes 100–136% of the *entire* remaining calendar time. As
specified (3 seeds × both models × full budget), **this does not fit before Aug 15** on this
hardware, full stop — there's no room left for R2-05, R2-11, R2-15, re-evaluation, or writing.

**Descoping options if this item is kept:**
- 2 seeds instead of 3: ~114–151h (4.75–6.3 days) — still consumes nearly the whole runway alone.
- FT V2-150 seeds only, single (non-reseeded) Direct Sup. run: ~131–177h (5.5–7.4 days) + 13.3–16.7h — marginal savings, still dominates the budget.
- Reduce Stage 1 seed variance runs to fewer epochs (e.g., 60 instead of 100) — changes what's being measured, not a free lunch; flagging but not recommending without your sign-off.

**Blocking dependency:** none external (same data/checkpoints as R2-05/R2-11); the blocker is
pure wall-clock against the deadline.

---

## 4. R2-14 — FreiHAND zero-shot eval

**Already done, and the data is already local.** `freihand/` contains the full extracted dataset
(130,240 training images, 3,960 evaluation images — matches FreiHAND's published 32,560 scenes ×
4 background variants; zips also present) — **no fetching needed.**

`results_freihand_zeroshot_gt.txt` already exists: FT V2-150 vs. FreiHAND (1,628-sample held-out
slice of the training set, used as pseudo-val since FreiHAND's public test split ships no 3D
annotations), **PCK@0.2=0.7573, MPJPE=36.01px, AUC=0.7187, Det=100%**.

**Caveat — this does not use the RHD2D crop protocol you asked to match.** I read
`freihand_dataset.py`: it does a plain whole-image resize (224×224 → 256×256, uniform scale),
with **no bbox-centered crop at all** — unlike every RHD2D-based script in this repo, which
crops via `GetBBoxCenterScale(padding=1.25)` + `TopdownAffine` before feeding the student. Since
FreiHAND images are captured hand-centered, the practical gap may be small, but it's a genuine
train/test protocol mismatch, and this repo already has a documented history of exactly this kind
of mismatch quietly distorting numbers (see the coordinate-space bug family fixed elsewhere in
`results/`). The existing result answers "zero-shot on FreiHAND, whole-image protocol," not
"...matching the RHD2D crop protocol."

**Cost to actually match the protocol:** derive a bbox from each sample's projected 2D keypoints
(min/max + 1.25 padding, i.e. RHD2D's own convention) and crop+resize instead of whole-image
resize — self-contained numpy/cv2 change, no mmpose pipeline dependency, **~30–45 min dev**, then
rerun the eval loop. Based on this repo's own measured eval throughput (`results/README.md`:
684.9s for 3 models × 2,727 samples ≈ 12 samples/sec including all overhead), 1,628 samples ≈
**2.3 min**; the full 3,960-sample true evaluation/ split ≈ **5.5 min**.

**Total R2-14: ~0 min if the existing whole-image-protocol number is accepted (with a footnote
disclosing the protocol); ~35–50 min end-to-end if redone to genuinely match RHD2D's crop
protocol (recommended).** Either way, **no training, and no blocking dependency** — data,
checkpoint, and GPU are all already in place.

---

## 5. R2-15 — soft-argmax / integral-regression baseline

Needs a new decode function — spatial-softmax over each of the teacher's 64×64 channels +
expected-coordinate ("integral") readout, rescaled ×4 to 256×256 — as a drop-in replacement for
`codec.decode(hms)` (MSRAHeatmap+DARK) inside `distillation_v2.py`'s target-generation step. This
is a well-known, low-risk technique (Sun et al. integral regression), self-contained since the raw
heatmap tensor (`sample.pred_fields.heatmaps`) is already available in the loop.

**Dev cost (not GPU time):** ~1–2h to write + wire in as a swappable codec, plus a sanity check
comparing soft-argmax output against the existing MSRAHeatmap decode on a few batches to confirm
scale/orientation are right before committing GPU time — recommend also budgeting a **short
smoke run (2–3 epochs, ~1h)** before the full commitment, given the deadline stakes of discovering
a scaling bug at epoch 90.

**Training cost:** identical to V2 Stage 1 config, single seed, 100 epochs, teacher-in-loop →
100 × 17.49–26.64 min = **1749–2664 min ≈ 29.2–44.4 hours**.

**Total R2-15: ~1–2h dev + ~1h smoke test + 29.2–44.4h full run ≈ 31.2–47.4 hours (~1.3–2.0
days).**

**Blocking dependency:** none — teacher checkpoint/config, RHD data, student init weights all
present. Purely an implementation-then-train item.

---

## Ranking

### By estimated wall-clock (ascending)

| Rank | Item | Estimate | Notes |
|---|---|---|---|
| 1 | **R2-14** FreiHAND zero-shot | ~0–1h | already computed; ~50 min if redone for RHD-matched protocol |
| 2 | **R2-05** Direct Sup. parity retrain | ~14–17h (~0.7 day) | requires mandatory bug-fix first |
| 3 | **R2-11** cheapest subset (2 runs) | ~29–44h (~1.2–1.85 days) | reuses existing ep-50 checkpoint |
| 3 | **R2-15** soft-argmax | ~31–47h (~1.3–2.0 days) | comparable cost to R2-11's subset |
| 4 | **R2-11** full set (6 runs) | ~88–133h (~3.65–5.55 days) | |
| 5 | **R2-12** seeds (6 runs, full spec) | **~171–227h (~7.1–9.5 days)** | exceeds the entire 7-day runway alone |

### By how directly it closes its reviewer item (my judgment, flag if you see it differently)

1. **R2-05** — fixes a real confound in the headline baseline; high rigor payoff for low cost.
2. **R2-12** — seed variance is usually what "is this difference real" reviewers want, but only
   if it actually finishes; a partial/descoped version communicates less than the full spec asks for.
3. **R2-11** — explains *why* the loss design was chosen; the 2-run cheap subset covers the most
   likely reviewer question (does bone loss matter) for a fraction of the full-set cost.
4. **R2-14** — demonstrates generalization beyond RHD; already essentially free, worth doing
   regardless, but doesn't address a correctness concern the way R2-05 does.
5. **R2-15** — a genuinely different research question (alternative decode), not a rigor fix;
   valuable but the least urgent under deadline pressure unless a reviewer specifically demanded it.

### By blocking dependency

- **R2-14**: no blocker (data present, protocol caveat is a quality issue, not a blocker).
- **R2-05**: blocked only by the mandatory training-script coordinate-space fix (~20–30 min, in-repo).
- **R2-11**: blocked only by parametrizing loss variants + fixing checkpoint-name collisions (~25–30 min).
- **R2-15**: blocked only by writing the soft-argmax decode (~1–2h dev).
- **R2-12**: no code blocker — the entire blocker is wall-clock against Aug 15.

---

## Portfolio math against the Aug 15 deadline

Running everything **except R2-12** back-to-back, worst case:
R2-14 (1h) + R2-05 (17h) + R2-11 cheapest subset (44h) + R2-15 (47h) ≈ **109h ≈ 4.5 days**,
leaving ~2.5 days of the 7-day runway for the final eval passes, the already-known paper-table
fixes, and write-up — tight but plausible if the machine runs largely unattended starting now.

**R2-12 as fully specified (3 seeds × both models × full budget) does not fit in the remaining
runway under any combination with the other four items** — it alone is ≥7.1 days. If R2-12 is
must-have for the rebuttal, it needs either a hard descope (see options in §3) or to run on
different/additional compute in parallel with everything else, not sequentially on this laptop.

**No runs have been started.** Awaiting your go-ahead on: (1) which items to run, (2) whether to
apply the mandatory R2-05 code fix now, (3) how to descope R2-12 if it's staying in scope.
