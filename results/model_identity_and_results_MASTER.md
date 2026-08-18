# Model identity and results — consolidated master reference

Generated 2026-08-07. Full re-verification of every model/system compared in
the paper, before any edits to `paper/`. Companion data file:
`results/model_identity_and_results_MASTER.csv` (one row per model, machine-
readable). This file gives identity/background prose plus the reasoning
behind each cross-check. Nothing under `paper/` was touched.

Three coordinate-space bugs were already found and fixed earlier in this
audit (`directsup_gt_comprehensive_eval.py`, `experiments/v2_GT_comprehensive_eval.py`,
`experiments/distilled_v2_ft_ep150_gt_comprehensive_eval.py` — all read
`gt_instances.keypoints` instead of `gt_instances.transformed_keypoints`,
without `PackPoseInputs(pack_transformed=True)`). This pass checked every
*remaining* eval script in the repo for the identical bug rather than
assuming it was isolated to those three, per instruction. **It was not
isolated — two more scripts had it** (`mediapipe_vs_blazehand_eval.py` and
`scripts/rescore_detection.py`), and fixing them changed the MediaPipe SDK
and zmurez BlazeHandLandmark numbers far more dramatically than any of the
three already fixed.

---

## 1. MediaPipe SDK (official)

**Identity.** `pip` package `mediapipe==0.10.14`, `mediapipe.solutions.hands.Hands`
— Google's own proprietary, TFLite-based hand-landmark model, installed
directly from PyPI in this project's `.venv`. This is **not** the zmurez
PyTorch port (item 2 below); it's the real closed-weights SDK, run through
its official Python bindings. Every MediaPipe eval script in the repo
(`mediapipe_gt_eval.py`, `mediapipe_vs_blazehand_eval.py`,
`scripts/rescore_detection.py`, `experiments/mediapipe_sdk_eval.py`,
`experiments/mediapipe_sdk_tc_comprehensive_eval.py`,
`experiments/mediapipe_sdk_teacher_eval.py`) uses the identical call
signature: `Hands(static_image_mode=True, max_num_hands=1,
model_complexity=1, min_detection_confidence=0.01)`.

**Background.** MediaPipe Hands is Google's shipped, cross-platform hand-
tracking pipeline (palm detector + 21-keypoint landmark regressor). It's
used here purely as an external reference point — "how does a
general-purpose, un-fine-tuned commercial hand tracker do on RHD2D's
synthetic hands" — not as a component of anything trained in this thesis.

**Reconciling the two conflicting result sets.**
- `detection_breakdown.csv` (old): full 2,727 samples, Det=66.23%,
  PCK@0.2=0.2225 (all-frames), 0.3360 (detected-only).
- `mediapipe_gt_corrected.txt` (old): 500 samples, Det=69.40%, PCK@0.2=0.2426.
  **Its generating script genuinely could not be located.** I searched every
  MediaPipe-related script in the repo (`mediapipe_gt_eval.py` produces a
  *different* file, `mediapipe_gt_full2727.txt`, at 2,727 samples, not 500;
  the three `experiments/mediapipe_sdk_*` scripts compare against
  teacher-decoded coordinates, not GT, and write differently-named outputs;
  `three_way_student_comparison.py` is also teacher-decoded-space and
  produces `three_way_comparison.txt`). No script anywhere in the working
  tree reproduces `mediapipe_gt_corrected.txt`'s exact 500-sample,
  69.40%-detection numbers. Confirmed lost, exactly as the prior audit
  flagged — not re-estimated.

  Both old numbers turned out to share the same root cause anyway: both
  `mediapipe_gt_eval.py` (source of the 2,727-sample family) and
  `mediapipe_vs_blazehand_eval.py` read `gt_instances.keypoints` through a
  plain `PackPoseInputs()` — the identical bug already fixed elsewhere.

**Fresh, script-traceable replacement (this session).** Fixed
`mediapipe_vs_blazehand_eval.py` (`PackPoseInputs(pack_transformed=True)`,
`gt_instances.transformed_keypoints`, `np.clip(gt, 0, 255)` for consistency
with the other clipped scripts) and reran the full 2,727-sample set:
**PCK@0.2=0.5967, MPJPE=100.75px, AUC=0.5526, Det=66.23%** (all-frames,
fixed 51.2px threshold — the same protocol as every trained-model row, not
a detection-confidence-dependent metric). Detection rate is *unchanged*
from the old buggy run (66.23% both times) — expected, since detection
never depended on the GT field, which is a useful internal sanity check
that only the distance computation moved.

**Cross-check.** Independently re-derived via `scripts/rescore_detection.py`
(fixed the same way, rerun with `--force`) — a genuinely different code
path: it caches raw per-frame predictions once, then derives metrics from
the cache, rather than looping directly. Result: **0.5967 / 100.7540px /
0.5526**, exact to 4 decimal places. A "detected-only" variant also exists
(PCK=0.9010, MPJPE=21.58px, n=1806) but is not the number comparable to the
rest of the paper, which never conditions on detection success.

---

## 2. zmurez BlazeHandLandmark (pre-distillation weights)

**Identity.** `student_model/blazehand_landmark.pth` (8,090,697 bytes).
Confirmed via files bundled in the same directory: `student_model/LICENSE`
states verbatim *"Conversion of MediaPipe from TFLite to PyTorch done by Zak
Murez in June 2020. Website: https://zak.murez.com"*, and
`student_model/README.md` is the actual README from
[zmurez/MediaPipePyTorch](https://github.com/zmurez/MediaPipePyTorch)
("MediaPipe in PyTorch... Port of MediaPipe tflite models to PyTorch...
Builds upon hollance/BlazeFace-PyTorch"). This is a **from-scratch PyTorch
reimplementation** of MediaPipe's hand-landmark architecture with weights
converted from the original TFLite files — architecturally identical to
this thesis's own `BlazeHandLandmark` (same `student_model/blazehand_landmark.py`
class, since the thesis's student model *is* this architecture), but with
zmurez's original converted weights rather than any distillation applied.
Distinct from item 1: no dependency on the `mediapipe` pip package at all —
it's pure PyTorch.

**Background.** This checkpoint is the starting point the thesis's
distillation pipeline (`distillation_v2.py`) fine-tunes from. Reporting its
un-distilled accuracy is what makes "how much did distillation help" a
measurable claim rather than an assertion.

**Bug check.** `mediapipe_vs_blazehand_eval.py` — confirmed by direct
inspection **before** editing — used `PackPoseInputs()` (no
`pack_transformed`) and read `gt_instances.keypoints[0]`. Same bug, fixed
identically to the other five scripts today.

**Old (buggy) vs. new numbers**, full 2,727 samples:

| | PCK@0.2 | AUC | MPJPE | Det. Rate |
|---|---|---|---|---|
| Old (buggy GT-space) | 0.3010 | 0.3951 | 104.29px | 80.05% |
| **New (fixed GT-space)** | **0.6267** | **0.5895** | **78.25px** | 80.05% |

Detection rate again unchanged (80.05% both times), confirming the fix only
touched the distance computation.

**Cross-check.** Same as item 1 — `scripts/rescore_detection.py --force`,
independent code path: **0.6267 / 78.2460px / 0.5895**, exact to 4 decimal
places.

---

## 3. Trained models: Direct Sup., Dist. V2-100, FT V2-150, Teacher

These four were already fixed and cross-validated earlier in this audit —
restated here only, no new evaluation run, per instruction.

### 3a. Direct Sup. (ep.25)
**Identity:** `checkpoints/direct_sup_best.pth` (8,124,182 bytes,
2026-04-26) — `BlazeHandLandmark`, trained via direct wing-loss regression
against GT coordinates (best validation checkpoint, epoch 25).
**Background:** the non-distillation baseline — same architecture and
initialization path as the distilled models, but supervised directly on
RHD2D annotations instead of the teacher's decoded output, isolating what
distillation itself contributes.
**Coordinate space:** GT-space, `gt_instances.transformed_keypoints`,
clipped `[0,255]` — confirmed by direct inspection of
`directsup_gt_comprehensive_eval.py` post-fix.
**Results:** PCK@0.2=0.4248, AUC=0.5377, MPJPE=59.87px, Det=99.96%.
**Cross-check:** matches `scripts/rerun_dual_space_eval.py`'s independently
re-derived GT-space number in `results/table_viii_both_spaces.csv` exactly.

### 3b. Dist. V2-100
**Identity:** `checkpoints/distilled_v2_epoch_100.pth` (8,125,548 bytes,
2026-04-25) — `BlazeHandLandmark`, knowledge-distilled against the
Teacher's decoded heatmap output, Phase 1, epoch 100.
**Background:** the end of Phase 1 distillation, before the RHD-specific
fine-tuning phase that produces FT V2-150.
**Coordinate space:** GT-space, `gt_instances.transformed_keypoints`,
clipped — confirmed via `experiments/v2_GT_comprehensive_eval.py` post-fix.
**Results:** PCK@0.2=0.8190, AUC=0.7670, MPJPE=30.35px, Det=100.00%.
**Cross-check:** matches `results/table_viii_both_spaces.csv` exactly.

### 3c. FT V2-150 (final proposed model)
**Identity:** `checkpoints/distilled_v2_ft_epoch_150.pth` (8,126,106 bytes,
2026-05-04) — Dist. V2-100 fine-tuned a further 50 epochs on RHD2D
directly.
**Background:** the model the paper's headline claims are about.
**Coordinate space:** GT-space, `gt_instances.transformed_keypoints`,
clipped — confirmed via `experiments/distilled_v2_ft_ep150_gt_comprehensive_eval.py`
post-fix (this script initially lacked the `[0,255]` clip that the other
two had; added and re-run in the prior follow-up in this same audit).
**Results:** PCK@0.2=0.8188, AUC=0.7686, MPJPE=30.14px, Det=100.00%.
**Cross-check — now triple-verified:** (1) `results/table_viii_both_spaces.csv`
(`scripts/rerun_dual_space_eval.py`'s single consistent pipeline), and (2)
today's fixed `scripts/rescore_detection.py`, whose `student` row — a
*third*, independently-written code path, originally built for a different
purpose (Table V detection-rate context) — reproduces **0.8188 / 30.1391px
/ 0.7686** exactly. Three separately-written scripts now agree to 4 decimal
places.

### 3d. Teacher (HRNetV2-W18 + DARK)
**Identity:** `checkpoints/hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth`
(38,951,119 bytes) — MMPose's official model-zoo checkpoint for config
`mmpose/configs/hand_2d_keypoint/topdown_heatmap/rhd2d/td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256.py`.
Fully trained (not a frozen backbone), used here as the fixed upper-bound
reference the student architecture is distilled from.
**Coordinate space — important distinction from every model above:**
`teacher_comprehensive_eval.py` does **not** use the ad hoc
`PackPoseInputs`/`transformed_keypoints` pattern at all. It runs mmpose's
own `Runner.test()` harness (the same one `tools/test.py` uses), whose
`TopdownPoseEstimator` automatically applies the *inverse* affine transform
to bring heatmap-decoded predictions back into original-image space before
packaging them as `pred_instances.keypoints` — meaning predictions and
`gt_instances.keypoints` (untransformed) are correctly, self-consistently
in the *same* space by construction. `pack_transformed` is simply not a
relevant flag here; there is no bug to check for in this script because the
framework handles the space alignment automatically, unlike the six ad hoc
scripts elsewhere in the repo that reimplement scoring by hand. This was
verified by reading `PerJointHook.after_test_iter` directly, not assumed.
**Results:** PCK@0.2=0.9918, AUC=0.9023, MPJPE=2.18px, Det=100.00% (the
detection rate is not architecturally meaningful for this model — HRNet has
no hand-presence head; it always emits a heatmap).
**Cross-check — triple match:** the live `Runner.test()` console output,
the saved raw log `eval_results/teacher_comprehensive/20260512_145302/20260512_145302.json`
(`PCK=0.99176, AUC=0.90225, EPE=2.18265`), and the hardcoded reference
constant in `scripts/make_tables.py` (`0.992/0.902/2.21`) all agree within
rounding.
**Caveat (pre-existing, restated for completeness):** the Teacher's PCK
threshold is **bbox-relative** (0.2× each sample's GT bbox size), not the
fixed 51.2px threshold every other row in this table uses. The coordinate
space is correct, but the threshold convention is not directly comparable
— this was already flagged in `results/provenance_manifest.csv` and still
stands.

---

## 4. Untrained BlazeHandLandmark (random init.)

**Identity:** same `student_model/blazehand_landmark.py` architecture as
every other BlazeHandLandmark row, `torch.manual_seed(0)`, **no checkpoint
loaded** — literally the module's default random initialization.

**The problem.** No source file for the paper's claimed
0.005/0.025/186.96px/100% row existed anywhere in the repo
(`results/provenance_manifest.csv` row 16 flagged this). The paper's
**100% detection rate** for a randomly-initialized presence head is,
as previously flagged, almost certainly backwards.

**Two candidate replacements were on the table; neither was adopted
as-is:**
1. The paper's own hardcoded constant (0.005/0.025/186.96px/**100%**) —
   rejected: no traceable source, and the 100% detection rate is
   implausible on its face.
2. `three_way_comparison.txt`'s "Untrained" block
   (PCK=0.0047/MPJPE=187.11px/AUC=0.0249/**Det=0.00%**) — the direction
   (near-zero detection) is right, but reading `three_way_student_comparison.py`
   directly shows this number is **not GT-space at all**: that script scores
   every model against the *Teacher's own decoded heatmap* (`codec.decode(hms)`),
   never touching `gt_instances.keypoints` or `.transformed_keypoints` in any
   form. It's also only 500 samples, not the full test set. Using it as a
   drop-in "vs. GT" replacement would just trade one mislabeling for
   another.

**Resolution: fresh run.** Wrote `scripts/untrained_gt_eval.py`, identical
GT-space protocol to `directsup_gt_comprehensive_eval.py`
(`pack_transformed=True`, `transformed_keypoints`, clipped `[0,255]`, fixed
51.2px threshold), full 2,727 samples, random-init model, no training
involved (~2.5 min wall-clock). Result:
**PCK@0.2=0.0014, MPJPE=188.88px, AUC=0.0137, Det=0.00%.**
Writes `results/results_untrained_GT.txt`.

**Cross-check.** Reran with a different seed (`--seed 1`): PCK=0.0014,
MPJPE=188.66px, AUC=0.0138, Det=0.00% — stable across random
initializations, confirming this isn't an artifact of one unlucky seed.
Directionally consistent with `three_way_comparison.txt`'s block despite
the different space/sample-count (unsurprising: near-random predictions
score badly in essentially any coordinate space).

---

## Summary table (see CSV for full machine-readable version)

| Model | PCK@0.2 | AUC | MPJPE (px) | Det. Rate | Coordinate space |
|---|---|---|---|---|---|
| MediaPipe SDK (official) | 0.5967 | 0.5526 | 100.75 | 66.23% | GT-space (fixed) |
| BlazeHandLandmark (zmurez) | 0.6267 | 0.5895 | 78.25 | 80.05% | GT-space (fixed) |
| Direct Sup. (ep.25) | 0.4248 | 0.5377 | 59.87 | 99.96% | GT-space |
| Dist. V2-100 | 0.8190 | 0.7670 | 30.35 | 100.00% | GT-space |
| **FT V2-150** | **0.8188** | **0.7686** | **30.14** | 100.00% | GT-space |
| Teacher (HRNetV2-W18+DARK) | 0.9918 | 0.9023 | 2.18 | 100.00%* | own-protocol (bbox-relative threshold) |
| Untrained BlazeHandLandmark | 0.0014 | 0.0137 | 188.88 | 0.00% | GT-space (fresh) |

\* Not architecturally meaningful for the Teacher (no presence head).

**Two files intentionally not touched in this pass** (out of explicit
scope, flagged for a future follow-up if wanted): `mediapipe_gt_eval.py`
(superseded by `mediapipe_vs_blazehand_eval.py`'s more complete fresh run,
same bug still present in the unused script) and
`experiments/directsup_learning_curve.py` / `mediapipe_gt_eval.py`-adjacent
scripts feeding Fig. 3-style curves, which were out of this task's scope.
`paper/` was not touched anywhere in this pass.
