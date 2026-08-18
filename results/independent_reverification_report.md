# Independent Re-verification Report

**Date:** 2026-08-08
**Method:** Full independent re-derivation — no existing eval script was re-run. Model
identity was checked byte-for-byte against upstream sources. The coordinate-space bug
was re-derived from `mmpose`'s own transform source, not from prior write-ups. All
accuracy numbers were recomputed from scratch by a new script, `independent_reverify.py`,
against the full 2,727-sample RHD2D test set.

## Bottom line

**Every number currently used in the paper draft that could be checked against a
tracked reference is confirmed.** Of 156 model/metric/joint comparisons run against
`results/table_iv_vii_viii_gt_space_FINAL.csv` and `results/model_identity_and_results_MASTER.csv`,
**155 PASS** within a 0.5% tolerance and essentially reproduce to 3–4 decimal places.
**1 FAIL**, and it is noise, not a real discrepancy: the *Untrained* (random-init)
baseline's PCK@0.2 came back 0.001362 vs. the recorded 0.0014 — a difference of about
2 joint-detections out of 57,267 scored (2,727 samples × 21 joints), on a model whose
entire point is to score near zero. See "The one FAIL" below.

Model identity is also fully confirmed: the student architecture is a byte-identical
copy of the public zmurez/MediaPipePyTorch port (code and pretrained weights both
hash-match upstream), and the teacher checkpoint is byte-identical to the official
MMPose model-zoo file, reproducing the official PCK=0.9918/AUC=0.9023/EPE=2.18 exactly
via MMPose's own `Runner.test()` harness.

Two things below are **not** a re-confirmation of a paper number but new findings from
this audit, both already load-bearing for how the results should be read:

1. **`directsup_baseline.py` — the Direct Sup. TRAINING script itself — has the
   coordinate-space bug**, not just its eval scripts. The Direct Sup. checkpoint was
   trained against `gt_instances.keypoints` (original-image space) while its own
   output is a 256-crop-space prediction. This is a training-time defect baked into
   `checkpoints/direct_sup_*.pth`, confirmed directly against `mmpose`'s
   `TopdownAffine` source (Part A/B below).
2. **`experiments/teacher_eval_B_fixed.py` is mislabeled.** Despite its name and an
   explicit comment claiming to use "crop-space GT from `gt_instances.keypoints`
   (after TopdownAffine)", that claim is false — `TopdownAffine` never writes to
   `results['keypoints']` — so this "_fixed" script still has the bug.

---

## Part A — Identity confirmation

### A.1 Student: zmurez/MediaPipePyTorch port

`student_model/blazehand_landmark.py`, `blazebase.py`, `blazepalm.py`, `LICENSE`, and
`README.md` were diffed byte-for-byte against
`raw.githubusercontent.com/zmurez/MediaPipePyTorch/master/...` — **all five files are
IDENTICAL**, not just architecturally similar. `student_model/LICENSE` credits Zak Murez
directly ("Conversion of MediaPipe from TFLite to PyTorch done by Zak Murez in June
2020... builds upon hollance/BlazeFace-PyTorch"), matching the upstream repo exactly.

The pretrained weights are also confirmed byte-identical: `sha256(student_model/
blazehand_landmark.pth)` = `fd0be668…bcb80`, matching a fresh download of
`zmurez/MediaPipePyTorch/master/blazehand_landmark.pth` exactly (8,090,697 bytes).

**Checkpoint provenance for the four training stages** (path / size / last-modified):

| Stage | File | Size (bytes) | Modified |
|---|---|---|---|
| (a) Pre-distillation init | `student_model/blazehand_landmark.pth` | 8,090,697 | 2026-02-12 01:36 |
| (b) Direct Sup. baseline | `checkpoints/direct_sup_best.pth` | 8,124,182 | 2026-04-26 00:07 |
| (c) Dist. V2-100 | `checkpoints/distilled_v2_epoch_100.pth` | 8,125,548 | 2026-04-25 05:01 |
| (d) FT V2-150 | `checkpoints/distilled_v2_ft_epoch_150.pth` | 8,126,106 | 2026-05-04 12:45 |

Each path was confirmed as the actual checkpoint loaded by its corresponding "golden"
eval script (`directsup_gt_comprehensive_eval.py`, `experiments/v2_GT_comprehensive_eval.py`,
`experiments/distilled_v2_ft_ep150_gt_comprehensive_eval.py`), not inferred from filename.

### A.2 Teacher: official MMPose HRNetV2-W18+DARK checkpoint

`checkpoints/hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth` (38,951,119 bytes) —
`sha256` = `4df3a347…316b9`. The hash **starts with `4df3a347`**, matching the file's own
OpenMMLab hash-suffix naming convention, and is **byte-identical** to a fresh download of
`download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth`,
the exact URL listed in MMPose's own model-zoo metadata
(`mmpose/configs/hand_2d_keypoint/topdown_heatmap/rhd2d/hrnetv2_dark_rhd2d.yml`), which
also records the official numbers this checkpoint is supposed to reproduce: PCK@0.2=0.992,
AUC=0.903, EPE=2.18.

The config file used, `checkpoints/td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256.py`,
looked like a modified copy on a naive text diff (different key ordering, `_base_`
inheritance flattened), but loading both through `mmengine.config.Config.fromfile()` and
comparing the resolved dicts shows **zero semantic differences** in model architecture,
optimizer, loss, or evaluation protocol. The only differences are `data_prefix.img` and
`ann_file` values, which just point at this repo's local dataset layout
(`evaluation/color/`, `rhd_evaluation.json` vs. the upstream `rhd_test.json` name) — a
path adaptation, not a modification of what's being trained/evaluated.

**Teacher is fully trained, not frozen/pretrained-only** — confirmed both by the
checkpoint hash match above and by independently reproducing its accuracy via MMPose's
own `Runner.test()` harness in Part C: PCK=0.991763, AUC=0.902255, EPE=2.182650,
matching the official 0.9918/0.9023/2.18 to 4+ decimal places.

### A.3 Training scripts vs. Table III / Figure 2

Read end-to-end: `distillation_v2.py`, `distillation_finetune.py`, `directsup_baseline.py`.

| Claim (Table III "V2" column / Fig. 2) | Code | Match? |
|---|---|---|
| Phase 1: epochs 1–100, lr 1e-4, Adam, batch 32 | `distillation_v2.py` L131,136,117: `Adam(lr=1e-4)`, `EPOCHS=100`, `batch_size=32` | ✅ |
| Cosine `1e-4→1e-6` over 100 epochs | L132-134: `CosineAnnealingLR(T_max=100, eta_min=1e-6)` | ✅ |
| Loss = Wing + 0.002·Bone vs. **teacher-decoded** coords | L197-199: `loss = loss_w + 0.002*loss_b`; targets built from `codec.decode(teacher heatmap)`, L163-190 | ✅ |
| Checkpoint: `distilled_v2_epoch_100.pth` | L275-276: saved at `epoch==EPOCHS` | ✅ |
| Phase 2: epochs 101–150, lr 5e-5, grad clip 5.0, early stop patience 25 | `distillation_finetune.py` L109-116: `EPOCHS=50, START_EPOCH=100`, `Adam(lr=5e-5)`, `clip_grad_norm_(max_norm=5.0)` L174, `patience=5` val-checks × `VAL_EVERY=5` = 25 epochs L116,244-247 | ✅ |
| Init from `distilled_v2_epoch_100.pth` | L57-59 | ✅ |
| Output: `distilled_v2_ft_epoch_150.pth` | L252-253 | ✅ |
| V1 vs. V2 table: loss MSE→Wing+Bone, no-aug→aug, fixed-lr→cosine, no-clip→[0,255], 210ep→100+50ep | Confirmed by the Phase 1/2 code above; V1 itself is not present as a runnable script in the current tree (superseded) | ✅ (V2 side); V1 not independently re-checked (no longer in tree) |

**Mismatch found (not a paper-vs-code mismatch, but a target-coordinate defect):**
`directsup_baseline.py` L105 reads `kps = ds.gt_instances.keypoints` as the training
target and L102 comments *"Already transformed to 256×256 space by the pipeline"* — this
is factually wrong. The pipeline at L41-58 uses plain `dict(type='PackPoseInputs')`
(no `pack_transformed=True`), so per `mmpose/datasets/transforms/topdown_transforms.py`
(read directly, see Part B), `gt_instances.keypoints` is **never touched** by
`TopdownAffine` and stays in original RHD image space (~320×320), while
`student_coords = pred_landmarks[:, :, :2] * 256` is a 256-crop-space prediction. The
wing-loss training objective at L117 is therefore comparing two different coordinate
spaces for the entire Direct Sup. training run — this is a defect in the checkpoint
itself, not merely in how it's evaluated afterward. `directsup_baseline.py`'s own
validation loop (L149-151) has the identical bug.

Distillation V2/FT do **not** have this defect: `distillation_v2.py` and
`distillation_finetune.py` never read `gt_instances.keypoints` at all (confirmed by
grep — zero matches in either file); their training target is the teacher's own
heatmap decode, entirely independent of the `PackPoseInputs`/`TopdownAffine` GT-packing
path that carries the bug.

---

## Part B — Repo-wide coordinate-space bug sweep

Grepped every `.py` file in the repo (125 files, excluding the vendored `mmpose/` and
`.venv/` trees, which are third-party/dependency code, not this project's own scripts)
for `gt_instances.keypoints`, `gt_instances.transformed_keypoints`, and
`pack_transformed`. 18 files reference the pattern; classified below by actually reading
each one (not by filename convention).

**Root cause, confirmed directly against installed `mmpose` source** (not taken on
faith from any prior write-up): `mmpose/datasets/transforms/topdown_transforms.py`,
`TopdownAffine.transform()`, L120-128 — writes the crop-warped keypoints to
`results['transformed_keypoints']` and **never reassigns `results['keypoints']`**.
`mmpose/datasets/transforms/formatting.py`, `PackPoseInputs.instance_mapping_table`
maps `gt_instances.keypoints` straight from that untouched `results['keypoints']`.
Only `PackPoseInputs(pack_transformed=True)` additionally exposes
`gt_instances.transformed_keypoints`. So `gt_instances.keypoints` is **original-image
space**, not 256×256 crop space, for every pipeline in this repo.

| File | Status | Why |
|---|---|---|
| `directsup_gt_comprehensive_eval.py` | **Clean** | `pack_transformed=True` + `gt_instances.transformed_keypoints`, clipped [0,255] |
| `mediapipe_vs_blazehand_eval.py` | **Clean** | same pattern; explicit comment cites the exact bug it avoids |
| `experiments/distilled_v2_ft_ep150_gt_comprehensive_eval.py` | **Clean** | same pattern |
| `experiments/v2_GT_comprehensive_eval.py` | **Clean** | same pattern |
| `scripts/rerun_dual_space_eval.py` | **Clean** | same pattern; deliberately also computes teacher-decoded-space for comparison, correctly labeled |
| `scripts/rescore_detection.py` | **Clean** | same pattern (3 occurrences, all correct) |
| `scripts/untrained_gt_eval.py` | **Clean** | same pattern |
| `experiments/teacher_eval_A_fixed.py` | **Clean** | avoids the field entirely — loads GT straight from the raw annotation JSON and compares against `pred_instances.keypoints` (mmpose's own inverse-affine'd, original-space prediction); internally consistent |
| `teacher_comprehensive_eval.py` (root) | **Clean** | uses mmpose's `Runner.test()` + `pred_instances.keypoints` (already inverse-affined to original space by `TopdownPoseEstimator`) vs. `gt_instances.keypoints` (also original space) — self-consistent by construction, `pack_transformed` not needed for this code path |
| `directsup_baseline.py` | **BUGGED — training-time** | see Part A.3; corrupts `checkpoints/direct_sup_*.pth` itself |
| `mediapipe_gt_eval.py` | **BUGGED** | plain `PackPoseInputs()`; MediaPipe SDK output (crop-space) scored against `gt_instances.keypoints` (original-space) |
| `experiments/directsup_learning_curve.py` | **BUGGED** | 2 occurrences (L82, L149); produces the committed `direct_sup_learning_curve.csv` |
| `experiments/mediapipe_sdk_eval.py` | **BUGGED** | comment falsely claims GT is "in 256x256 transformed space" — it reads `.keypoints`, not `.transformed_keypoints`, and the pipeline has no `pack_transformed=True` |
| `experiments/teacher_gt_comprehensive_eval.py` | **BUGGED** | teacher target via raw `codec.decode(heatmap)` (crop-space) vs. `gt_instances.keypoints` (original-space); produces the "~83px broken number" independently reproduced in Part C |
| `experiments/teacher_eval_B_fixed.py` | **BUGGED, mislabeled** | despite the name and an explicit comment claiming "crop-space GT from `gt_instances.keypoints` (after TopdownAffine)", that claim is false per the root-cause finding above — this file was never actually fixed |
| `experiments/dataloader_test.py` | N/A | prints `.shape` for a debug sanity check only; no scoring |
| `experiments/comprehensive_eval.py` | N/A | fetches `gt_instances.keypoints` (L59) but the variable is never used in any metric (scores student-vs-teacher only); also references a checkpoint that doesn't exist in this repo (`checkpoints/distilled_student_epoch_210.pth`), so the script cannot currently run at all — dead/obsolete code |
| `scripts/teacher_target_sanity.py` | N/A (diagnostic) | intentionally uses `gt_instances.keypoints` as the correct original-space reference, specifically to characterize this exact bug (raw vs. inverse-affined vs. GT); not a scoring path for any paper number |

`distillation_v2.py` and `distillation_finetune.py` were also grepped and confirmed to
have **zero** occurrences of `gt_instances.keypoints`/`transformed_keypoints` — their
distillation target is teacher-decoded, not GT-annotation-based, so this bug class does
not apply to them at all.

**Tally: 9 clean, 6 bugged, 3 not applicable, out of 18 files touching this pattern.**
Everything currently feeding the paper's tables (`directsup_gt_comprehensive_eval.py`,
`experiments/v2_GT_comprehensive_eval.py`,
`experiments/distilled_v2_ft_ep150_gt_comprehensive_eval.py`, plus the corrected
`scripts/rerun_dual_space_eval.py`, `scripts/rescore_detection.py`,
`scripts/untrained_gt_eval.py`, and `teacher_comprehensive_eval.py`) is clean. The six
bugged files are all either superseded (their "_GT"/non-suffixed twin already exists and
is clean) or diagnostic dead ends — except `directsup_baseline.py`, which is a genuine
open issue since it's the *training* script (see "What this does and doesn't mean" below).

**Secondary finding, out of scope for a deep dive but worth flagging:** every
GT-space eval/training script that feeds `BlazeHandLandmark` directly
(`student(images)`, no `extract_roi`) passes it the raw BGR tensor `mmpose`'s
`LoadImage` produces (`color_type='color'`, `imdecode_backend='cv2'`, confirmed via
`mmcv.transforms.LoadImageFromFile.__init__` defaults) with **no BGR→RGB conversion**,
even though `BlazeHandLandmark` is a port of a MediaPipe model designed for RGB input.
The teacher avoids this because MMPose's `PoseDataPreprocessor` has `bgr_to_rgb=True`
built in; `mediapipe_vs_blazehand_eval.py` avoids it by explicitly calling
`cv2.cvtColor(..., COLOR_BGR2RGB)`. `distillation_v2.py`, `distillation_finetune.py`,
and `directsup_baseline.py` do not convert, so all three trained student checkpoints
were both trained and evaluated on BGR-ordered pixels — internally consistent with each
other, but a channel-order mismatch relative to the architecture's original design and
relative to the zmurez/MediaPipe SDK rows in the same tables. Independently confirmed by
reading `mmcv.transforms.LoadImageFromFile.__init__` (`color_type='color'`) and every
training/eval script's tensor handling; not further quantified here since it's a
separate bug class from what this audit was scoped to check.

---

## Part C — Numeric re-verification (full 2,727-sample RHD2D test set)

`independent_reverify.py` (new script, root of repo) loads each of the 7 models fresh
from checkpoint and re-scores against `gt_instances.transformed_keypoints`
(fixed 51.2px PCK@0.2 threshold, all-frames scoring, detection failures penalized at
256px per-joint for the two-stage MediaPipe/zmurez pipelines, per the paper's stated
protocol) — the same protocol as the reference CSVs, since the point is to check whether
*those* numbers hold up, not to invent a new methodology. Manual 5-sample sanity check
before scoring confirmed `gt_instances.keypoints` (orig. space, range ~0–320) vs.
`gt_instances.transformed_keypoints` (crop space, range ~0–255) are visibly different
fields, as expected.

**Full aggregate table — old (tracked reference) vs. new (independently recomputed):**

| Model | Metric | Old | New | % diff | Verdict |
|---|---|---:|---:|---:|:---:|
| Direct Sup. (ep.25) | PCK@0.2 | 0.4248 | 0.424712 | 0.021% | PASS |
| Direct Sup. (ep.25) | AUC | 0.5377 | 0.537700 | 0.000% | PASS |
| Direct Sup. (ep.25) | MPJPE (px) | 59.8680 | 59.868130 | 0.000% | PASS |
| Dist. V2-100 | PCK@0.2 | 0.8190 | 0.818988 | 0.001% | PASS |
| Dist. V2-100 | AUC | 0.7670 | 0.767037 | 0.005% | PASS |
| Dist. V2-100 | MPJPE (px) | 30.3453 | 30.344137 | 0.004% | PASS |
| FT V2-150 | PCK@0.2 | 0.8188 | 0.818779 | 0.003% | PASS |
| FT V2-150 | AUC | 0.7686 | 0.768570 | 0.004% | PASS |
| FT V2-150 | MPJPE (px) | 30.1391 | 30.139282 | 0.001% | PASS |
| MediaPipe SDK (official) | PCK@0.2 | 0.5967 | 0.596696 | 0.001% | PASS |
| MediaPipe SDK (official) | AUC | 0.5526 | 0.552628 | 0.005% | PASS |
| MediaPipe SDK (official) | MPJPE (px) | 100.7540 | 100.754046 | 0.000% | PASS |
| BlazeHandLandmark (zmurez pre-distillation) | PCK@0.2 | 0.6267 | 0.626679 | 0.003% | PASS |
| BlazeHandLandmark (zmurez pre-distillation) | AUC | 0.5895 | 0.589484 | 0.003% | PASS |
| BlazeHandLandmark (zmurez pre-distillation) | MPJPE (px) | 78.2460 | 78.245997 | 0.000% | PASS |
| Teacher (HRNetV2-W18+DARK), official own-protocol | PCK@0.2 | 0.9918 | 0.991763 | 0.004% | PASS |
| Teacher (HRNetV2-W18+DARK), official own-protocol | AUC | 0.9023 | 0.902255 | 0.005% | PASS |
| Teacher (HRNetV2-W18+DARK), official own-protocol | MPJPE/EPE (px) | 2.1827 | 2.182650 | 0.002% | PASS |
| Untrained BlazeHandLandmark (random init.) | PCK@0.2 | 0.0014 | 0.001362 | **2.711%** | **FAIL** |
| Untrained BlazeHandLandmark (random init.) | AUC | 0.0137 | 0.013670 | 0.219% | PASS |
| Untrained BlazeHandLandmark (random init.) | MPJPE (px) | 188.8758 | 188.875824 | 0.000% | PASS |

**Per-joint table (Table VII / `table_iv_vii_viii_gt_space_FINAL.csv`), summarized —
126 comparisons (3 models × 21 joints × PCK+MPJPE), all PASS:**

| Model | n compared | max drift | mean drift |
|---|---:|---:|---:|
| Direct Sup. (ep.25) | 42 | 0.103% | 0.009% |
| Dist. V2-100 | 42 | 0.045% | 0.007% |
| FT V2-150 | 42 | 0.051% | 0.005% |

Full per-joint figures for all 21 joints × 3 models, plus every row above, are in
`results/independent_reverify_comparison.csv` (156 rows) and the raw recomputed numbers
(including Det.Rate and full per-joint breakdowns for all 7 models) are in
`results/independent_reverify_raw.csv`.

**Grand total: 156 comparisons, 155 PASS, 1 FAIL, at a 0.5% tolerance.**

### The one FAIL

Untrained BlazeHandLandmark's PCK@0.2 came back 0.001362 vs. the recorded 0.0014 — a
2.71% *relative* difference, but in absolute terms that's the model landing within 51.2px
on **78 keypoint-instances out of 57,267 scored**, vs. 80 previously — a difference of 2.
`torch.manual_seed(0)` was used to match the reference protocol
(`scripts/untrained_gt_eval.py --seed 0`), but bit-exact reproduction of a random
initialization also depends on the exact sequence and count of RNG draws made before
weight init, which can differ by a call or two between two independently-written
scripts without being wrong. AUC and MPJPE for the same model — both far more stable
statistics on near-random output — matched to 0.22% and 0.0004% respectively. This is
noise intrinsic to comparing two random initializations, not a data-processing or
identity discrepancy, and it does not change any conclusion the paper draws (the
untrained baseline is reported as ~0%, and it is ~0% either way).

### Teacher: two protocols, not one FAIL

The paper's teacher numbers (0.9918/0.9023/2.18) were produced under a **different**
protocol than the other six rows: MMPose's own `Runner.test()` with the officially
registered `PCKAccuracy`/`AUC`/`EPE` metrics (bbox-relative PCK threshold, original-image
coordinate space, automatic inverse-affine). Reproducing that number meant re-invoking
that same official harness — done in Part 3 of `independent_reverify.py`, reported above
as an exact match. Separately, `independent_reverify.py` also computed the teacher under
the *same* crop-space/fixed-51.2px-threshold protocol as the other six rows, for internal
consistency: **PCK@0.2=0.6866, MPJPE=43.70px, AUC=0.6811**. Neither reference CSV
contains an entry for this number, so it is reported here as new/supplementary, not as a
pass or fail. An informal "manual" bbox-relative cross-check was also computed by hand
from the same per-sample distances (see script Part 3) and came back around
PCK≈0.64/EPE≈18px — well off the official 0.9918 — indicating that hand-rolled formula
uses a different bbox-normalization convention than MMPose's actual `PCKAccuracy`/`AUC`
implementation. That manual number should be disregarded; the Runner-based figure above,
which used MMPose's own metric code rather than a reimplementation, is the one that
matches and is authoritative.

## What this does and doesn't mean for the paper

- Every accuracy/AUC/MPJPE number currently sourced from
  `table_iv_vii_viii_gt_space_FINAL.csv` and `model_identity_and_results_MASTER.csv` —
  i.e., everything feeding Tables IV, VII, and VIII plus the model-identity appendix —
  is independently reproduced to within noise. Nothing needs to be re-pulled or
  re-typeset on numeric grounds.
- Model identity (student architecture/weights, teacher checkpoint/config) is
  independently confirmed byte-for-byte against upstream sources, not just by re-running
  the same repo's own scripts.
- The one open issue from this audit that **isn't** a "did the number reproduce"
  question: `directsup_baseline.py` trained the Direct Sup. checkpoint against
  GT keypoints in the wrong coordinate space. The checkpoint's low reported accuracy
  (PCK@0.2=0.42) is thus consistent with — and to some extent *expected from* — a
  corrupted training objective, not only from direct supervision being weaker than
  distillation in principle. This doesn't change any number in the current tables (they
  score the checkpoint as it actually is), but it does affect how much weight the
  "Direct Sup. vs. distillation" comparison can carry as evidence about the two
  *training paradigms* in the abstract, versus these two particular training runs.
