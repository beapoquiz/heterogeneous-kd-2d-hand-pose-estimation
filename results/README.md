# results/ — camera-ready numeric audit (paper #1571328446)

Generated 2026-08-07, revised 2026-08-07 (same day — see **CORRECTION**
below). Read/eval-only: no training configs were touched, no model was
retrained, nothing in `paper/` was edited. Every number below comes from
either (a) an existing checkpoint run through a script already in this repo,
or (b) `scripts/rerun_dual_space_eval.py`, written for this audit, which
scores all three models against both coordinate spaces from the same
forward pass so the two spaces can't diverge from re-running a model twice.

## CORRECTION (same-day revision)

**The first version of this package had the wrong headline finding.** It
reported a "reversal" where Direct Supervision wins in GT-space and the
distilled models win in teacher-decoded-space. That GT-space number for
Direct Supervision was itself computed with a coordinate-space bug — once
fixed, Direct Supervision actually loses to *both* distilled models in
*both* spaces, and by a wide margin. Root cause and full corrected numbers
below. The bug was caught because a fix to `scripts/rerun_dual_space_eval.py`
(changing `gt_instances.keypoints` to `gt_instances.transformed_keypoints`
with `PackPoseInputs(pack_transformed=True)`) landed in the working tree
mid-audit; verifying that fix against mmpose's actual source before trusting
it is what surfaced this.

## Root cause: `gt_instances.keypoints` is not in crop space

Confirmed directly from the installed mmpose package (not inferred):

- `mmpose/datasets/transforms/topdown_transforms.py`, `TopdownAffine.transform()`:
  it warps the image and writes the warped keypoints to
  `results['transformed_keypoints']`. **It never reassigns
  `results['keypoints']`** — that field is left exactly as it was before the
  crop/affine step, i.e. in the *original, un-cropped image coordinate
  space*.
- `mmpose/datasets/transforms/formatting.py`, `PackPoseInputs`: its
  `instance_mapping_table` maps `keypoints -> gt_instances.keypoints`
  directly from `results['keypoints']` — the untouched, original-space
  field. `results['transformed_keypoints']` (the correct, 256×256-crop-space
  field) only reaches `gt_instances.transformed_keypoints` if the pipeline
  passes `PackPoseInputs(pack_transformed=True)`.
- Every eval script audited in this repo calls plain `PackPoseInputs()`
  (no `pack_transformed=True`) and then reads `gt_instances.keypoints[0]` as
  "the ground truth" — meaning **every one of them has been comparing a
  256×256-crop-space model prediction against an original-image-space
  annotation.** Measured directly (`dataset/rhd` test split, 50 samples):
  mean per-sample L2 gap between the two fields is **69.7px** (range
  28-150px) — comparable in size to the worst space-mismatch bug already
  documented in this package (the teacher's ~80px missing-inverse-affine
  bug). This is not a rounding-level discrepancy.

**This is worse than a shared eval-script bug — it's baked into training
too.** `directsup_baseline.py:101-110` trains Direct Supervision directly
against `gt_instances.keypoints` (comment at L101-102: "Already transformed
to 256×256 space by the pipeline" — false, per above) with
`wing_loss(student_coords, gt_targets)`, i.e. it was trained to make a
256-space-scaled output match an original-space target. By contrast,
`distillation_v2.py` / `distillation_finetune.py` never touch
`gt_instances.keypoints` at all — their target is the teacher's own decoded
heatmap output (`MSRAHeatmap.decode()`), which is already correctly in
256-crop space independent of this bug. **So Direct Supervision's training
objective itself was corrupted; distillation's was not.**

That explains the shape of the original (wrong) result: Direct Sup. was
evaluated with the *same* broken reference its training used, so the two
errors partially canceled and it looked artificially competitive
(PCK≈0.74). The distilled models were evaluated against that same broken
reference despite being trained against a *correct* one, so they looked
artificially bad (PCK≈0.34). Once the reference is fixed to genuinely match
crop space, the artificial cancellation for Direct Sup. disappears and its
real weakness shows.

## Corrected headline finding

| Model | GT-space PCK@0.2 (corrected) | GT-space PCK@0.2 (original, WRONG) | teacher-decoded-space PCK@0.2 (unaffected by this bug) |
|---|---|---|---|
| Direct Sup. (ep.25) | **0.4248** | ~~0.7447~~ | 0.3378 |
| Dist. V2-100 | **0.8190** | ~~0.3367~~ | 0.7460 |
| FT V2-150 | **0.8188** | ~~0.3381~~ | 0.7479 |

(teacher-decoded-space numbers are untouched by this correction: the
teacher-decoded target comes from `MSRAHeatmap.decode()` on the teacher's
own heatmap, never from `gt_instances.keypoints`, so it was never affected.)

**Both distilled models now beat Direct Supervision in *both* spaces**, and
per-joint it's unanimous both ways: FT V2-150 beats Direct Sup. on 21/21
joints in teacher-decoded-space (already true in the first version) *and*
now also 21/21 joints in the correctly-computed GT-space (margins +0.20 to
+0.65 PCK — see `per_joint_winloss_gt_space.csv`). There is no more
"reversal" story — distillation wins cleanly either way once the reference
frame is actually correct.

What this means for the paper's own numbers: Table IV/VII/VIII's Direct
Sup. row (PCK=0.745, "vs. GT annotations") is not just mislabeled as to
*which* space it used (see "Still-valid findings" below) — even taking it
at face value as a GT-space number, **0.745 is not Direct Supervision's
real GT-space accuracy; 0.4248 is.** Conversely, the distilled models'
*real* GT-space accuracy (≈0.819) is meaningfully *better* than the
teacher-decoded-space number the paper actually reports for them
(0.746/0.748) — the paper is, if anything, **understating** how good the
distilled models are relative to real ground truth, while dramatically
**overstating** how good Direct Supervision is.

## Still-valid findings from the original audit

These are independent of the bug above and remain accurate:

- **Table IV/VII/VIII's shared "vs. GT annotations" caption is still false
  for the distilled-model rows** — they still come from teacher-decoded
  targets (`experiments/v2_ep100_comprehensive_eval.py`,
  `experiments/v2_ft_ep150_comprehensive_eval.py`), a labeling problem
  independent of whether the GT-space numbers were computed correctly.
- **Table V's "MediaPipe-inspired BlazeHandLandmark" mislabeling** and the
  **orphaned zmurez citation** (Step 5 below) still stand as described.
- **The teacher's own coordinate-space bug** (`teacher_gt_comprehensive_eval.py`,
  missing inverse-affine) still stands, and is a *different* bug from this
  one (it's on the prediction side, not the GT side — see Step 4).
- **Fig. 3 / Appendix F's 3-way MSE mismatch** still stands (Step 2).
- **Figure 6 webcam checkpoint provenance** (Addendum) still stands.

## Newly-flagged: same bug likely affects Step 5's MediaPipe/zmurez numbers too — NOT yet re-verified

`mediapipe_gt_eval.py`, `mediapipe_vs_blazehand_eval.py`, and
`scripts/rescore_detection.py` (source of `detection_breakdown.csv`, which
the original version of this package cited as independent corroboration)
**all read `gt_instances.keypoints[0]` the same unpatched way.** Their
"vs. GT" numbers for the official MediaPipe SDK and the zmurez
BlazeHandLandmark checkpoint (`mediapipe_gt_corrected.txt`,
`mediapipe_gt_full2727.txt`, `mediapipe_vs_blazehand_results.txt`,
`detection_breakdown.csv`) are therefore likely wrong too — probably
**understated** (unlike Direct Sup., neither of those two systems was
*trained* on this codebase's data, so there's no train/eval error
cancellation to inflate them; a pure eval-time space mismatch should only
hurt their measured accuracy). This was surfaced too late in the audit to
re-run before this delivery — `webcam_provenance_check.csv`/Figure-6 work
and the corrected 3-model rerun took priority. **Flagging explicitly per
your instructions rather than estimating a corrected number.** Re-running
those two scripts with `PackPoseInputs(pack_transformed=True)` +
`gt_instances.transformed_keypoints` (same fix already applied to
`scripts/rerun_dual_space_eval.py`) would resolve this — say the word if
you want that done now.

## Files

| File | Contents | Command | Runtime |
|---|---|---|---|
| `provenance_manifest.csv` | Coordinate-space provenance for every metric currently in Tables III-VIII / Figs 3-6, with exact file+line citations | manual code audit | ~45 min (research, no execution) |
| `table_viii_both_spaces.csv` | Direct Sup. / Dist.V2-100 / FT V2-150, full 2,727-sample RHD test set, scored against BOTH GT (**corrected**, `transformed_keypoints`) and teacher-decoded targets in one pass, overall + per-joint PCK@0.2/MPJPE/AUC | `python scripts/rerun_dual_space_eval.py` | **684.9s (11.4 min)**, 2,727 samples, 1× RTX 4050 Laptop GPU (corrected run; superseded an earlier 840.0s run that used the buggy GT reference) |
| `per_joint_winloss_gt_space.csv` | Direct Sup. vs. FT V2-150, per-joint PCK@0.2 winner, **corrected** GT-space only | same run as above | (same run) |
| `per_joint_winloss_teacher_space.csv` | Direct Sup. vs. FT V2-150, per-joint PCK@0.2 winner, teacher-decoded-space only (unaffected by the bug/fix) | same run as above | (same run) |
| `rerun_dual_space_summary.txt` | Human-readable summary of the corrected run | same run as above | (same run) |
| `fig3_pck_mse_by_epoch.csv` | Full per-epoch PCK/AUC/MPJPE/MSE series behind Fig. 3, plus the values hardcoded into the Fig. 3 tikz block and the Appendix F table, side by side | data export from `experiments/v2_learning_curve_complete.csv` + `paper/main.tex` | ~5 min (export only) |
| `table_iv_teacher_own_protocol.csv` | Teacher's own-pipeline PCK/AUC/MPJPE (mmpose Runner protocol) + a fresh `teacher_target_sanity.py` run quantifying the separate "missing inverse-affine" bug in `experiments/teacher_gt_comprehensive_eval.py` | `python scripts/teacher_target_sanity.py` (fresh, 20 samples) + reused `eval_results/teacher_comprehensive/20260512_145302/20260512_145302.json` | ~25s (fresh part only) |
| `detection_breakdown.csv` | Pre-existing (2026-08-05) artifact from `scripts/rescore_detection.py`; **kept as-is but now flagged above as likely affected by the same untransformed-keypoints bug** — no longer treated as independent corroboration | *(not re-run; already in the repo before this audit)* | n/a |
| `webcam_provenance_check.csv` | Per-screenshot watermark/timestamp evidence establishing that `paper/webcam_collage.png` (Fig. 6) was captured with FT V2-150, not Dist. V2-100 | manual file/image inspection | ~5 min |
| `_rerun_dual_space_full.log` | stdout/stderr of the **original, buggy** 840.0s run (kept for the record, do not use its numbers) | — | — |
| `_rerun_dual_space_full_v2.log` | stdout/stderr of the **corrected** 684.9s run — this is the one `table_viii_both_spaces.csv` etc. reflect | — | — |

## Step-by-step findings

### Step 0 — Coordinate-space provenance
See `provenance_manifest.csv`, plus the root-cause section above (the more
important finding, discovered after the manifest was first written — the
manifest documents the teacher-decoded-vs-GT mislabeling; it does not yet
have a row for the deeper `transformed_keypoints` bug, which supersedes
part of its analysis for the Direct Sup. row specifically).

- **Table IV (`tab:trained`) / Table VII (`tab:perjoint`) / Appendix E (`tab:full_per_joint`)**: caption says "vs. GT annotations" uniformly. The distilled-model rows are teacher-decoded-space (still true). The Direct Sup. row genuinely used `gt_instances.keypoints` as its label claims, but that field is not GT in the space that matters — see root cause above.
- **Table V (`tab:pretrain_ref`)**: "Untrained"/"MediaPipe-inspired" rows are hardcoded constants in `scripts/make_tables.py` (lines 75-78), not parsed from any current file; the closest surviving analogs are themselves now suspect per the Step 5 flag above.
- **Table VI (`tab:throughput`)**: pure speed benchmark, no coordinate-space dependency; unaffected by anything in this report.
- **Table VIII (`tab:consolidated`)**: mirrors Table IV + V's sources exactly.
- **Fig. 3 / Appendix F**: teacher-decoded-space throughout (unaffected by the GT bug, since it never reads `gt_instances.keypoints`), but has its own unrelated 3-way MSE mismatch — see `fig3_pck_mse_by_epoch.csv`.
- **README.md**: "Direct Supervision Baseline" row matches `results_direct_sup_GT.txt` — which is now known to be the *buggy* GT-space number (0.745), not real GT-space accuracy (0.425). "Student" row matches ep.150 (teacher-decoded-space, unaffected by this bug). "MediaPipe SDK (zmurez weights)" row — see Step 5 flag above, now suspect.

### Step 1 — Both-spaces re-run
`scripts/rerun_dual_space_eval.py` (now using `PackPoseInputs(pack_transformed=True)`
and `gt_instances.transformed_keypoints` for the GT target) loads the
teacher once, all three student checkpoints once, and for every one of the
2,727 RHD test images computes BOTH targets from that single sample, then
scores all three students against both — see `table_viii_both_spaces.csv`.
See "Corrected headline finding" above for the topline numbers.

### Step 2 — Training curves
Unaffected by the GT-space bug (the learning-curve script's target is
teacher-decoded, never `gt_instances.keypoints`). `fig3_pck_mse_by_epoch.csv`
still documents its own, separate 3-way MSE mismatch between the Fig. 3
tikz block, the Appendix F table, and the current `experiments/v2_learning_curve_complete.csv`
— unresolved, flagged as-is.

### Step 3 — Per-joint win/loss recount
`per_joint_winloss_gt_space.csv` now reflects the corrected GT target:
**FT V2-150 wins all 21/21 joints**, margins +0.20 to +0.65 PCK.
`per_joint_winloss_teacher_space.csv` is unchanged from the first version
(FT V2-150 wins all 21/21 joints there too, margins +0.29 to +0.60 PCK).
Both spaces now agree on the winner — see "Corrected headline finding."

### Step 4 — Teacher row provenance
Unchanged from the first version of this report — the teacher's numbers
come from `teacher_comprehensive_eval.py`'s mmengine-Runner-based path
(matching MMPose's own official benchmark, PCK=0.9918), not from
`gt_instances.keypoints`, so this bug does not touch it. Its own, separate
missing-inverse-affine bug (`teacher_gt_comprehensive_eval.py`) is
unaffected by today's finding and still stands as originally documented —
see `table_iv_teacher_own_protocol.csv`. The "identical conditions" caveat
(Runner-harness + bbox-relative threshold vs. every other row's fixed
51.2px threshold) also still stands.

### Step 5 — zmurez baseline provenance
The mislabeling and orphaned-citation findings still stand (Table V's
"MediaPipe-inspired BlazeHandLandmark" row is really the official SDK; the
zmurez `\bibitem` is never `\cite{}`'d). **New as of this revision**: the
actual PCK/AUC/MPJPE numbers behind both that row and the correctly-labeled
README "MediaPipe SDK (zmurez weights)" row are now suspected of the same
`gt_instances.keypoints` bug and have not been re-verified — see the
flagged section above. Do not treat `mediapipe_vs_blazehand_results.txt`'s
absolute numbers as reliable until re-run with the fix.

## Addendum — Figure 6 / webcam_collage.png checkpoint provenance

Unaffected by today's finding (this was a pure file/pixel provenance check,
no model evaluation involved). Still stands as originally reported:

- `webcam_demo.py` loads `checkpoints/distilled_v2_epoch_100.pth` (confirmed
  again on request: `webcam_demo.py:11`).
- `webcam_demo_finetuned.py` loads `checkpoints/distilled_v2_ft_epoch_150.pth`.
- All 8 surviving screenshots (`thesis_figures/webcam_demo_1.png`-`_8.png`,
  captured 2026-05-17 17:44-17:47) carry the `'RHD Fine-tuned | Epoch 150'`
  watermark burned into the frame by `webcam_demo_finetuned.py` — confirmed
  by directly viewing `_1`, `_5`, `_8`. `paper/webcam_collage.png`'s six
  panels are tight crops of this same session; no epoch-100 webcam capture
  survives anywhere in the repository.
- Both demo scripts write to the identical output path pattern
  (`thesis_figures/webcam_demo_{saved}.png`, counter restarting at 1 each
  run), so an earlier epoch-100 session, if one ever existed, would have
  been silently overwritten by this one with no versioning.
- You've indicated new epoch-100 screenshots have since been captured and
  uploaded (2026-08-07, after this audit's original delivery). Those are
  **not yet incorporated anywhere in this results package or the paper** —
  I have not located them in the repository as of this writing. Point me at
  where they landed (a path, or re-share them) and tell me explicitly if
  you want `paper/webcam_collage.png` / `main.tex` actually updated — that
  would be the first edit this audit makes to `paper/`, so I want it to be
  an explicit instruction, not an inference.

## What could not be reconstructed

- The script that generated `mediapipe_gt_corrected.txt` (500-sample
  official-MediaPipe-SDK run) is not present anywhere in the current working
  tree or `experiments/`.
- `distillation_learning_curve.csv` at repo root does not exist; only the
  differently-dated `experiments/v2_learning_curve_complete.csv` survives.
- The script behind `results_direct_sup_best.txt` / `results_direct_sup_TC.txt`
  was not located by filename; its logic is independently reproduced by
  `scripts/rerun_dual_space_eval.py`'s teacher-decoded-space path.
- Corrected (bug-fixed) re-runs of `mediapipe_gt_eval.py` and
  `mediapipe_vs_blazehand_eval.py` — not done yet, flagged above, not
  estimated.
- Any epoch-100 webcam screenshots beyond the 8 that survive on disk — not
  located, not reconstructed.

## Repo git context

Single-commit repository (`06705b8`); all dates above are filesystem mtimes
captured 2026-08-07, not git history.

## Total wall-clock time

| Step | Time |
|---|---|
| Original audit (Steps 0-5, first delivery) | ~45 min |
| Root-cause investigation (mmpose source read, `directsup_baseline.py` check, 50-sample diff measurement) | ~10 min |
| Corrected full rerun, `scripts/rerun_dual_space_eval.py` (2,727 samples) | 684.9s (~11.4 min) |
| Figure 6 / webcam addendum follow-up | ~10 min |
| **Total wall-clock for the full audit including this correction** | **~75 minutes** |
