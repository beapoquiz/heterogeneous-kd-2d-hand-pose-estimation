"""
TEACHER EVAL — FILE C
"What did the distillation training metric actually measure?"

This file explains and quantifies the THREE differences between
the PCK@0.2 reported during training (distillation_v2.py) and
the official PCK@0.2 from tools/test.py:

  Difference 1 — Reference:
    Training    : student vs TEACHER  (teacher as pseudo-GT)
    Official    : model vs GROUND TRUTH

  Difference 2 — Threshold:
    Training    : fixed 0.2 × 256 = 51.2 px  (regardless of hand size)
    Official    : 0.2 × bbox_size             (scales with hand size)

  Difference 3 — Sample count:
    Training    : 200 samples (first 200 from test loader)
    Official    : all 2727 samples

This file uses the teacher itself as a "perfect reference" to isolate
the effect of differences 2 and 3 WITHOUT needing the student model.
It answers: "Even if the student were a perfect copy of the teacher,
how much would the training metric over/under-report accuracy?"

Run from project root:
    python teacher_eval_C.py
"""

import os
import json
import numpy as np
import torch
from mmpose.apis import init_model
from mmpose.datasets import Rhd2DDataset
from torch.utils.data import DataLoader
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope

init_default_scope('mmpose')

BASE   = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(BASE, 'mmpose/configs/hand_2d_keypoint/topdown_heatmap/'
                      'rhd2d/td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256.py')
CKPT   = os.path.join(BASE, 'checkpoints',
                      'hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')
print('Loading teacher...')
teacher = init_model(CONFIG, CKPT, device=device)
teacher.eval()

pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(type='PackPoseInputs'),
]
ds = Rhd2DDataset(
    data_root=os.path.join(BASE, 'dataset', 'rhd'),
    ann_file='annotations/rhd_test.json',
    pipeline=pipeline,
)
loader = DataLoader(ds, batch_size=32, shuffle=False,
                    num_workers=0, collate_fn=pseudo_collate)
print(f'Test set : {len(ds)} samples\n')

# ── collect per-keypoint info ──────────────────────────────────────────────────
kp_dists_vs_gt    = []   # teacher vs ground truth
bbox_sizes_all    = []
sample_indices    = []   # which sample index each keypoint belongs to

sample_count = 0
for b_idx, batch in enumerate(loader):
    img_tensor = torch.stack(batch['inputs']).float() / 255.0

    with torch.no_grad():
        results = teacher.predict(img_tensor.to(device), batch['data_samples'])

    for res, ds_sample in zip(results, batch['data_samples']):
        pred_kps = np.array(res.pred_instances.keypoints[0]).reshape(21, 2)

        gt     = ds_sample.gt_instances
        gt_kps = np.array(gt.keypoints[0]).reshape(21, 2)
        gt_vis = (np.array(gt.keypoints_visible[0])
                  if hasattr(gt, 'keypoints_visible') else np.ones(21))

        bbox      = np.array(gt.bboxes[0])
        bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])

        vis = gt_vis > 0
        if vis.sum() == 0:
            continue

        dists = np.linalg.norm(pred_kps - gt_kps, axis=1)
        for k in range(21):
            if vis[k]:
                kp_dists_vs_gt.append(dists[k])
                bbox_sizes_all.append(float(bbox_size))
                sample_indices.append(sample_count)

        sample_count += 1

    print(f'  {min((b_idx+1)*32, len(ds)):4d}/{len(ds)}...', end='\r')

print()

kp_dists   = np.array(kp_dists_vs_gt)
bbox_sizes = np.array(bbox_sizes_all)
s_indices  = np.array(sample_indices)

FIXED_THR = 0.2 * 256   # 51.2 px — what distillation_v2.py used
TOTAL_KP  = len(kp_dists)

# ── Effect of Difference 2: threshold choice ───────────────────────────────────
pck_fixed_allsamples = float((kp_dists < FIXED_THR).mean())
pck_bbox_allsamples  = float((kp_dists < bbox_sizes * 0.2).mean())

thr_median = float(np.median(bbox_sizes * 0.2))
thr_mean   = float(np.mean(bbox_sizes * 0.2))

# ── Effect of Difference 3: sample count ──────────────────────────────────────
# First 200 unique sample indices
first200_mask = np.isin(s_indices, np.arange(200))
pck_fixed_200 = float((kp_dists[first200_mask] < FIXED_THR).mean())
pck_bbox_200  = float((kp_dists[first200_mask] < bbox_sizes[first200_mask] * 0.2).mean())
n_kp_200      = int(first200_mask.sum())

# ── Official confirmed result ──────────────────────────────────────────────────
PCK_OFFICIAL = 0.991763   # from running python mmpose/tools/test.py (2026-05-11)
AUC_OFFICIAL = 0.902255
EPE_OFFICIAL = 2.182650

print()
print('=' * 65)
print('  TEACHER — FILE C  (Training metric vs Official metric)')
print('=' * 65)
print()
print('─── DIFFERENCE 1: What is being measured ───────────────────────')
print('  Training metric: student vs TEACHER (teacher = pseudo ground truth)')
print('  Official metric: model vs TRUE GROUND TRUTH annotations')
print('  ⟹  Training PCK measures "student-teacher agreement",')
print('      NOT "how accurate is the model vs GT".')
print()
print('─── DIFFERENCE 2: Threshold normalisation ───────────────────────')
print(f'  Training threshold : FIXED  = 0.2 × 256   = {FIXED_THR:.1f} px')
print(f'  Official threshold : BBOX   = 0.2 × bbox  = ~{thr_mean:.1f} px (mean), '
      f'~{thr_median:.1f} px (median)')
print()
print(f'  Teacher PCK@0.2 [fixed  51.2px] on ALL 2727 samples : {pck_fixed_allsamples:.4f}')
print(f'  Teacher PCK@0.2 [bbox-norm]     on ALL 2727 samples : {pck_bbox_allsamples:.4f}')
print(f'  Official result (tools/test.py)                      : {PCK_OFFICIAL:.4f}')
print()
inflation = pck_fixed_allsamples - pck_bbox_allsamples
print(f'  Inflation from fixed threshold: {inflation:+.4f}  '
      f'({inflation*100:.2f} percentage points)')
print()
print('─── DIFFERENCE 3: Sample count (200 vs 2727) ───────────────────')
print(f'  Keypoints evaluated in training validation ({n_kp_200} keypoints, 200 samples):')
print(f'    PCK@0.2 [fixed]     : {pck_fixed_200:.4f}')
print(f'    PCK@0.2 [bbox-norm] : {pck_bbox_200:.4f}')
print(f'  vs full 2727 samples  :')
print(f'    PCK@0.2 [fixed]     : {pck_fixed_allsamples:.4f}')
print(f'    PCK@0.2 [bbox-norm] : {pck_bbox_allsamples:.4f}')
print()
print('─── COMBINED EFFECT ─────────────────────────────────────────────')
print(f'  Official tool result  PCK@0.2 = {PCK_OFFICIAL:.4f}  (bbox-norm, all 2727)')
print(f'  Training-style metric PCK@0.2 = {pck_fixed_200:.4f}  (fixed 51.2px, 200 samples)')
print(f'  Gap                           = {pck_fixed_200 - PCK_OFFICIAL:+.4f}')
print()
print('─── IMPLICATIONS FOR THESIS (Heterogeneous KD) ──────────────────')
print()
print('  1. METRIC INCONSISTENCY')
print('     The "PCK@0.2" printed during distillation training is NOT')
print('     the same metric as the official PCK@0.2. The training metric')
print('     is inflated by both the fixed threshold and smaller sample size.')
print()
print('  2. SURROGATE SUPERVISION SIGNAL')
print('     The student learned to mimic teacher coordinates, not ground')
print('     truth. In heterogeneous KD, the teacher is the only supervisor.')
print('     If the teacher makes errors (~0.8% keypoints wrong), those')
print('     errors become the student\'s training targets.')
print()
print('  3. INFORMATION BOTTLENECK')
print('     The teacher\'s heatmap encodes spatial uncertainty (a probability')
print('     distribution over 64×64 locations). The distillation pipeline')
print('     compressed this to a single (x,y) coordinate via argmax decode.')
print('     Uncertainty information is lost — the student only sees the')
print('     teacher\'s best guess, not its confidence.')
print()
print('  4. THRESHOLD SENSITIVITY IN EVALUATION')
print(f'     The fixed threshold (51.2px) is {FIXED_THR/thr_mean:.2f}× the mean bbox-normalised')
print(f'     threshold ({thr_mean:.1f}px). Comparisons between models evaluated')
print('     with different threshold conventions cannot be made directly.')
print()
print('  5. PERFORMANCE CEILING')
print(f'     Teacher accuracy = {PCK_OFFICIAL:.4f}. The student can at best match')
print('     this ceiling; it cannot exceed the teacher\'s quality because')
print('     the teacher\'s decoded coordinates ARE the training labels.')

print()
print('─── OFFICIAL CONFIRMED METRICS (tools/test.py, 2026-05-11) ─────')
print(f'  PCK@0.2 : {PCK_OFFICIAL}')
print(f'  AUC     : {AUC_OFFICIAL}')
print(f'  EPE     : {EPE_OFFICIAL} px')
print('─' * 65)
print('\nDone — File C.')
