"""
TEACHER EVAL — FILE A
Teacher vs Ground Truth using the STANDARD decode path.
(pred_instances.keypoints — the same path tools/test.py uses)

Metrics computed:
  • EPE         : mean Euclidean distance per keypoint (px)
  • PCK@0.2     : bbox-normalised  (official MMPose metric)
  • PCK@0.2 fix : fixed 0.2×256 = 51.2 px threshold (what distillation used)
  • AUC         : area under PCK curve, 0–30 px, 20 steps (MMPose AUC metric)

Expected output (should match tools/test.py):
  PCK@0.2 ≈ 0.9918  |  AUC ≈ 0.9023  |  EPE ≈ 2.18 px

Run from project root:
    python teacher_eval_A.py
"""

import os
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
BATCH  = 32
AUC_NORM   = 30.0   # MMPose default: AUC normalised over 0–30 px
AUC_STEPS  = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')
print('Loading teacher (standard decode path — no output_heatmaps)...')
teacher = init_model(CONFIG, CKPT, device=device)
# Do NOT set output_heatmaps — use internal decode (same as tools/test.py)
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
loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                    num_workers=0, collate_fn=pseudo_collate)
print(f'Test set : {len(ds)} samples | batch={BATCH} | {len(loader)} batches\n')

# ── single pass ────────────────────────────────────────────────────────────────
kp_dists      = []   # per-visible-keypoint Euclidean distance (px)
bbox_sizes    = []   # corresponding bbox max(w,h) for the image
img_epe_list  = []

for b_idx, batch in enumerate(loader):
    img_tensor = torch.stack(batch['inputs']).float() / 255.0

    with torch.no_grad():
        results = teacher.predict(img_tensor.to(device), batch['data_samples'])

    for res, ds_sample in zip(results, batch['data_samples']):
        # Standard decoded keypoints — already in 256×256 crop space
        pred_kps = np.array(res.pred_instances.keypoints[0]).reshape(21, 2)

        gt      = ds_sample.gt_instances
        gt_kps  = np.array(gt.keypoints[0]).reshape(21, 2)
        gt_vis  = (np.array(gt.keypoints_visible[0])
                   if hasattr(gt, 'keypoints_visible') else np.ones(21))

        bbox      = np.array(gt.bboxes[0])          # [x1,y1,x2,y2] in crop space
        bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])

        vis = gt_vis > 0
        if vis.sum() == 0:
            continue

        dists = np.linalg.norm(pred_kps - gt_kps, axis=1)[vis]
        kp_dists.extend(dists.tolist())
        bbox_sizes.extend([float(bbox_size)] * int(vis.sum()))
        img_epe_list.append(float(dists.mean()))

    print(f'  {min((b_idx+1)*BATCH, len(ds)):4d}/{len(ds)}...', end='\r')

print()

kp_dists   = np.array(kp_dists)
bbox_sizes = np.array(bbox_sizes)
img_epes   = np.array(img_epe_list)

# ── metrics ────────────────────────────────────────────────────────────────────
epe       = float(kp_dists.mean())
pck_bbox  = float((kp_dists < bbox_sizes * 0.2).mean())
pck_fixed = float((kp_dists < 0.2 * 256).mean())

# AUC: MMPose uses 0–30 px, 20 equally-spaced thresholds
thrs      = np.linspace(0, AUC_NORM, AUC_STEPS + 1)[1:]
auc       = float(np.mean([(kp_dists < t).mean() for t in thrs]))

print()
print('=' * 58)
print('  TEACHER — FILE A  (standard / pred_instances.keypoints)')
print('=' * 58)
print(f'  EPE            : {epe:.4f} px   (official: ~2.18 px)')
print(f'  PCK@0.2 [bbox] : {pck_bbox:.4f}      (official: ~0.9918)')
print(f'  PCK@0.2 [fixed]: {pck_fixed:.4f}      (0.2×256=51.2 px threshold)')
print(f'  AUC  [0–30 px] : {auc:.4f}      (official: ~0.9023)')
print('=' * 58)
print()
print('PCK@0.2 [bbox] bbox-size distribution:')
print(f'  Median bbox_size : {np.median(bbox_sizes):.1f} px')
print(f'  Median threshold : {np.median(bbox_sizes * 0.2):.1f} px')
print(f'  Fixed  threshold : {0.2*256:.1f} px  ← what distillation_v2.py used')
print()

# Per-image EPE histogram
print('Per-image EPE distribution:')
for lo, hi in [(0,2),(2,5),(5,10),(10,20),(20,9999)]:
    n   = int(((img_epes >= lo) & (img_epes < hi)).sum())
    pct = 100 * n / len(img_epes)
    hi_str = f'{hi}' if hi < 9999 else '∞'
    print(f'  [{lo:2d}–{hi_str:>4s} px]: {n:5d} imgs  ({pct:5.1f}%)  '
          + '█' * int(pct / 2))

print('\nDone — File A.')
