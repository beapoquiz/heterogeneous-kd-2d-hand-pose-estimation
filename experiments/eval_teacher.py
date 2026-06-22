"""
Evaluate Teacher (HRNetV2-W18 + DARK) on the RHD test set.
Single pass at batch_size=32. Prints PCK@0.2, AUC, EPE and a
per-image EPE distribution so you can see why 0.992 ≠ pixel-perfect.

Usage (from project root):
    python eval_teacher.py
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
loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                    num_workers=0, collate_fn=pseudo_collate)
print(f'Test set : {len(ds)} samples  |  batch={BATCH}  |  {len(loader)} batches\n')

# ── single pass ───────────────────────────────────────────────────────────────
img_epe_list  = []   # mean EPE per image  (px)
img_pck_list  = []   # PCK@0.2 per image   (fraction)
img_thr_list  = []   # pixel threshold per image (bbox × 0.2)
all_kp_epe    = []   # every visible-keypoint distance (for global metrics)
all_kp_bbox   = []   # corresponding bbox size

for b_idx, batch in enumerate(loader):
    img_tensor = torch.stack(batch['inputs']).float() / 255.0

    with torch.no_grad():
        results = teacher.predict(img_tensor.to(device), batch['data_samples'])

    for res, ds_sample in zip(results, batch['data_samples']):
        # Keypoints are already decoded by the model's predict()
        pred_kps = res.pred_instances.keypoints[0].reshape(21, 2)

        gt      = ds_sample.gt_instances
        gt_kps  = np.array(gt.keypoints[0]).reshape(21, 2)
        gt_vis  = (np.array(gt.keypoints_visible[0])
                   if hasattr(gt, 'keypoints_visible') else np.ones(21))

        bbox      = np.array(gt.bboxes[0])          # [x1, y1, x2, y2]
        bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        thr_px    = bbox_size * 0.2

        vis = gt_vis > 0
        if vis.sum() == 0:
            continue

        dists = np.linalg.norm(pred_kps - gt_kps, axis=1)[vis]

        img_epe_list.append(float(dists.mean()))
        img_pck_list.append(float((dists < thr_px).mean()))
        img_thr_list.append(float(thr_px))
        all_kp_epe.extend(dists.tolist())
        all_kp_bbox.extend([float(bbox_size)] * int(vis.sum()))

    done = min((b_idx + 1) * BATCH, len(ds))
    print(f'  {done:4d}/{len(ds)} samples...', end='\r')

print()

img_epes = np.array(img_epe_list)
img_pcks = np.array(img_pck_list)
img_thrs = np.array(img_thr_list)
all_epe  = np.array(all_kp_epe)
all_bbox = np.array(all_kp_bbox)

# ── global metrics ─────────────────────────────────────────────────────────────
pck_02 = float((all_epe < all_bbox * 0.2).mean())
epe    = float(all_epe.mean())

# AUC: mean PCK across thresholds 0.01…0.20 (mirrors MMPose AUC metric)
thrs       = np.linspace(0, 0.2, 21)[1:]
auc_scores = [(all_epe < all_bbox * t).mean() for t in thrs]
auc        = float(np.mean(auc_scores))

print()
print('=' * 52)
print('  TEACHER  —  RHD test-set metrics')
print('=' * 52)
print(f'  PCK@0.2  : {pck_02:.4f}    (model zoo: 0.992)')
print(f'  AUC      : {auc:.4f}    (model zoo: 0.902)')
print(f'  EPE      : {epe:.2f} px   (model zoo: 2.21 px)')
print('=' * 52)
print()

# ── per-image EPE distribution ─────────────────────────────────────────────────
print('EPE distribution across images:')
bands = [(0,2,'excellent'),(2,5,'good'),(5,10,'ok'),(10,20,'poor'),(20,9999,'bad')]
for lo, hi, label in bands:
    n   = int(((img_epes >= lo) & (img_epes < hi)).sum())
    pct = 100 * n / len(img_epes)
    bar = '█' * int(pct / 2)
    hi_str = f'{hi}' if hi < 9999 else '∞'
    print(f'  [{lo:2d}–{hi_str:>3s}px] {label:9s}: {n:5d} imgs  ({pct:5.1f}%)  {bar}')

print()
thr_med = np.median(img_thrs)
print(f'PCK@0.2 pixel tolerance  (bbox × 0.2):')
print(f'  Median : {thr_med:.1f} px  |  '
      f'Range: {img_thrs.min():.1f}–{img_thrs.max():.1f} px')
print()
print(f'  ➜  A keypoint up to ~{thr_med:.0f} px from GT still counts as "correct".')
print(f'     That gap is why PCK@0.2 = 0.992 but individual frames can look off.')
print()

worst = np.argsort(img_epes)[-10:][::-1]
print('10 worst images by mean EPE:')
for rank, i in enumerate(worst, 1):
    print(f'  {rank:2d}. mean EPE={img_epes[i]:.1f}px  '
          f'PCK@0.2={img_pcks[i]:.2f}  thr={img_thrs[i]:.1f}px')

print('\nDone.')
