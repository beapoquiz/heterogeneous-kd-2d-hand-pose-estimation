"""
TEACHER EVAL — FILE B
Teacher vs Ground Truth using the MSRA DECODE PATH.
(output_heatmaps=True → pred_fields.heatmaps → codec.decode)

This is the EXACT same path used during distillation training in
distillation_v2.py and distillation_finetune.py.

Comparing File B results vs File A answers the question:
  "How much accuracy does the manual MSRA decode lose compared
   to the official decode?"

If File B ≈ File A  → no loss from MSRA decode path
If File B < File A  → there is real degradation from the manual decode

Run from project root:
    python teacher_eval_B.py
"""

import os
import numpy as np
import torch
from mmpose.apis import init_model
from mmpose.codecs import MSRAHeatmap
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
AUC_NORM  = 30.0
AUC_STEPS = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')
print('Loading teacher (MSRA decode path — output_heatmaps=True)...')
teacher = init_model(CONFIG, CKPT, device=device)

# Enable heatmap output — same as distillation_v2.py
teacher.cfg.model.test_cfg.output_heatmaps = True
if hasattr(teacher, 'head') and hasattr(teacher.head, 'test_cfg'):
    if isinstance(teacher.head.test_cfg, dict):
        teacher.head.test_cfg['output_heatmaps'] = True
    else:
        teacher.head.test_cfg.output_heatmaps = True
teacher.eval()

# Codec — same params as distillation_v2.py
codec = MSRAHeatmap(input_size=(256, 256), heatmap_size=(64, 64),
                    sigma=2, unbiased=True)

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
kp_dists_msra  = []   # MSRA decode path distances
kp_dists_std   = []   # standard decode distances (for direct comparison)
coord_diffs    = []   # absolute coordinate difference between both paths (px)
bbox_sizes     = []
img_epe_list   = []

for b_idx, batch in enumerate(loader):
    img_tensor = torch.stack(batch['inputs']).float() / 255.0

    with torch.no_grad():
        results = teacher.predict(img_tensor.to(device), batch['data_samples'])

    for res, ds_sample in zip(results, batch['data_samples']):
        # ── MSRA decode path ──────────────────────────────────────────────────
        hms = res.pred_fields.heatmaps
        if torch.is_tensor(hms):
            hms = hms.detach().cpu().numpy()
        dec = codec.decode(hms)
        if isinstance(dec, dict):
            coords = dec.get('keypoints', dec.get('pred_keypoints'))
            if isinstance(coords, list):
                coords = coords[0]
            pred_msra = np.array(coords).reshape(21, 2)
        else:
            pred_msra = np.array(dec[0]).reshape(21, 2)

        # ── Standard decode path (from pred_instances) ───────────────────────
        pred_std = np.array(res.pred_instances.keypoints[0]).reshape(21, 2)

        # ── Ground truth ──────────────────────────────────────────────────────
        gt      = ds_sample.gt_instances
        gt_kps  = np.array(gt.keypoints[0]).reshape(21, 2)
        gt_vis  = (np.array(gt.keypoints_visible[0])
                   if hasattr(gt, 'keypoints_visible') else np.ones(21))

        bbox      = np.array(gt.bboxes[0])
        bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])

        vis = gt_vis > 0
        if vis.sum() == 0:
            continue

        d_msra = np.linalg.norm(pred_msra - gt_kps, axis=1)[vis]
        d_std  = np.linalg.norm(pred_std  - gt_kps, axis=1)[vis]
        d_diff = np.linalg.norm(pred_msra - pred_std, axis=1)[vis]

        kp_dists_msra.extend(d_msra.tolist())
        kp_dists_std.extend(d_std.tolist())
        coord_diffs.extend(d_diff.tolist())
        bbox_sizes.extend([float(bbox_size)] * int(vis.sum()))
        img_epe_list.append(float(d_msra.mean()))

    print(f'  {min((b_idx+1)*BATCH, len(ds)):4d}/{len(ds)}...', end='\r')

print()

kp_dists_msra = np.array(kp_dists_msra)
kp_dists_std  = np.array(kp_dists_std)
coord_diffs   = np.array(coord_diffs)
bbox_sizes    = np.array(bbox_sizes)
img_epes      = np.array(img_epe_list)

thrs     = np.linspace(0, AUC_NORM, AUC_STEPS + 1)[1:]
thr_bbox = bbox_sizes * 0.2

# MSRA path metrics
epe_msra       = float(kp_dists_msra.mean())
pck_msra_bbox  = float((kp_dists_msra < thr_bbox).mean())
pck_msra_fixed = float((kp_dists_msra < 0.2 * 256).mean())
auc_msra       = float(np.mean([(kp_dists_msra < t).mean() for t in thrs]))

# Standard path metrics
epe_std       = float(kp_dists_std.mean())
pck_std_bbox  = float((kp_dists_std < thr_bbox).mean())
pck_std_fixed = float((kp_dists_std < 0.2 * 256).mean())
auc_std       = float(np.mean([(kp_dists_std < t).mean() for t in thrs]))

# Coordinate difference between the two paths
mean_diff = float(coord_diffs.mean())
max_diff  = float(coord_diffs.max())
zero_diff = float((coord_diffs < 0.01).mean())  # fraction that are identical

print()
print('=' * 60)
print('  TEACHER — FILE B  (MSRA decode path)')
print('=' * 60)
print(f'  {"Metric":<22}  {"MSRA decode":>12}  {"Standard":>12}  {"Diff":>8}')
print(f'  {"-"*22}  {"-"*12}  {"-"*12}  {"-"*8}')
print(f'  {"EPE (px)":<22}  {epe_msra:>12.4f}  {epe_std:>12.4f}  {epe_msra-epe_std:>+8.4f}')
print(f'  {"PCK@0.2 [bbox]":<22}  {pck_msra_bbox:>12.4f}  {pck_std_bbox:>12.4f}  {pck_msra_bbox-pck_std_bbox:>+8.4f}')
print(f'  {"PCK@0.2 [fixed]":<22}  {pck_msra_fixed:>12.4f}  {pck_std_fixed:>12.4f}  {pck_msra_fixed-pck_std_fixed:>+8.4f}')
print(f'  {"AUC [0-30px]":<22}  {auc_msra:>12.4f}  {auc_std:>12.4f}  {auc_msra-auc_std:>+8.4f}')
print('=' * 60)
print()
print('Direct coordinate comparison (MSRA coords vs Standard coords):')
print(f'  Mean coord difference : {mean_diff:.4f} px')
print(f'  Max  coord difference : {max_diff:.4f} px')
print(f'  Fraction identical    : {zero_diff:.4f}  (<0.01 px apart)')
print()

if mean_diff < 0.1:
    print('  ✓ MSRA decode and standard decode give effectively IDENTICAL')
    print('    keypoint coordinates. No accuracy loss from MSRA decode path.')
else:
    print(f'  ! Mean coordinate difference = {mean_diff:.3f} px.')
    print('    There is a small divergence between the two decode paths.')
    print('    This contributes to the difference in reported metrics.')

print('\nDone — File B.')
