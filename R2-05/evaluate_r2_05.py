#!/usr/bin/env python3
"""
R2-05/evaluate_r2_05.py -- Phase 4 evaluation for R2-05.

Scores the R2-05 parity-controlled Direct Supervision checkpoint against
Dist. V2-100 (epoch-matched, scientifically controlled comparison) and a
FRESH re-score of FT V2-150 (the paper's headline model) -- all on the
identical full 2,727-sample rhd_test.json GT-space protocol used
everywhere else in this audit:

  - Full RHD2D test split (dataset/rhd/annotations/rhd_test.json, 2,727)
  - Pipeline: LoadImage -> GetBBoxCenterScale -> TopdownAffine(256,256)
    -> PackPoseInputs(pack_transformed=True)
  - Reference: gt_instances.transformed_keypoints[0], clipped [0,255]
    (256x256 crop space -- NOT gt_instances.keypoints, the coordinate-
    space bug documented in R2-05/R2-05_config_parity.md)
  - PCK@0.2 threshold: fixed 51.2px (0.2*256), not bbox-relative
  - AUC: trapezoid over 20 thresholds on linspace(0, 128, 20), / 128
  - MPJPE: mean L2 in px. Det. Rate: hand_flag>=0.5, informational only.

Matches the protocol in directsup_gt_comprehensive_eval.py,
scripts/rerun_dual_space_eval.py, and independent_reverify.py.

None of the numbers below are copied from any existing results_*.txt --
every model here is loaded from its checkpoint and scored fresh in this
run, including FT V2-150 (per R2-05 Phase 4 requirement: "re-scored
fresh, not copied from the results table").

Usage:
    python R2-05/evaluate_r2_05.py

Writes:
    R2-05/results/phase4_eval_raw.csv           -- per-model summary + per-joint
    R2-05/results/phase4_eval_summary.txt       -- human-readable summary
"""

import csv
import os
import time

import numpy as np
import torch

from mmpose.datasets import Rhd2DDataset
from torch.utils.data import DataLoader
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope

init_default_scope('mmpose')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

import sys  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, 'student_model'))
from blazehand_landmark import BlazeHandLandmark  # noqa: E402

JOINT_NAMES = [
    'Wrist',
    'Thumb_MCP', 'Thumb_PIP', 'Thumb_DIP', 'Thumb_Tip',
    'Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_Tip',
    'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_Tip',
    'Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Ring_Tip',
    'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP', 'Pinky_Tip',
]

MODELS = [
    ('R2-05 Parity (Direct Sup., ep.100, v2match)',
     os.path.join(ROOT, 'R2-05', 'runs', 'parity_baseline', 'checkpoints', 'parity_final.pth')),
    ('R2-05 Parity (Direct Sup., best-monitor ep.90)',
     os.path.join(ROOT, 'R2-05', 'runs', 'parity_baseline', 'checkpoints', 'parity_best.pth')),
    ('Dist. V2-100 (epoch-matched control)',
     os.path.join(ROOT, 'checkpoints', 'distilled_v2_epoch_100.pth')),
    ('FT V2-150 (headline model, fresh re-score)',
     os.path.join(ROOT, 'checkpoints', 'distilled_v2_ft_epoch_150.pth')),
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(type='PackPoseInputs', pack_transformed=True),
]
ds = Rhd2DDataset(
    data_root=os.path.join(ROOT, 'dataset', 'rhd'),
    ann_file='annotations/rhd_test.json',
    pipeline=pipeline,
)
N = len(ds)
print(f'RHD2D test split: {N} samples (expected 2727)')
assert N == 2727, f'unexpected test-set size: {N}'

loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0,
                     collate_fn=pseudo_collate)

FIXED_THR = 0.2 * 256.0
AUC_THRESHOLDS = np.linspace(0, 0.5 * 256, 20)

students = {}
for name, ckpt in MODELS:
    assert os.path.exists(ckpt), f'checkpoint not found: {ckpt}'
    m = BlazeHandLandmark().to(device)
    sd = torch.load(ckpt, map_location=device)
    m.load_state_dict(sd)
    m.eval()
    students[name] = m
    sz = os.path.getsize(ckpt)
    print(f'  loaded {name:<45s} <- {os.path.relpath(ckpt, ROOT)}  ({sz:,} bytes)')

acc = {name: {'dists': [], 'correct': np.zeros(21), 'flags': []} for name, _ in MODELS}

t0 = time.time()
for b_idx, batch in enumerate(loader):
    img_tensor = torch.stack(batch['inputs']).to(device).float() / 255.0
    gt_crop = np.stack([
        np.clip(np.array(ds_.gt_instances.transformed_keypoints[0]), 0, 255)
        for ds_ in batch['data_samples']
    ]).astype(np.float32)

    with torch.no_grad():
        for name, model in students.items():
            flag, _, pred = model(img_tensor)
            sc = pred[:, :, :2].cpu().numpy() * 256.0
            d = np.sqrt(np.sum((sc - gt_crop) ** 2, axis=2))
            acc[name]['dists'].append(d)
            acc[name]['correct'] += (d <= FIXED_THR).sum(axis=0)
            acc[name]['flags'].extend(flag.cpu().numpy().tolist())

    done = min((b_idx + 1) * loader.batch_size, N)
    if done % 500 < loader.batch_size or done == N:
        print(f'  [{done:4d}/{N}]  {time.time()-t0:6.1f}s elapsed')

elapsed = time.time() - t0
print(f'\nEval done in {elapsed:.1f}s ({N} samples x {len(MODELS)} models)\n')

results = {}
for name, _ in MODELS:
    dists = np.concatenate(acc[name]['dists'], axis=0)
    correct = acc[name]['correct']
    flags = np.array(acc[name]['flags'])
    pck = float(correct.sum() / (N * 21))
    mpjpe = float(dists.mean())
    mse = float(np.mean(np.sum(dists ** 2, axis=1) / 21))
    auc_v = [(dists <= t).mean() for t in AUC_THRESHOLDS]
    auc = float(np.trapz(auc_v, AUC_THRESHOLDS / (0.5 * 256)))
    det = float((flags >= 0.5).mean() * 100)
    pck_pj = correct / N
    mpjpe_pj = dists.mean(axis=0)
    results[name] = dict(pck=pck, mpjpe=mpjpe, mse=mse, auc=auc, det=det,
                          pck_pj=pck_pj, mpjpe_pj=mpjpe_pj)
    print(f'{name:<45s} PCK={pck:.4f}  AUC={auc:.4f}  MPJPE={mpjpe:7.4f}px  Det={det:6.2f}%')

# ---- gaps, labeled -----------------------------------------------------
parity = results['R2-05 Parity (Direct Sup., ep.100, v2match)']
dist100 = results['Dist. V2-100 (epoch-matched control)']
ft150 = results['FT V2-150 (headline model, fresh re-score)']

gap_vs_dist100 = dist100['auc'] - parity['auc']
gap_vs_ft150 = ft150['auc'] - parity['auc']

print('\n' + '=' * 78)
print('AUC GAPS (parity-controlled, labeled per R2-05 Gate-2 instruction)')
print('=' * 78)
print(f'Parity vs Dist. V2-100 (epoch-matched, THE controlled comparison): '
      f'{gap_vs_dist100*100:+.2f}pp AUC')
print(f'Parity vs FT V2-150   (headline model, 150ep vs 100ep -- NOT epoch-matched): '
      f'{gap_vs_ft150*100:+.2f}pp AUC')
print('=' * 78)

# ---- write outputs -------------------------------------------------------
raw_csv = os.path.join(RESULTS_DIR, 'phase4_eval_raw.csv')
with open(raw_csv, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['model', 'joint', 'metric', 'value'])
    for name, m in results.items():
        for key, label in [('pck', 'PCK@0.2'), ('mpjpe', 'MPJPE'), ('auc', 'AUC'),
                            ('mse', 'MSE'), ('det', 'Det.Rate')]:
            w.writerow([name, 'ALL', label, f'{m[key]:.6f}'])
        for j, jn in enumerate(JOINT_NAMES):
            w.writerow([name, jn, 'PCK@0.2', f'{m["pck_pj"][j]:.6f}'])
            w.writerow([name, jn, 'MPJPE', f'{m["mpjpe_pj"][j]:.6f}'])
print(f'Wrote {raw_csv}')

summary_path = os.path.join(RESULTS_DIR, 'phase4_eval_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('R2-05 Phase 4 evaluation -- fresh, this session\n')
    f.write(f'Script: R2-05/evaluate_r2_05.py\n')
    f.write(f'Command: python R2-05/evaluate_r2_05.py\n')
    f.write(f'Dataset: dataset/rhd/annotations/rhd_test.json, full split, N={N}\n')
    f.write('Protocol: GT-space (gt_instances.transformed_keypoints, pack_transformed=True), '
            'fixed 51.2px PCK@0.2 threshold, AUC over linspace(0,128,20)/128\n')
    f.write(f'Wall-clock: {elapsed:.1f}s\n\n')
    for name, m in results.items():
        f.write(f'{name}\n')
        f.write(f'  PCK@0.2={m["pck"]:.4f}  AUC={m["auc"]:.4f}  MPJPE={m["mpjpe"]:.4f}px  '
                f'MSE={m["mse"]:.4f}px^2  Det.Rate={m["det"]:.2f}%\n\n')
    f.write(f'AUC gap, Parity vs Dist. V2-100 (epoch-matched, controlled): '
            f'{gap_vs_dist100*100:+.2f}pp\n')
    f.write(f'AUC gap, Parity vs FT V2-150 (headline, NOT epoch-matched): '
            f'{gap_vs_ft150*100:+.2f}pp\n')
print(f'Wrote {summary_path}')
