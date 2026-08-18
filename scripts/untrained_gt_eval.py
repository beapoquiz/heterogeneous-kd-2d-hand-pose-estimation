#!/usr/bin/env python3
"""
scripts/untrained_gt_eval.py

Fresh GT-space eval of a randomly-initialized (untrained) BlazeHandLandmark,
for Table V's "Untrained BlazeHandLandmark (random init.)" row. No source
file for that row existed anywhere in the repo (flagged in
results/provenance_manifest.csv row 16) -- this replaces it with a
traceable, full-2,727-sample run using the SAME corrected GT-space protocol
(PackPoseInputs(pack_transformed=True) + gt_instances.transformed_keypoints,
clipped to [0,255]) as directsup_gt_comprehensive_eval.py /
v2_GT_comprehensive_eval.py / distilled_v2_ft_ep150_gt_comprehensive_eval.py,
so it is directly comparable to those three rows.

No training involved -- BlazeHandLandmark() is constructed and evaluated
with its random initialization as-is (same pattern as the 'untrained' model
in three_way_student_comparison.py, but scored against true GT-space instead
of teacher-decoded-space, and on the full test set instead of 500 samples).

Usage:
    python scripts/untrained_gt_eval.py

Writes:
    results/results_untrained_GT.txt
"""

import argparse
import os
import sys

import numpy as np
import torch
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmpose.datasets import Rhd2DDataset
from torch.utils.data import DataLoader

init_default_scope('mmpose')

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', type=str, default='results_untrained_GT.txt',
                 help='output filename under results/')
args = ap.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT, 'student_model'))
from blazehand_landmark import BlazeHandLandmark  # noqa: E402

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

torch.manual_seed(args.seed)
student = BlazeHandLandmark().to(device)
student.eval()
print(f'Model: BlazeHandLandmark, random initialization (no checkpoint loaded, seed={args.seed})')

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
loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=pseudo_collate)

N = len(ds)
threshold = 0.2 * 256
auc_thresholds = np.linspace(0, 0.5 * 256, 20)
correct = np.zeros(21)
dists = []
flags = []

print('Untrained BlazeHandLandmark — GT Reference Evaluation')
print(f'Full RHD test set: {N} samples')
print()

for idx, batch in enumerate(loader):
    img_tensor = torch.stack(batch['inputs']).float() / 255.0

    gt_raw = batch['data_samples'][0].gt_instances.transformed_keypoints[0]
    tc = np.clip(gt_raw, 0, 255).astype(np.float32)

    with torch.no_grad():
        flag, _, pred = student(img_tensor.to(device))
        sc = pred[0, :, :2].cpu().numpy() * 256
        flags.append(flag.item())

    dist = np.sqrt(np.sum((sc - tc) ** 2, axis=1))
    dists.append(dist)
    correct += (dist <= threshold).astype(float)

    if (idx + 1) % 500 == 0:
        print(f'  [{idx+1}/{N}] Running PCK@0.2: {correct.sum()/((idx+1)*21):.4f}')

dists = np.array(dists)
flags = np.array(flags)
pck = correct.sum() / (N * 21)
mpjpe = dists.mean()
mse = np.mean(np.sum(dists ** 2, axis=1) / 21)
det = (flags >= 0.5).mean() * 100
auc_v = [(dists <= t).mean() for t in auc_thresholds]
auc = np.trapz(auc_v, auc_thresholds / (0.5 * 256))

joint_names = ['Wrist',
               'Thumb_MCP', 'Thumb_PIP', 'Thumb_DIP', 'Thumb_Tip',
               'Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_Tip',
               'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_Tip',
               'Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Ring_Tip',
               'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP', 'Pinky_Tip']

print()
print('=' * 60)
print('UNTRAINED BLAZEHANDLANDMARK (random init.) — GT REFERENCE EVALUATION')
print(f'Dataset: RHD test split ({N} samples)')
print('Reference: gt_instances.transformed_keypoints (256x256 crop space)')
print('=' * 60)
print(f'PCK@0.2      : {pck:.4f} ({pck*100:.2f}%)')
print(f'MPJPE        : {mpjpe:.4f} px')
print(f'MSE          : {mse:.4f} px²')
print(f'AUC          : {auc:.4f}')
print(f'Det. Rate    : {det:.2f}%')
print()
pck_per_joint = correct / N
mpjpe_per_joint = dists.mean(axis=0)
print(f'{"Joint":<15} {"PCK@0.2":>10} {"MPJPE (px)":>12}')
print('-' * 40)
for i, name in enumerate(joint_names):
    print(f'{name:<15} {pck_per_joint[i]:>10.4f} {mpjpe_per_joint[i]:>12.4f}')
print('=' * 60)

out_path = os.path.join(ROOT, 'results', args.out)
with open(out_path, 'w') as f:
    f.write('Untrained BlazeHandLandmark (random init.) — GT Reference\n')
    f.write(f'Dataset: RHD test split ({N} samples)\n')
    f.write('Reference: gt_instances.transformed_keypoints, clipped [0,255]\n\n')
    f.write(f'PCK@0.2:    {pck:.4f}\n')
    f.write(f'MPJPE:      {mpjpe:.4f} px\n')
    f.write(f'MSE:        {mse:.4f} px2\n')
    f.write(f'AUC:        {auc:.4f}\n')
    f.write(f'Det Rate:   {det:.2f}%\n\n')
    f.write(f'{"Joint":<15} {"PCK@0.2":>10} {"MPJPE":>12}\n')
    f.write('-' * 40 + '\n')
    for i, name in enumerate(joint_names):
        f.write(f'{name:<15} {pck_per_joint[i]:>10.4f} {mpjpe_per_joint[i]:>12.4f}\n')

print(f'Saved: {out_path}')
