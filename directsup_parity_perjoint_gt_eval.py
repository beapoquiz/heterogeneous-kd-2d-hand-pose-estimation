#!/usr/bin/env python3
"""
directsup_parity_perjoint_gt_eval.py

Per-joint GT-space evaluation of the R2-05 parity-controlled Direct
Supervision checkpoint (R2-05/runs/parity_baseline/checkpoints/parity_final.pth,
protocol v2match, ep.100), for Table V (per-joint PCK@0.2) of the paper.

Protocol matches experiments/v2_GT_comprehensive_eval.py and
experiments/distilled_v2_ft_ep150_gt_comprehensive_eval.py exactly:
  - Full RHD2D test split (dataset/rhd/annotations/rhd_test.json, 2,727 samples)
  - Pipeline: LoadImage -> GetBBoxCenterScale -> TopdownAffine(256,256)
    -> PackPoseInputs(pack_transformed=True)
  - Reference: gt_instances.transformed_keypoints[0] (256x256 crop space),
    clipped to [0, 255]. NOT gt_instances.keypoints.
  - Fixed PCK@0.2 threshold = 0.2*256 = 51.2px (not bbox-relative)
  - batch_size=1, same loader settings as the two experiments/ scripts above

Corrected joint-index mapping (found and verified earlier this session by
measuring wrist-to-keypoint pixel distance across 70 samples in both
rhd_train.json and rhd_test.json: index 1 averages ~64px from the wrist
(near the fingertip), index 4 averages ~18px (near the palm)):

    Within each finger's 4-index block, the raw annotation order is
    Tip, DIP, PIP, MCP -- i.e. index 1 = Tip, index 4 = MCP.

This is the OPPOSITE of the ascending MCP->PIP->DIP->Tip mapping hardcoded
in directsup_gt_comprehensive_eval.py, experiments/v2_GT_comprehensive_eval.py,
experiments/distilled_v2_ft_ep150_gt_comprehensive_eval.py, and
R2-05/evaluate_r2_05.py (all of them build joint_names as
['Wrist','Thumb_MCP','Thumb_PIP','Thumb_DIP','Thumb_Tip', ...], assigning
labels in ascending-index order). Those scripts' per-joint PCK NUMBERS are
still valid (the math never used the names), but any table or comparison
that reads across finger positions by name (Tip vs MCP, etc.) using their
printed/written labels is mislabeled. This includes
results/table_iv_vii_viii_gt_space_FINAL.csv, which was generated with the
same hardcoded (incorrect) joint_names list.

Usage:
    python directsup_parity_perjoint_gt_eval.py

Writes:
    results/directsup_parity_perjoint_gt_eval.csv     -- model,joint,metric,value
    results/directsup_parity_perjoint_gt_eval_summary.txt
"""
import csv
import os

import numpy as np
import torch
from mmpose.datasets import Rhd2DDataset
from torch.utils.data import DataLoader
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope

init_default_scope('mmpose')

import sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, 'student_model'))
from blazehand_landmark import BlazeHandLandmark

RESULTS_DIR = os.path.join(ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

CKPT = os.path.join(ROOT, 'R2-05', 'runs', 'parity_baseline', 'checkpoints', 'parity_final.pth')
MODEL_LABEL = 'Direct Sup. (parity, ep.100, v2match)'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

assert os.path.exists(CKPT), f'checkpoint not found: {CKPT}'
student = BlazeHandLandmark().to(device)
student.load_state_dict(torch.load(CKPT, map_location=device))
student.eval()
print(f'Loaded: {os.path.relpath(CKPT, ROOT)}')

pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256, 256)),
    # pack_transformed=True is required: TopdownAffine writes the
    # crop-warped keypoints to results['transformed_keypoints'] and
    # leaves results['keypoints'] as the original, un-warped image-space
    # annotation. Without this flag, gt_instances.keypoints is NOT in the
    # same 256x256 crop space as the student's predictions below.
    dict(type='PackPoseInputs', pack_transformed=True)
]
ds = Rhd2DDataset(
    data_root=os.path.join(ROOT, 'dataset', 'rhd'),
    ann_file='annotations/rhd_test.json',
    pipeline=pipeline
)
loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=pseudo_collate)

N = len(ds)
print(f'RHD2D test split: {N} samples (expected 2727)')
assert N == 2727, f'unexpected test-set size: {N}'

threshold = 0.2 * 256
auc_thresholds = np.linspace(0, 0.5 * 256, 20)
correct = np.zeros(21)
dists = []
flags = []

print('Direct Sup. (parity, ep.100, v2match) — Per-Joint GT-Reference Evaluation')
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

# Corrected mapping: within each finger's 4-index block, raw annotation
# order is Tip, DIP, PIP, MCP (index 1 = Tip, index 4 = MCP) -- verified
# by wrist-distance measurement (see module docstring). This is the
# reverse of the ascending MCP->PIP->DIP->Tip order used elsewhere.
JOINT_NAMES = ['Wrist',
               'Thumb_Tip', 'Thumb_DIP', 'Thumb_PIP', 'Thumb_MCP',
               'Index_Tip', 'Index_DIP', 'Index_PIP', 'Index_MCP',
               'Middle_Tip', 'Middle_DIP', 'Middle_PIP', 'Middle_MCP',
               'Ring_Tip', 'Ring_DIP', 'Ring_PIP', 'Ring_MCP',
               'Pinky_Tip', 'Pinky_DIP', 'Pinky_PIP', 'Pinky_MCP']

pck_per_joint = correct / N
mpjpe_per_joint = dists.mean(axis=0)

print()
print('=' * 60)
print('DIRECT SUP. (PARITY, EP.100, V2MATCH) — GT REFERENCE EVALUATION')
print(f'Dataset: RHD test split ({N} samples)')
print('Reference: GT transformed_keypoints (256x256 crop space), clipped [0,255]')
print('=' * 60)
print(f'PCK@0.2      : {pck:.6f} ({pck*100:.2f}%)')
print(f'MPJPE        : {mpjpe:.6f} px')
print(f'MSE          : {mse:.6f} px^2')
print(f'AUC          : {auc:.6f}')
print(f'Det. Rate    : {det:.2f}%')
print()
print(f'{"Joint":<15} {"PCK@0.2":>10} {"MPJPE (px)":>12}')
print('-' * 40)
for i, name in enumerate(JOINT_NAMES):
    print(f'{name:<15} {pck_per_joint[i]:>10.4f} {mpjpe_per_joint[i]:>12.4f}')
print('=' * 60)

# ---- write CSV, same (model,joint,metric,value) format as
# results/table_iv_vii_viii_gt_space_FINAL.csv --------------------------
csv_path = os.path.join(RESULTS_DIR, 'directsup_parity_perjoint_gt_eval.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['model', 'joint', 'metric', 'value'])
    w.writerow([MODEL_LABEL, 'ALL', 'PCK@0.2', f'{pck:.6f}'])
    w.writerow([MODEL_LABEL, 'ALL', 'MPJPE', f'{mpjpe:.6f}'])
    w.writerow([MODEL_LABEL, 'ALL', 'AUC', f'{auc:.6f}'])
    w.writerow([MODEL_LABEL, 'ALL', 'MSE', f'{mse:.6f}'])
    w.writerow([MODEL_LABEL, 'ALL', 'Det.Rate', f'{det:.6f}'])
    for i, name in enumerate(JOINT_NAMES):
        w.writerow([MODEL_LABEL, name, 'PCK@0.2', f'{pck_per_joint[i]:.6f}'])
        w.writerow([MODEL_LABEL, name, 'MPJPE', f'{mpjpe_per_joint[i]:.6f}'])
print(f'Wrote {csv_path}')

summary_path = os.path.join(RESULTS_DIR, 'directsup_parity_perjoint_gt_eval_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('Direct Sup. (parity, ep.100, v2match) -- Per-Joint GT Reference Evaluation\n')
    f.write('Checkpoint: R2-05/runs/parity_baseline/checkpoints/parity_final.pth\n')
    f.write(f'Dataset: dataset/rhd/annotations/rhd_test.json, full split, N={N}\n')
    f.write('Protocol: GT-space (gt_instances.transformed_keypoints, pack_transformed=True), '
            'fixed 51.2px PCK@0.2 threshold, AUC over linspace(0,128,20)/128. '
            'Matches experiments/v2_GT_comprehensive_eval.py and '
            'experiments/distilled_v2_ft_ep150_gt_comprehensive_eval.py.\n')
    f.write('Joint mapping: corrected (index 1=Tip .. index 4=MCP per finger block); '
            'see module docstring.\n\n')
    f.write(f'PCK@0.2:    {pck:.6f}\n')
    f.write(f'MPJPE:      {mpjpe:.6f} px\n')
    f.write(f'MSE:        {mse:.6f} px2\n')
    f.write(f'AUC:        {auc:.6f}\n')
    f.write(f'Det Rate:   {det:.2f}%\n\n')
    f.write(f'{"Joint":<15} {"PCK@0.2":>10} {"MPJPE":>12}\n')
    f.write('-' * 40 + '\n')
    for i, name in enumerate(JOINT_NAMES):
        f.write(f'{name:<15} {pck_per_joint[i]:>10.6f} {mpjpe_per_joint[i]:>12.4f}\n')
print(f'Wrote {summary_path}')
