#!/usr/bin/env python3
"""
R2-05/measure_coordinate_offset.py -- fresh measurement (this audit, not
reused from any prior write-up) of the mean per-joint L2 gap between
gt_instances.keypoints (original ~320x320 RHD image space) and
gt_instances.transformed_keypoints (256x256 crop space), on the full
2,727-sample rhd_test.json split. Confirms the magnitude of the
coordinate-space bug that directsup_baseline.py's training target had,
and that directsup_parity_v2.py's PackPoseInputs(pack_transformed=True)
fix actually changes something (not a no-op).

No GPU, no model -- pure dataloading + annotation arithmetic.

Usage:
    python R2-05/measure_coordinate_offset.py

Writes:
    R2-05/coordinate_offset_measurement.txt
"""

import os
import time

import numpy as np

from mmpose.datasets import Rhd2DDataset
from mmengine.registry import init_default_scope

init_default_scope('mmpose')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ROOT, 'dataset', 'rhd')

pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(type='PackPoseInputs', pack_transformed=True),
]

ds = Rhd2DDataset(
    data_root=DATA_ROOT,
    ann_file='annotations/rhd_test.json',
    pipeline=pipeline,
)
N = len(ds)
print(f'rhd_test.json: {N} samples')

t0 = time.time()
gaps = []
orig_means = []
warped_means = []
for i in range(N):
    s = ds[i]
    gi = s['data_samples'].gt_instances
    orig = np.array(gi.keypoints[0])              # original ~320x320 space
    warped = np.array(gi.transformed_keypoints[0])  # 256x256 crop space
    gap = np.linalg.norm(orig - warped, axis=1)     # (21,) per-joint L2, px
    gaps.append(gap)
    orig_means.append(orig.mean(axis=0))
    warped_means.append(warped.mean(axis=0))
    if (i + 1) % 500 == 0:
        print(f'  [{i+1}/{N}]  {time.time()-t0:.1f}s elapsed')

gaps = np.concatenate(gaps)  # (N*21,)
orig_means = np.array(orig_means)
warped_means = np.array(warped_means)

elapsed = time.time() - t0
report = []
report.append(f'Coordinate-space offset measurement (fresh, this session)')
report.append(f'Script: R2-05/measure_coordinate_offset.py')
report.append(f'Dataset: dataset/rhd/annotations/rhd_test.json, full split, N={N} samples')
report.append(f'Fields compared: gt_instances.keypoints (orig. image space) vs.')
report.append(f'                 gt_instances.transformed_keypoints (256x256 crop space)')
report.append(f'Wall-clock: {elapsed:.1f}s')
report.append('')
report.append(f'Per-joint L2 gap, {gaps.shape[0]} joint-instances ({N} samples x 21 joints):')
report.append(f'  mean   = {gaps.mean():.2f} px')
report.append(f'  median = {np.median(gaps):.2f} px')
report.append(f'  std    = {gaps.std():.2f} px')
report.append(f'  min    = {gaps.min():.2f} px')
report.append(f'  max    = {gaps.max():.2f} px')
report.append('')
report.append(f'Mean sample-level centroid, orig. space:      {orig_means.mean(axis=0).round(1)}')
report.append(f'Mean sample-level centroid, crop (256) space: {warped_means.mean(axis=0).round(1)}')
report.append('')
report.append('Interpretation: this is the per-joint distance directsup_baseline.py\'s')
report.append('training loss was implicitly asked to close between a 256-crop-space')
report.append('prediction and an original-image-space target -- i.e., roughly this many')
report.append('px of the ~40-60px MPJPE gap reported for that checkpoint could be')
report.append('coordinate-space noise rather than a real supervision-signal deficit.')

text = '\n'.join(report)
print()
print(text)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coordinate_offset_measurement.txt')
with open(out_path, 'w') as f:
    f.write(text + '\n')
print(f'\nSaved: {out_path}')
