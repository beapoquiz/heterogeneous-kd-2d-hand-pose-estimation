#!/usr/bin/env python3
"""
scripts/rerun_dual_space_eval.py

Audit re-run for the APSIPA camera-ready results/ package (paper #1571328446).

Re-evaluates Direct Sup. (ep.25), Dist. V2-100, and FT V2-150 on the FULL
2,727-sample RHD2D test set, scoring each model's predictions against BOTH
candidate reference targets in a single pass:

  GT-space             : batch['data_samples'][0].gt_instances.transformed_keypoints
                          (raw RHD2D annotations, warped into 256x256 crop
                          space by the TopdownAffine pipeline step; requires
                          PackPoseInputs(pack_transformed=True) -- the
                          untransformed .keypoints field is left in
                          original-image space by TopdownAffine and is NOT
                          comparable to student/teacher predictions)
  teacher-decoded-space : HRNetV2-W18+DARK heatmap, decoded via MSRAHeatmap
                          (same codec/config as distillation_v2.py's
                          teacher_targets computation), in 256x256 space

Both targets are computed from the SAME loaded sample and the SAME teacher
forward pass, then every one of the three students runs on the SAME image
tensor -- so the two spaces cannot silently diverge from re-running a model
twice with different code paths, and all three models see identical inputs.

This does not change methodology versus the original ad hoc eval scripts
(directsup_gt_comprehensive_eval.py, experiments/v2_ep100_comprehensive_eval.py,
experiments/v2_GT_comprehensive_eval.py, experiments/v2_ft_ep150_comprehensive_eval.py,
experiments/distilled_v2_ft_ep150_gt_comprehensive_eval.py) beyond making the
GT-clipping and teacher-coord-clipping conventions consistent across all three
models (both are np.clip(..., 0, 255) here; the original scripts disagreed on
this in ways noted in results/provenance_manifest.csv).

Usage:
    python scripts/rerun_dual_space_eval.py [--samples N]

Writes:
    results/table_viii_both_spaces.csv
    results/per_joint_winloss_gt_space.csv
    results/per_joint_winloss_teacher_space.csv
    results/rerun_dual_space_summary.txt
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT, 'student_model'))
sys.path.insert(0, ROOT)

from mmpose.apis import init_model                      # noqa: E402
from mmpose.codecs import MSRAHeatmap                    # noqa: E402
from mmpose.datasets import Rhd2DDataset                 # noqa: E402
from torch.utils.data import DataLoader                  # noqa: E402
from mmengine.dataset import pseudo_collate               # noqa: E402
from mmengine.registry import init_default_scope          # noqa: E402
from blazehand_landmark import BlazeHandLandmark          # noqa: E402

init_default_scope('mmpose')

JOINT_NAMES = [
    'Wrist',
    'Thumb_MCP', 'Thumb_PIP', 'Thumb_DIP', 'Thumb_Tip',
    'Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_Tip',
    'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_Tip',
    'Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Ring_Tip',
    'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP', 'Pinky_Tip',
]

MODELS = [
    ('Direct Sup. (ep.25)', 'checkpoints/direct_sup_best.pth'),
    ('Dist. V2-100', 'checkpoints/distilled_v2_epoch_100.pth'),
    ('FT V2-150', 'checkpoints/distilled_v2_ft_epoch_150.pth'),
]

TEACHER_CONFIG = os.path.join(
    ROOT, 'mmpose/configs/hand_2d_keypoint/topdown_heatmap/rhd2d/'
    'td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256.py')
TEACHER_CHECKPOINT = os.path.join(
    ROOT, 'checkpoints/hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=None,
                     help='cap on number of samples (default: full test set)')
    args = ap.parse_args()

    t_start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('Loading teacher (HRNetV2-W18 + DARK)...')
    teacher = init_model(TEACHER_CONFIG, TEACHER_CHECKPOINT, device=device)
    teacher.cfg.model.test_cfg.output_heatmaps = True
    if hasattr(teacher, 'head') and hasattr(teacher.head, 'test_cfg'):
        if isinstance(teacher.head.test_cfg, dict):
            teacher.head.test_cfg['output_heatmaps'] = True
        else:
            teacher.head.test_cfg.output_heatmaps = True
    teacher.eval()
    codec = MSRAHeatmap(input_size=(256, 256), heatmap_size=(64, 64),
                         sigma=2, unbiased=True)

    print('Loading students...')
    students = {}
    for name, ckpt in MODELS:
        m = BlazeHandLandmark().to(device)
        m.load_state_dict(torch.load(os.path.join(ROOT, ckpt), map_location=device))
        m.eval()
        students[name] = m
        print(f'  loaded {name} <- {ckpt}')

    pipeline = [
        dict(type='LoadImage'),
        dict(type='GetBBoxCenterScale'),
        dict(type='TopdownAffine', input_size=(256, 256)),
        # pack_transformed=True is required: TopdownAffine writes the
        # crop-warped keypoints to results['transformed_keypoints'] and
        # leaves results['keypoints'] as the original, un-warped
        # image-space annotation. Without this flag, gt_instances.keypoints
        # is NOT in the same 256x256 crop space as the student/teacher
        # predictions below.
        dict(type='PackPoseInputs', pack_transformed=True),
    ]
    ds = Rhd2DDataset(
        data_root=os.path.join(ROOT, 'dataset', 'rhd'),
        ann_file='annotations/rhd_test.json',
        pipeline=pipeline,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                         collate_fn=pseudo_collate)

    N = len(ds) if args.samples is None else min(args.samples, len(ds))
    threshold = 0.2 * 256
    auc_thresholds = np.linspace(0, 0.5 * 256, 20)

    # accumulators[model][space] -> dict(correct=(21,), dists=list of (21,))
    acc = {
        name: {
            'gt': {'correct': np.zeros(21), 'dists': []},
            'teacher': {'correct': np.zeros(21), 'dists': []},
        }
        for name, _ in MODELS
    }

    print(f'\nRunning dual-space eval on {N} RHD test samples...')
    it = iter(loader)
    for idx in range(N):
        batch = next(it)
        img_tensor = torch.stack(batch['inputs']).float() / 255.0

        gt_raw = np.array(batch['data_samples'][0].gt_instances.transformed_keypoints[0])
        gt_256 = np.clip(gt_raw, 0, 255).astype(np.float32)

        with torch.no_grad():
            out = teacher.predict(img_tensor.to(device), batch['data_samples'])
            hms = out[0].pred_fields.heatmaps.cpu().numpy()
            decoded = codec.decode(hms)
            tc = (np.array(decoded['keypoints']) if isinstance(decoded, dict)
                  else np.array(decoded[0])).reshape(21, 2)
            tc_256 = np.clip(tc, 0, 255).astype(np.float32)

        for name, _ in MODELS:
            with torch.no_grad():
                _, _, pred = students[name](img_tensor.to(device))
                sc = pred[0, :, :2].cpu().numpy() * 256

            d_gt = np.sqrt(np.sum((sc - gt_256) ** 2, axis=1))
            d_tc = np.sqrt(np.sum((sc - tc_256) ** 2, axis=1))
            acc[name]['gt']['dists'].append(d_gt)
            acc[name]['gt']['correct'] += (d_gt <= threshold).astype(float)
            acc[name]['teacher']['dists'].append(d_tc)
            acc[name]['teacher']['correct'] += (d_tc <= threshold).astype(float)

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t_start
            print(f'  [{idx+1}/{N}]  {elapsed:6.1f}s elapsed')

    elapsed = time.time() - t_start
    print(f'\nInference done in {elapsed:.1f}s ({N} samples, {len(MODELS)} models, 2 spaces)')

    # ---- aggregate + write CSVs -----------------------------------------
    results_dir = os.path.join(ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)

    summary = {}  # summary[name][space] = dict(pck, mpjpe, auc, pck_pj, mpjpe_pj)
    for name, _ in MODELS:
        summary[name] = {}
        for space in ('gt', 'teacher'):
            dists = np.array(acc[name][space]['dists'])
            correct = acc[name][space]['correct']
            pck = correct.sum() / (N * 21)
            mpjpe = dists.mean()
            auc_v = [(dists <= t).mean() for t in auc_thresholds]
            auc = float(np.trapz(auc_v, auc_thresholds / (0.5 * 256)))
            summary[name][space] = dict(
                pck=pck, mpjpe=mpjpe, auc=auc,
                pck_pj=correct / N, mpjpe_pj=dists.mean(axis=0),
            )

    command = (f'python scripts/rerun_dual_space_eval.py'
               + (f' --samples {args.samples}' if args.samples else ''))

    # table_viii_both_spaces.csv
    csv_path = os.path.join(results_dir, 'table_viii_both_spaces.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['model', 'joint', 'metric', 'value', 'coordinate_space',
                    'script', 'command'])
        for name, _ in MODELS:
            for space, space_label in [('gt', 'GT-space'), ('teacher', 'teacher-decoded-space')]:
                s = summary[name][space]
                w.writerow([name, 'ALL', 'PCK@0.2', f'{s["pck"]:.6f}', space_label,
                            'scripts/rerun_dual_space_eval.py', command])
                w.writerow([name, 'ALL', 'MPJPE', f'{s["mpjpe"]:.6f}', space_label,
                            'scripts/rerun_dual_space_eval.py', command])
                w.writerow([name, 'ALL', 'AUC', f'{s["auc"]:.6f}', space_label,
                            'scripts/rerun_dual_space_eval.py', command])
                for j, jname in enumerate(JOINT_NAMES):
                    w.writerow([name, jname, 'PCK@0.2', f'{s["pck_pj"][j]:.6f}', space_label,
                                'scripts/rerun_dual_space_eval.py', command])
                    w.writerow([name, jname, 'MPJPE', f'{s["mpjpe_pj"][j]:.6f}', space_label,
                                'scripts/rerun_dual_space_eval.py', command])
    print(f'Wrote {csv_path}')

    # per_joint_winloss_{gt,teacher}_space.csv : Direct Sup vs FT V2-150 (final proposed model)
    for space, space_label, fname in [
        ('gt', 'GT-space', 'per_joint_winloss_gt_space.csv'),
        ('teacher', 'teacher-decoded-space', 'per_joint_winloss_teacher_space.csv'),
    ]:
        direct_pj = summary['Direct Sup. (ep.25)'][space]['pck_pj']
        ft_pj = summary['FT V2-150'][space]['pck_pj']
        path = os.path.join(results_dir, fname)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['joint', 'direct_value', 'distilled_value', 'winner', 'margin'])
            for j, jname in enumerate(JOINT_NAMES):
                dv, fv = direct_pj[j], ft_pj[j]
                winner = 'Direct Sup.' if dv > fv else ('FT V2-150' if fv > dv else 'tie')
                w.writerow([jname, f'{dv:.6f}', f'{fv:.6f}', winner, f'{fv - dv:+.6f}'])
        print(f'Wrote {path}  (space={space_label}, Direct Sup. vs FT V2-150)')

    # human-readable summary
    txt_path = os.path.join(results_dir, 'rerun_dual_space_summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('rerun_dual_space_eval.py — summary\n')
        f.write(f'Samples: {N} | Device: {device} | Wall-clock: {elapsed:.1f}s\n')
        f.write(f'Command: {command}\n\n')
        for name, _ in MODELS:
            f.write(f'{name}\n')
            for space, space_label in [('gt', 'GT-space'), ('teacher', 'teacher-decoded-space')]:
                s = summary[name][space]
                f.write(f'  {space_label:22s} PCK@0.2={s["pck"]:.4f}  '
                        f'MPJPE={s["mpjpe"]:.4f}px  AUC={s["auc"]:.4f}\n')
            f.write('\n')
    print(f'Wrote {txt_path}')

    print('\n' + '=' * 78)
    for name, _ in MODELS:
        for space, space_label in [('gt', 'GT-space'), ('teacher', 'teacher-decoded-space')]:
            s = summary[name][space]
            print(f'{name:22s} {space_label:22s} PCK={s["pck"]:.4f} MPJPE={s["mpjpe"]:7.4f}px AUC={s["auc"]:.4f}')
    print('=' * 78)
    print(f'Total wall-clock: {elapsed:.1f}s')


if __name__ == '__main__':
    main()
