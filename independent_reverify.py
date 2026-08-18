#!/usr/bin/env python3
"""
independent_reverify.py

Independent, from-scratch re-verification of every trained/reference model's
accuracy numbers on the full 2,727-sample RHD2D test set.

This script does NOT import or call into directsup_gt_comprehensive_eval.py,
experiments/v2_GT_comprehensive_eval.py, experiments/distilled_v2_ft_ep150_gt_
comprehensive_eval.py, scripts/rerun_dual_space_eval.py, scripts/untrained_gt_
eval.py, or mediapipe_vs_blazehand_eval.py. It is written fresh against the
raw checkpoints, the raw RHD2D annotations, and the mmpose/BlazeHandLandmark
APIs directly. Where it necessarily reproduces a *protocol* choice from those
scripts (coordinate space, detection-failure penalty, AUC threshold grid) that
choice is called out explicitly in a comment, because the whole point of this
run is to check whether the previously reported numbers hold up under the
SAME, sound protocol -- not to invent a new, non-comparable one.

Protocol summary (GT-space, used for 6 of the 7 models):
  - GT reference  = gt_instances.transformed_keypoints (the crop-warped
    keypoints PackPoseInputs(pack_transformed=True) exposes), clipped to
    [0, 255]. NOT gt_instances.keypoints, which TopdownAffine leaves in
    original-image space (verified directly against
    mmpose/datasets/transforms/topdown_transforms.py in this repo -- see
    Part B of the audit this script belongs to).
  - Fixed PCK@0.2 threshold = 0.2*256 = 51.2 px (not bbox-relative).
  - Direct Sup. / Dist. V2-100 / FT V2-150 / Untrained: BlazeHandLandmark
    consumes the full GT-cropped 256x256 affine-warped crop directly, no
    separate hand-detection/rejection step, so every sample is scored
    (hand_flag only feeds a separate Det.Rate statistic, per directsup_gt_
    comprehensive_eval.py's own convention, reproduced here after
    independently reading that file, not by importing it).
  - zmurez pre-distillation weights / MediaPipe SDK: two-stage pipeline
    (own hand-presence decision). A rejected crop is NOT excluded from
    scoring -- all 21 joints are penalized at 256 px, per the paper's
    stated protocol (Sec. 5.1) and mediapipe_vs_blazehand_eval.py's
    convention.

Teacher is the one model NOT run under the fixed-threshold GT-space
protocol above by default, because the number this audit needs to check
(PCK=0.9918/AUC=0.9023/EPE=2.18, claimed in the paper and in
results/table_iv_teacher_own_protocol.csv) was itself produced under a
different, official protocol: mmpose's own Runner.test() with the
official registered PCKAccuracy/AUC/EPE metrics (bbox-relative PCK
threshold, original-image coordinate space, inverse-affine applied
automatically by TopdownPoseEstimator). Reproducing THAT number means
invoking that same official harness -- not reimplementing mmpose's metric
math by hand, which would just be a second opinion on a formula, not a
verification of the number. This script also additionally computes the
teacher's crop-space/fixed-threshold number (for internal consistency
with the other six rows), clearly labelled as supplementary since neither
reference CSV contains an entry to check it against.

Usage:
    python independent_reverify.py

Writes:
    results/independent_reverify_raw.csv
    results/independent_reverify_comparison.csv
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

_ap = argparse.ArgumentParser()
_ap.add_argument('--limit', type=int, default=None,
                  help='debug only: cap sample count (skips the 2727-sample assert)')
_cli_args = _ap.parse_args()

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'student_model'))
sys.path.insert(0, ROOT)

from mmengine.registry import init_default_scope          # noqa: E402
from mmengine.dataset import pseudo_collate                # noqa: E402
from mmengine.config import Config                          # noqa: E402
from mmengine.runner import Runner                           # noqa: E402
from mmpose.apis import init_model                          # noqa: E402
from mmpose.codecs import MSRAHeatmap                        # noqa: E402
from mmpose.datasets import Rhd2DDataset                     # noqa: E402
from torch.utils.data import DataLoader                       # noqa: E402

from blazehand_landmark import BlazeHandLandmark              # noqa: E402

init_default_scope('mmpose')

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
JOINT_NAMES = [
    'Wrist',
    'Thumb_MCP', 'Thumb_PIP', 'Thumb_DIP', 'Thumb_Tip',
    'Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_Tip',
    'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_Tip',
    'Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Ring_Tip',
    'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP', 'Pinky_Tip',
]
FIXED_THR = 0.2 * 256.0                       # 51.2 px
AUC_THRESHOLDS = np.linspace(0, 0.5 * 256, 20)  # same grid as the eval scripts audited in Part B
RESULTS_DIR = os.path.join(ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

TEACHER_CONFIG = os.path.join(
    ROOT, 'mmpose/configs/hand_2d_keypoint/topdown_heatmap/rhd2d/'
    'td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256.py')
TEACHER_CKPT = os.path.join(
    ROOT, 'checkpoints/hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth')

CHECKPOINTS = {
    'Direct Sup. (ep.25)': os.path.join(ROOT, 'checkpoints/direct_sup_best.pth'),
    'Dist. V2-100': os.path.join(ROOT, 'checkpoints/distilled_v2_epoch_100.pth'),
    'FT V2-150': os.path.join(ROOT, 'checkpoints/distilled_v2_ft_epoch_150.pth'),
}
ZMUREZ_CKPT = os.path.join(ROOT, 'student_model/blazehand_landmark.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')


def metrics_from_dists(dists, correct, n):
    """dists: list of per-sample or per-batch (k,21) / (21,) per-joint L2
    error arrays; concatenated here into (n,21). correct: (21,) count of
    joints within FIXED_THR, accumulated by caller (so callers that apply a
    detection-failure penalty can just feed the penalized distances in and
    the correct-count falls out consistently)."""
    dists = np.concatenate([np.atleast_2d(x) for x in dists], axis=0)
    assert dists.shape == (n, 21), f'expected ({n},21), got {dists.shape}'
    pck = float(correct.sum() / (n * 21))
    mpjpe = float(dists.mean())
    mse = float(np.mean(np.sum(dists ** 2, axis=1) / 21))
    auc_v = [(dists <= t).mean() for t in AUC_THRESHOLDS]
    auc = float(np.trapz(auc_v, AUC_THRESHOLDS / (0.5 * 256)))
    pck_pj = correct / n
    mpjpe_pj = dists.mean(axis=0)
    return dict(pck=pck, mpjpe=mpjpe, mse=mse, auc=auc, pck_pj=pck_pj, mpjpe_pj=mpjpe_pj)


# ==========================================================================
# PART 1 -- dataset, sanity check
# ==========================================================================
print('=' * 78)
print('PART 1 -- loading RHD2D test split, sanity-checking coordinate space')
print('=' * 78)

pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256, 256)),
    # pack_transformed=True exposes BOTH gt_instances.keypoints (original
    # image space, untouched by TopdownAffine) AND gt_instances.
    # transformed_keypoints (crop-warped, 256x256 space) on every sample,
    # so a single pipeline/dataloader serves every model below regardless
    # of which coordinate space each one needs.
    dict(type='PackPoseInputs', pack_transformed=True),
]
ds = Rhd2DDataset(
    data_root=os.path.join(ROOT, 'dataset', 'rhd'),
    ann_file='annotations/rhd_test.json',
    pipeline=pipeline,
)
N = len(ds)
print(f'RHD2D test split: {N} samples (expected 2727)')
if _cli_args.limit is None:
    assert N == 2727, f'Unexpected test-set size: {N}'
else:
    from torch.utils.data import Subset
    N = min(_cli_args.limit, N)
    ds = Subset(ds, list(range(N)))
    print(f'DEBUG: limiting to first {N} samples (--limit)')

print('\nManual sanity check -- first 5 samples, wrist joint (index 0):')
print(f'{"idx":>3}  {"orig-space (keypoints)":>26}  {"crop-space (transformed_keypoints)":>36}  {"img w x h":>10}')
for i in range(5):
    s = ds[i]
    gt_o = np.array(s['data_samples'].gt_instances.keypoints[0])[0]
    gt_t = np.array(s['data_samples'].gt_instances.transformed_keypoints[0])[0]
    meta = s['data_samples'].metainfo
    print(f'{i:>3}  {str(np.round(gt_o, 1)):>26}  {str(np.round(gt_t, 1)):>36}  '
          f'{meta.get("ori_shape", meta.get("img_shape", "?"))}')
print('\nExpectation: orig-space values are RHD-native (image is 320x320, so')
print('values can range roughly 0-320); crop-space values should mostly fall')
print('inside [0,255] since the GT bbox crop is centered on the hand.')
print('If crop-space values above are NOT roughly in [0,255], STOP -- the')
print('space is wrong and nothing downstream should be trusted.\n')

for i in range(5):
    s = ds[i]
    gt_t = np.array(s['data_samples'].gt_instances.transformed_keypoints[0])
    assert -5 <= gt_t[:, 0].mean() <= 260 and -5 <= gt_t[:, 1].mean() <= 260, \
        f'Sample {i}: crop-space GT mean out of plausible [0,255] range: {gt_t.mean(axis=0)}'
print('Sanity check passed: crop-space GT coordinates are in the expected range.\n')

loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0,
                     collate_fn=pseudo_collate)

# ==========================================================================
# PART 2 -- batched pass: Direct Sup. / Dist. V2-100 / FT V2-150 / Untrained
#           / Teacher, all fed the SAME 256x256 GT-cropped tensor per batch
# ==========================================================================
print('=' * 78)
print('PART 2 -- Direct Sup. / Dist. V2-100 / FT V2-150 / Untrained / Teacher')
print('=' * 78)

students = {}
for name, ckpt in CHECKPOINTS.items():
    m = BlazeHandLandmark().to(device)
    sd = torch.load(ckpt, map_location=device)
    m.load_state_dict(sd)
    m.eval()
    students[name] = m
    sz = os.path.getsize(ckpt)
    print(f'  loaded {name:<22s} <- {os.path.relpath(ckpt, ROOT)}  ({sz:,} bytes)')

torch.manual_seed(0)
untrained = BlazeHandLandmark().to(device)
untrained.eval()
students['Untrained'] = untrained
print('  loaded Untrained             <- random init, torch.manual_seed(0), no checkpoint')

print('  loading Teacher (HRNetV2-W18 + DARK)...')
teacher = init_model(TEACHER_CONFIG, TEACHER_CKPT, device=device)
teacher.cfg.model.test_cfg.output_heatmaps = True
if hasattr(teacher, 'head') and hasattr(teacher.head, 'test_cfg'):
    if isinstance(teacher.head.test_cfg, dict):
        teacher.head.test_cfg['output_heatmaps'] = True
    else:
        teacher.head.test_cfg.output_heatmaps = True
teacher.eval()
codec = MSRAHeatmap(input_size=(256, 256), heatmap_size=(64, 64), sigma=2, unbiased=True)
sz = os.path.getsize(TEACHER_CKPT)
print(f'  loaded Teacher                <- {os.path.relpath(TEACHER_CKPT, ROOT)}  ({sz:,} bytes)\n')

model_names = list(students.keys())  # Direct Sup., Dist. V2-100, FT V2-150, Untrained
acc = {name: {'dists': [], 'correct': np.zeros(21), 'flags': []} for name in model_names}
acc['Teacher_cropspace'] = {'dists': [], 'correct': np.zeros(21)}          # supplementary
acc['Teacher_ownprotocol'] = {'dists': [], 'bbox': []}                     # for cross-check only

t0 = time.time()
for b_idx, batch in enumerate(loader):
    img_tensor = torch.stack(batch['inputs']).to(device).float() / 255.0
    bsz = img_tensor.shape[0]

    gt_crop = np.stack([
        np.clip(np.array(ds_.gt_instances.transformed_keypoints[0]), 0, 255)
        for ds_ in batch['data_samples']
    ]).astype(np.float32)  # (b,21,2)

    with torch.no_grad():
        for name, model in students.items():
            flag, _, pred = model(img_tensor)
            sc = pred[:, :, :2].cpu().numpy() * 256.0  # (b,21,2)
            d = np.sqrt(np.sum((sc - gt_crop) ** 2, axis=2))  # (b,21)
            acc[name]['dists'].append(d)
            acc[name]['correct'] += (d <= FIXED_THR).sum(axis=0)
            acc[name]['flags'].extend(flag.cpu().numpy().tolist())

        # Teacher: single forward pass feeds both the crop-space decode
        # (raw heatmap -> MSRAHeatmap.decode(), same codec distillation_v2.py
        # trains against) and the official inverse-affine'd pred_instances
        # (what PCKAccuracy/AUC/EPE in the official config score against).
        out = teacher.predict(img_tensor, batch['data_samples'])
        for i, res in enumerate(out):
            hms = res.pred_fields.heatmaps
            if torch.is_tensor(hms):
                hms = hms.detach().cpu().numpy()
            dec = codec.decode(hms)
            tc = (np.array(dec['keypoints']) if isinstance(dec, dict) else np.array(dec[0]))
            tc = np.clip(tc.reshape(21, 2), 0, 255)
            d_crop = np.sqrt(np.sum((tc - gt_crop[i]) ** 2, axis=1))
            acc['Teacher_cropspace']['dists'].append(d_crop)
            acc['Teacher_cropspace']['correct'] += (d_crop <= FIXED_THR)

            pred_orig = np.array(res.pred_instances.keypoints[0]).reshape(21, 2)
            gt_orig = np.array(batch['data_samples'][i].gt_instances.keypoints[0]).reshape(21, 2)
            d_orig = np.sqrt(np.sum((pred_orig - gt_orig) ** 2, axis=1))
            bbox = np.array(batch['data_samples'][i].gt_instances.bboxes[0])
            bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            acc['Teacher_ownprotocol']['dists'].append(d_orig)
            acc['Teacher_ownprotocol']['bbox'].append(bbox_size)

    done = min((b_idx + 1) * loader.batch_size, N)
    if done % 512 < loader.batch_size or done == N:
        print(f'  [{done:4d}/{N}]  {time.time()-t0:6.1f}s elapsed')

print(f'\nPart 2 done in {time.time()-t0:.1f}s\n')

results = {}
for name in model_names:
    m = metrics_from_dists(acc[name]['dists'], acc[name]['correct'], N)
    flags = np.array(acc[name]['flags'])
    m['det_rate'] = float((flags >= 0.5).mean() * 100)
    results[name] = m
    print(f'{name:<22s} PCK={m["pck"]:.4f}  MPJPE={m["mpjpe"]:7.4f}px  '
          f'AUC={m["auc"]:.4f}  Det={m["det_rate"]:6.2f}%')

# Teacher, crop-space / fixed-threshold (supplementary -- no reference to check against)
tcs = acc['Teacher_cropspace']
m = metrics_from_dists(tcs['dists'], tcs['correct'], N)
m['det_rate'] = 100.0  # top-down teacher always receives a bbox
results['Teacher (crop-space, fixed-thr, supplementary)'] = m
print(f'{"Teacher (crop-space, supplementary)":<22s} PCK={m["pck"]:.4f}  '
      f'MPJPE={m["mpjpe"]:7.4f}px  AUC={m["auc"]:.4f}')

# ==========================================================================
# PART 3 -- Teacher, official own-protocol reproduction via mmpose Runner
#           (independent authority for the 0.9918/0.9023/2.18 claim: uses
#           mmpose's own registered PCKAccuracy/AUC/EPE metrics rather than
#           a hand-rolled formula, so this is a check of the NUMBER, not a
#           second implementation of the metric math)
# ==========================================================================
print()
print('=' * 78)
print('PART 3 -- Teacher official own-protocol reproduction (mmengine Runner)')
print('=' * 78)

cfg = Config.fromfile(TEACHER_CONFIG)
cfg.load_from = TEACHER_CKPT
cfg.work_dir = os.path.join(ROOT, 'eval_results', 'independent_reverify_teacher')
for split in ('val_dataloader', 'test_dataloader'):
    dl = cfg.get(split)
    if dl is not None:
        dl['dataset']['data_root'] = os.path.join(ROOT, 'dataset', 'rhd')
        # avoid Windows multiprocessing-spawn issues when this script is
        # invoked directly (not guarded by if __name__ == '__main__')
        dl['num_workers'] = 0
        dl['persistent_workers'] = False
os.makedirs(cfg.work_dir, exist_ok=True)
runner = Runner.from_cfg(cfg)
official_metrics = runner.test()
print('\nOfficial Runner.test() metrics dict:')
for k, v in official_metrics.items():
    print(f'  {k}: {v}')

# Also compute a manual cross-check of the official numbers using the
# original-image-space distances collected in Part 2, purely as a second,
# independent arithmetic path on the SAME per-sample data the Runner used
# (not a re-derivation of mmpose's own metric internals).
teacher_dists = np.array(acc['Teacher_ownprotocol']['dists'])   # (N,21)
teacher_bbox = np.array(acc['Teacher_ownprotocol']['bbox'])     # (N,)
thr_bbox = teacher_bbox[:, None] * 0.2
manual_pck_bbox = float((teacher_dists < thr_bbox).mean())
manual_epe = float(teacher_dists.mean())
auc_thrs_30 = np.linspace(0, 30.0, 21)[1:]
manual_auc = float(np.mean([(teacher_dists < t).mean() for t in auc_thrs_30]))
print('\nManual cross-check (same original-space distances, bbox-relative thr):')
print(f'  PCK@0.2 (bbox) : {manual_pck_bbox:.4f}')
print(f'  EPE            : {manual_epe:.4f} px')
print(f'  AUC (0-30px)   : {manual_auc:.4f}')

results['Teacher (official own-protocol, Runner)'] = dict(
    pck=float(official_metrics.get('PCK', official_metrics.get('pck', float('nan')))),
    auc=float(official_metrics.get('AUC', official_metrics.get('auc', float('nan')))),
    mpjpe=float(official_metrics.get('EPE', official_metrics.get('epe', float('nan')))),
    manual_pck=manual_pck_bbox, manual_auc=manual_auc, manual_mpjpe=manual_epe,
)

# ==========================================================================
# PART 4 -- zmurez pre-distillation weights + MediaPipe SDK (two-stage,
#           detection-failure-penalized protocol, matching the paper's
#           stated evaluation convention in Sec. 5.1)
# ==========================================================================
print()
print('=' * 78)
print('PART 4 -- zmurez pre-distillation weights + official MediaPipe SDK')
print('=' * 78)

import cv2                     # noqa: E402
import mediapipe as mp          # noqa: E402

zmurez = BlazeHandLandmark().to(device)
zmurez.load_state_dict(torch.load(ZMUREZ_CKPT, map_location=device))
zmurez.eval()
sz = os.path.getsize(ZMUREZ_CKPT)
print(f'  loaded zmurez pre-distillation <- {os.path.relpath(ZMUREZ_CKPT, ROOT)}  ({sz:,} bytes)')

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True, max_num_hands=1,
    min_detection_confidence=0.01, model_complexity=1,
)
print('  loaded MediaPipe SDK (pip mediapipe, model_complexity=1)\n')

# RHD is tip-first per finger; MediaPipe is wrist/MCP-first. Map MediaPipe's
# joint order onto the RHD GT index it corresponds to (same topology fact
# used throughout the repo's other MP-vs-RHD scripts -- this is a dataset
# convention, not eval "logic").
MP_TO_RHD = [
    0,
    4, 3, 2, 1,
    8, 7, 6, 5,
    12, 11, 10, 9,
    16, 15, 14, 13,
    20, 19, 18, 17,
]

loader1 = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=pseudo_collate)

zm_dists, zm_correct, zm_detected = [], np.zeros(21), 0
mp_dists, mp_correct, mp_detected = [], np.zeros(21), 0

t0 = time.time()
for idx, batch in enumerate(loader1):
    img_tensor = torch.stack(batch['inputs']).float() / 255.0
    gt_crop = np.clip(np.array(batch['data_samples'][0].gt_instances.transformed_keypoints[0]), 0, 255)
    gt_ordered = gt_crop[MP_TO_RHD]

    img_np = (img_tensor[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # BGR, as loaded
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    # -- zmurez BlazeHandLandmark, own extract_roi/denormalize pipeline,
    #    identity ROI (whole 256x256 crop = the hand ROI, same as the GT
    #    bbox crop already isolates the hand) --
    with torch.no_grad():
        xc = torch.tensor([128.0]); yc = torch.tensor([128.0])
        scale = torch.tensor([256.0]); theta = torch.tensor([0.0])
        roi, affine, _ = zmurez.extract_roi(img_rgb, xc, yc, theta, scale)
        flags, _, norm_lm = zmurez(roi.to(device))
    if flags[0].item() > 0.5:
        zm_detected += 1
        lm = zmurez.denormalize_landmarks(norm_lm.cpu(), affine)
        xy = lm[0, :, :2].numpy()
        d = np.linalg.norm(xy - gt_ordered, axis=1)
        zm_correct += (d <= FIXED_THR)
    else:
        d = np.full(21, 256.0)
    zm_dists.append(d)

    # -- Official MediaPipe SDK --
    res = mp_hands.process(img_rgb)
    if res.multi_hand_landmarks:
        mp_detected += 1
        lm = res.multi_hand_landmarks[0].landmark
        xy = np.array([[p.x * 256, p.y * 256] for p in lm])
        d = np.linalg.norm(xy - gt_ordered, axis=1)
        mp_correct += (d <= FIXED_THR)
    else:
        d = np.full(21, 256.0)
    mp_dists.append(d)

    if (idx + 1) % 500 == 0:
        print(f'  [{idx+1:4d}/{N}]  {time.time()-t0:6.1f}s elapsed')

print(f'\nPart 4 done in {time.time()-t0:.1f}s\n')

m = metrics_from_dists(zm_dists, zm_correct, N)
m['det_rate'] = float(zm_detected / N * 100)
results['BlazeHandLandmark (zmurez pre-distillation)'] = m
print(f'{"zmurez pre-distillation":<22s} PCK={m["pck"]:.4f}  MPJPE={m["mpjpe"]:7.4f}px  '
      f'AUC={m["auc"]:.4f}  Det={m["det_rate"]:6.2f}%')

m = metrics_from_dists(mp_dists, mp_correct, N)
m['det_rate'] = float(mp_detected / N * 100)
results['MediaPipe SDK (official)'] = m
print(f'{"MediaPipe SDK (official)":<22s} PCK={m["pck"]:.4f}  MPJPE={m["mpjpe"]:7.4f}px  '
      f'AUC={m["auc"]:.4f}  Det={m["det_rate"]:6.2f}%')

# ==========================================================================
# PART 5 -- write raw results CSV
# ==========================================================================
raw_csv = os.path.join(RESULTS_DIR, 'independent_reverify_raw.csv')
with open(raw_csv, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['model', 'joint', 'metric', 'value'])
    for name, m in results.items():
        for key, label in [('pck', 'PCK@0.2'), ('mpjpe', 'MPJPE'), ('auc', 'AUC')]:
            if key in m:
                w.writerow([name, 'ALL', label, f'{m[key]:.6f}'])
        if 'pck_pj' in m:
            for j, jn in enumerate(JOINT_NAMES):
                w.writerow([name, jn, 'PCK@0.2', f'{m["pck_pj"][j]:.6f}'])
                w.writerow([name, jn, 'MPJPE', f'{m["mpjpe_pj"][j]:.6f}'])
        if 'det_rate' in m:
            w.writerow([name, 'ALL', 'Det.Rate', f'{m["det_rate"]:.4f}'])
print(f'\nWrote {raw_csv}')

# ==========================================================================
# PART 6 -- compare against the two reference CSVs named in the audit brief
# ==========================================================================
print()
print('=' * 78)
print('PART 6 -- comparison against existing results/*.csv')
print('=' * 78)

TOL_PCT = 0.5  # percent; flag ANY discrepancy above this, per audit brief


def pct_diff(old, new):
    if old == 0:
        return 0.0 if new == 0 else float('inf')
    return abs(new - old) / abs(old) * 100.0


comparison_rows = []

# --- table_iv_vii_viii_gt_space_FINAL.csv: Direct Sup./Dist.V2-100/FT V2-150, ALL + per-joint ---
final_csv_path = os.path.join(RESULTS_DIR, 'table_iv_vii_viii_gt_space_FINAL.csv')
with open(final_csv_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    ref_rows = list(reader)

for row in ref_rows:
    model, joint, metric, old_val = row['model'], row['joint'], row['metric'], float(row['value'])
    if model not in results:
        continue
    m = results[model]
    if joint == 'ALL':
        new_val = {'PCK@0.2': m.get('pck'), 'MPJPE': m.get('mpjpe'), 'AUC': m.get('auc')}.get(metric)
    else:
        if joint not in JOINT_NAMES or 'pck_pj' not in m:
            continue
        j = JOINT_NAMES.index(joint)
        new_val = {'PCK@0.2': m['pck_pj'][j], 'MPJPE': m['mpjpe_pj'][j]}.get(metric)
    if new_val is None:
        continue
    diff = pct_diff(old_val, new_val)
    verdict = 'PASS' if diff <= TOL_PCT else 'FAIL'
    comparison_rows.append([
        'table_iv_vii_viii_gt_space_FINAL.csv', model, joint, metric,
        f'{old_val:.6f}', f'{new_val:.6f}', f'{diff:.4f}', verdict,
    ])

# --- model_identity_and_results_MASTER.csv: all 7 models, ALL only ---
master_csv_path = os.path.join(RESULTS_DIR, 'model_identity_and_results_MASTER.csv')
with open(master_csv_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    master_rows = list(reader)

MASTER_NAME_MAP = {
    'MediaPipe SDK (official)': 'MediaPipe SDK (official)',
    'BlazeHandLandmark (zmurez pre-distillation weights)': 'BlazeHandLandmark (zmurez pre-distillation)',
    'Direct Sup. (ep.25)': 'Direct Sup. (ep.25)',
    'Dist. V2-100': 'Dist. V2-100',
    'FT V2-150': 'FT V2-150',
    'Teacher (HRNetV2-W18 + DARK)': 'Teacher (official own-protocol, Runner)',
    'Untrained BlazeHandLandmark (random init.)': 'Untrained',
}
for row in master_rows:
    old_name = row['model_name']
    new_name = MASTER_NAME_MAP.get(old_name)
    if new_name is None or new_name not in results:
        continue
    m = results[new_name]
    for metric, col in [('PCK@0.2', 'pck_0.2'), ('AUC', 'auc'), ('MPJPE', 'mpjpe')]:
        old_val = float(row[col])
        if new_name == 'Teacher (official own-protocol, Runner)':
            new_val = {'PCK@0.2': m.get('pck'), 'AUC': m.get('auc'), 'MPJPE': m.get('mpjpe')}.get(metric)
        else:
            new_val = {'PCK@0.2': m.get('pck'), 'AUC': m.get('auc'), 'MPJPE': m.get('mpjpe')}.get(metric)
        if new_val is None or (isinstance(new_val, float) and np.isnan(new_val)):
            continue
        diff = pct_diff(old_val, new_val)
        verdict = 'PASS' if diff <= TOL_PCT else 'FAIL'
        comparison_rows.append([
            'model_identity_and_results_MASTER.csv', old_name, 'ALL', metric,
            f'{old_val:.6f}', f'{new_val:.6f}', f'{diff:.4f}', verdict,
        ])

cmp_csv = os.path.join(RESULTS_DIR, 'independent_reverify_comparison.csv')
with open(cmp_csv, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['reference_file', 'model', 'joint', 'metric', 'old_value', 'new_value',
                'pct_diff', 'verdict'])
    w.writerows(comparison_rows)
print(f'Wrote {cmp_csv}\n')

n_pass = sum(1 for r in comparison_rows if r[-1] == 'PASS')
n_fail = sum(1 for r in comparison_rows if r[-1] == 'FAIL')
print(f'Comparisons: {len(comparison_rows)} total, {n_pass} PASS, {n_fail} FAIL '
      f'(tolerance: {TOL_PCT}%)')
if n_fail:
    print('\nFAILs:')
    for r in comparison_rows:
        if r[-1] == 'FAIL':
            print(f'  [{r[0]}] {r[1]} / {r[2]} / {r[3]}: old={r[4]} new={r[5]} diff={r[6]}%')

print('\nDone.')
