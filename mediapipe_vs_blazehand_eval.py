"""
Comparison: MediaPipe SDK vs BlazeHandLandmark (zmurez starting weights)
Dataset   : RHD2D test set (256x256 crops, GT in 256x256 space)
Reference : Ground truth keypoints from rhd_test.json

Both models run on the same pre-cropped 256x256 hand images.
BlazeHandLandmark uses the original pre-trained weights from the zmurez
repository (student_model/blazehand_landmark.pth) — no distillation applied.
"""

import sys
import os

# Make student_model/ importable from root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'student_model'))

import cv2
import numpy as np
import torch
import mediapipe as mp
from tqdm import tqdm
from torch.utils.data import DataLoader
from mmpose.datasets import Rhd2DDataset
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope

from blazehand_landmark import BlazeHandLandmark

init_default_scope('mmpose')

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
STUDENT_DIR = os.path.join(SCRIPT_DIR, 'student_model')
RHD_ROOT    = os.path.join(SCRIPT_DIR, 'dataset', 'rhd')
ANN_FILE    = 'annotations/rhd_test.json'
SAMPLES     = None           # None = full test set; set an int (e.g. 500) for quick eval
THRESHOLD   = 0.2 * 256      # PCK@0.2 in 256x256 pixel space (= 51.2 px)
OUT_FILE    = os.path.join(SCRIPT_DIR, 'mediapipe_vs_blazehand_results.txt')

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')

# ── MediaPipe SDK (official full model, model_complexity=1, ~1.98M params) ───
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.01,
    model_complexity=1,
)
print('Loaded : MediaPipe SDK (model_complexity=1)')

# ── BlazeHandLandmark — zmurez starting weights ───────────────────────────────
hand_regressor = BlazeHandLandmark().to(device)
hand_regressor.load_weights(os.path.join(STUDENT_DIR, 'blazehand_landmark.pth'))
print(f'Loaded : BlazeHandLandmark (student_model/blazehand_landmark.pth)')

# ── RHD2D test dataset (MMPose pipeline → 256x256 crops) ─────────────────────
pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(type='PackPoseInputs'),
]
ds     = Rhd2DDataset(data_root=RHD_ROOT, ann_file=ANN_FILE, pipeline=pipeline)
loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=pseudo_collate)
N      = len(ds) if SAMPLES is None else min(SAMPLES, len(ds))
print(f'Dataset: RHD2D test set — {N} samples\n')

# ── Joint ordering ────────────────────────────────────────────────────────────
# RHD GT uses a tip-first ordering; MediaPipe uses wrist-first / MCP-first.
# MP_TO_RHD[i] = RHD GT index that corresponds to MediaPipe joint i.
MP_TO_RHD = [
    0,            # Wrist
    4, 3, 2, 1,   # Thumb  : CMC, MCP, IP, Tip
    8, 7, 6, 5,   # Index  : MCP, PIP, DIP, Tip
    12,11,10, 9,  # Middle : MCP, PIP, DIP, Tip
    16,15,14,13,  # Ring   : MCP, PIP, DIP, Tip
    20,19,18,17,  # Pinky  : MCP, PIP, DIP, Tip
]

JOINT_NAMES = [
    'Wrist',
    'Thumb_CMC', 'Thumb_MCP', 'Thumb_IP',  'Thumb_Tip',
    'Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_Tip',
    'Middle_MCP','Middle_PIP','Middle_DIP','Middle_Tip',
    'Ring_MCP',  'Ring_PIP',  'Ring_DIP',  'Ring_Tip',
    'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP', 'Pinky_Tip',
]

# ── Accumulators ──────────────────────────────────────────────────────────────
auc_thresholds = np.linspace(0, 0.5 * 256, 20)

mp_correct  = np.zeros(21); mp_dists  = []; mp_detected  = 0; mp_failed  = 0
blz_correct = np.zeros(21); blz_dists = []; blz_detected = 0; blz_failed = 0

torch.set_grad_enabled(False)

# ── Evaluation loop ───────────────────────────────────────────────────────────
for idx, batch in enumerate(tqdm(loader, total=N, desc='Evaluating')):
    if idx >= N:
        break

    img_tensor = torch.stack(batch['inputs']).float() / 255.0          # (1,3,256,256)
    gt_kpts    = np.array(batch['data_samples'][0].gt_instances.keypoints[0])  # (21,2)
    gt_ordered = gt_kpts[MP_TO_RHD]   # reorder GT to MediaPipe joint numbering

    # BGR→RGB uint8  (256×256×3)
    img_np  = (img_tensor[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    # ── MediaPipe SDK ─────────────────────────────────────────────────────────
    result = mp_hands.process(img_rgb)
    if result.multi_hand_landmarks:
        mp_detected += 1
        lm      = result.multi_hand_landmarks[0].landmark
        mp_xy   = np.array([[l.x * 256, l.y * 256] for l in lm])   # (21,2)
        dist_mp = np.linalg.norm(mp_xy - gt_ordered, axis=1)
        mp_correct += (dist_mp <= THRESHOLD)
    else:
        mp_failed += 1
        dist_mp = np.full(21, 256.0)
    mp_dists.append(dist_mp)

    # ── BlazeHandLandmark — treat entire 256×256 crop as the hand ROI ─────────
    # The MMPose crop already isolates the hand, matching how BlazePalm would
    # crop it in the full pipeline.  We pass an identity-like ROI so that
    # extract_roi maps the whole image into BlazeHandLandmark's 256×256 input.
    xc    = torch.tensor([128.0])
    yc    = torch.tensor([128.0])
    scale = torch.tensor([256.0])
    theta = torch.tensor([0.0])
    img_roi, affine, _ = hand_regressor.extract_roi(img_rgb, xc, yc, theta, scale)
    flags, _, norm_lm  = hand_regressor(img_roi.to(device))

    if flags[0].item() > 0.5:
        blz_detected += 1
        landmarks = hand_regressor.denormalize_landmarks(norm_lm.cpu(), affine)  # (1,21,3)
        blz_xy    = landmarks[0, :, :2].numpy()                                  # (21,2)
        dist_blz  = np.linalg.norm(blz_xy - gt_ordered, axis=1)
        blz_correct += (dist_blz <= THRESHOLD)
    else:
        blz_failed += 1
        dist_blz = np.full(21, 256.0)
    blz_dists.append(dist_blz)

    if (idx + 1) % 500 == 0:
        tqdm.write(
            f'[{idx+1}/{N}]  '
            f'MP  PCK@0.2={mp_correct.sum()/((idx+1)*21):.4f}  Det={mp_detected/(idx+1)*100:.1f}%  |  '
            f'BLZ PCK@0.2={blz_correct.sum()/((idx+1)*21):.4f}  Det={blz_detected/(idx+1)*100:.1f}%'
        )

# ── Aggregate metrics ─────────────────────────────────────────────────────────
mp_dists  = np.array(mp_dists)    # (N, 21)
blz_dists = np.array(blz_dists)   # (N, 21)


def compute_metrics(dists, correct, detected, n):
    pck       = correct.sum() / (n * 21)
    mpjpe     = dists.mean()
    mse       = np.mean(np.sum(dists ** 2, axis=1) / 21)
    det_rate  = detected / n * 100
    auc_vals  = [(dists <= t).mean() for t in auc_thresholds]
    auc       = float(np.trapz(auc_vals, auc_thresholds / (0.5 * 256)))
    pck_per   = correct / n
    mpjpe_per = dists.mean(axis=0)
    return dict(
        pck=pck, mpjpe=mpjpe, mse=mse, auc=auc,
        det_rate=det_rate, detected=detected, failed=n - detected,
        pck_per=pck_per, mpjpe_per=mpjpe_per,
    )


mp_m  = compute_metrics(mp_dists,  mp_correct,  mp_detected,  N)
blz_m = compute_metrics(blz_dists, blz_correct, blz_detected, N)

# ── Console output ────────────────────────────────────────────────────────────
SEP = '=' * 68

print()
print(SEP)
print('  MEDIAPIPE SDK  vs  BLAZEHANDLANDMARK (zmurez starting weights)')
print('  Dataset  : RHD2D test set  |  Reference : Ground truth')
print(f'  Samples  : {N}  |  Threshold : PCK@0.2 ({THRESHOLD:.1f} px in 256×256)')
print(SEP)
print(f'  {"Metric":<22} {"MediaPipe SDK":>18} {"BlazeHand (zmurez)":>18}')
print(f'  {"-"*58}')

rows = [
    ('PCK@0.2',       f'{mp_m["pck"]:.4f} ({mp_m["pck"]*100:.2f}%)',    f'{blz_m["pck"]:.4f} ({blz_m["pck"]*100:.2f}%)'),
    ('MPJPE (px)',    f'{mp_m["mpjpe"]:.4f}',                             f'{blz_m["mpjpe"]:.4f}'),
    ('MSE (px²)',f'{mp_m["mse"]:.4f}',                               f'{blz_m["mse"]:.4f}'),
    ('AUC',           f'{mp_m["auc"]:.4f}',                               f'{blz_m["auc"]:.4f}'),
    ('Detection Rate',f'{mp_m["det_rate"]:.2f}%',                         f'{blz_m["det_rate"]:.2f}%'),
    ('Detected',      f'{mp_m["detected"]}/{N}',                          f'{blz_m["detected"]}/{N}'),
    ('Not Detected',  f'{mp_m["failed"]}/{N}',                            f'{blz_m["failed"]}/{N}'),
]
for label, mv, bv in rows:
    print(f'  {label:<22} {mv:>18} {bv:>18}')

print(SEP)
print()
print(f'  {"Joint":<16} {"MP PCK@0.2":>11} {"MP MPJPE":>10} {"BLZ PCK@0.2":>13} {"BLZ MPJPE":>11}')
print(f'  {"-"*64}')
for i, name in enumerate(JOINT_NAMES):
    print(
        f'  {name:<16}'
        f' {mp_m["pck_per"][i]:>11.4f}'
        f' {mp_m["mpjpe_per"][i]:>10.4f}'
        f' {blz_m["pck_per"][i]:>13.4f}'
        f' {blz_m["mpjpe_per"][i]:>11.4f}'
    )
print(SEP)

# ── Save to text file ─────────────────────────────────────────────────────────
with open(OUT_FILE, 'w', encoding='utf-8') as f:
    f.write('MediaPipe SDK vs BlazeHandLandmark (zmurez starting weights)\n')
    f.write('Dataset  : RHD2D test set | Reference: Ground truth\n')
    f.write(f'Samples  : {N} | Threshold: PCK@0.2 ({THRESHOLD:.1f} px in 256x256)\n\n')

    f.write(f'{"Metric":<22} {"MediaPipe SDK":>22} {"BlazeHand (zmurez)":>22}\n')
    f.write('-' * 68 + '\n')
    flat_rows = [
        ('PCK@0.2',      mp_m['pck'],      blz_m['pck']),
        ('MPJPE (px)',   mp_m['mpjpe'],    blz_m['mpjpe']),
        ('MSE (px2)',    mp_m['mse'],      blz_m['mse']),
        ('AUC',          mp_m['auc'],      blz_m['auc']),
        ('Det Rate (%)', mp_m['det_rate'], blz_m['det_rate']),
    ]
    for label, mv, bv in flat_rows:
        f.write(f'{label:<22} {mv:>22.4f} {bv:>22.4f}\n')
    f.write(f'{"Detected":<22} {mp_m["detected"]:>21}/{N} {blz_m["detected"]:>21}/{N}\n')
    f.write(f'{"Not Detected":<22} {mp_m["failed"]:>21}/{N} {blz_m["failed"]:>21}/{N}\n')

    f.write('\nPer-joint breakdown:\n')
    f.write(f'{"Joint":<16} {"MP PCK@0.2":>11} {"MP MPJPE":>10} {"BLZ PCK@0.2":>13} {"BLZ MPJPE":>11}\n')
    f.write('-' * 64 + '\n')
    for i, name in enumerate(JOINT_NAMES):
        f.write(
            f'{name:<16}'
            f' {mp_m["pck_per"][i]:>11.4f}'
            f' {mp_m["mpjpe_per"][i]:>10.4f}'
            f' {blz_m["pck_per"][i]:>13.4f}'
            f' {blz_m["mpjpe_per"][i]:>11.4f}\n'
        )

print(f'\nSaved : {OUT_FILE}')
