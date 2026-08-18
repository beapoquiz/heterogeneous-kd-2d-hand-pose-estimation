#!/usr/bin/env python3
"""
scripts/rescore_detection.py

For each of the three systems compared in Table V (MediaPipe SDK, the
zmurez BlazeHandLandmark starting weights, and our distilled student),
run inference ONCE on the RHD2D test set, cache the raw per-frame
predictions + detection outcome, and derive BOTH reported metric sets
from that single cached run:

  1. all-frames   -- every frame counted, undetected frames scored at the
                      standard 256 px failure penalty (worst case in a
                      256x256 crop)
  2. detected-only -- undetected frames dropped entirely

Because both numbers come from the same saved predictions file per
system, they can't silently diverge (e.g. from re-running a model twice
with different code paths).

Also resolves the Section V-A / Table V contradiction: Section V-A states the zmurez
weights run directly on pre-cropped 256x256 inputs "bypassing the
BlazePalm stage" and defines detection failure as "BlazePalm rejecting a
crop" -- but BlazePalm is never constructed anywhere in that eval path
(see run_zmurez_blazehand below: only BlazeHandLandmark is loaded, fed
through an IDENTITY roi). The 19.95% failure rate in Table V actually
comes from BlazeHandLandmark's OWN hand_flag presence head (sigmoid
output <= 0.5) -- a model-internal confidence threshold, not a BlazePalm
rejection. This script asserts blazepalm is never imported during that
run and prints the exact criterion string to quote in the corrected text.

Usage:
    python scripts/rescore_detection.py                 # full 2727-sample test set
    python scripts/rescore_detection.py --samples 200    # quick smoke test
    python scripts/rescore_detection.py --force          # ignore cached predictions
    python scripts/rescore_detection.py --systems mediapipe_sdk zmurez_blazehand
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT, 'student_model'))

from mmengine.dataset import pseudo_collate           # noqa: E402
from mmengine.registry import init_default_scope      # noqa: E402
from mmpose.datasets import Rhd2DDataset               # noqa: E402

init_default_scope('mmpose')

# ── Paths ──────────────────────────────────────────────────────────────────
RHD_ROOT = os.path.join(ROOT, 'dataset', 'rhd')
ANN_FILE = 'annotations/rhd_test.json'
CACHE_DIR = os.path.join(ROOT, 'results', 'predictions_cache')
OUT_CSV = os.path.join(ROOT, 'results', 'detection_breakdown.csv')
ZMUREZ_WEIGHTS = os.path.join(ROOT, 'student_model', 'blazehand_landmark.pth')
STUDENT_WEIGHTS = os.path.join(ROOT, 'checkpoints', 'distilled_v2_ft_epoch_150.pth')

# ── Protocol constants (shared across all three systems) ───────────────────
THRESH_PX = 0.2 * 256                       # PCK@0.2 in 256x256 pixel space
AUC_THRESHOLDS = np.linspace(0, 0.5 * 256, 20)
FAIL_PENALTY_PX = 256.0                     # the "256 px failure penalty" for all-frames scoring

# RHD (tip-first) GT index -> MediaPipe (wrist/MCP-first) joint order.
# Needed for any system whose landmark head outputs MediaPipe joint
# ordering (MediaPipe SDK, zmurez BlazeHandLandmark). The distilled
# student was trained against RHD-ordered teacher/GT supervision, so its
# output is already in RHD order and must NOT be remapped.
MP_TO_RHD = [
    0,
    4, 3, 2, 1,
    8, 7, 6, 5,
    12, 11, 10, 9,
    16, 15, 14, 13,
    20, 19, 18, 17,
]

FAILURE_CRITERIA = {
    'mediapipe_sdk': (
        "MediaPipe Hands SDK (mp.solutions.hands.Hands: static_image_mode=True, "
        "max_num_hands=1, model_complexity=1, min_detection_confidence=0.01). "
        "Failure iff result.multi_hand_landmarks is None -- the SDK's internal "
        "BlazePalm detector + hand-landmark model jointly return no hand for the "
        "frame. min_detection_confidence=0.01 is the only externally-set "
        "threshold; everything else is MediaPipe's own internal graph."
    ),
    'zmurez_blazehand': (
        "BlazeHandLandmark (student_model/blazehand_landmark.pth), run on the "
        "full 256x256 crop through an IDENTITY roi "
        "(extract_roi(img, xc=128, yc=128, theta=0, scale=256)) -- BlazePalm is "
        "never constructed or called anywhere in this path. Failure iff "
        "hand_flag <= 0.5, where hand_flag = sigmoid(hand_flag_head(x)) is "
        "BlazeHandLandmark's OWN presence output "
        "(student_model/blazehand_landmark.py: self.hand_flag = "
        "nn.Conv2d(288, 1, 2); forward() returns hand_flag.view(-1).sigmoid()). "
        "This is a model-internal confidence threshold, NOT a BlazePalm "
        "rejection -- BlazePalm is absent from this path entirely."
    ),
    'student': (
        "BlazeHandLandmark (checkpoints/distilled_v2_ft_epoch_150.pth), fed the "
        "raw 256x256 crop tensor directly -- no extract_roi, no BlazePalm. "
        "Failure iff hand_flag <= 0.5, the same presence head as the zmurez "
        "weights (identical architecture, distilled parameters). Empirically "
        "this never fires on the RHD test set -- see detection_rate_pct below."
    ),
}


# ── Dataset ──────────────────────────────────────────────────────────────────
def get_loader():
    pipeline = [
        dict(type='LoadImage'),
        dict(type='GetBBoxCenterScale'),
        dict(type='TopdownAffine', input_size=(256, 256)),
        # pack_transformed=True: TopdownAffine writes the crop-warped
        # keypoints to results['transformed_keypoints'] and leaves
        # results['keypoints'] in original, un-warped image space. Every
        # runner below predicts in the warped 256x256 crop space, so the
        # GT reference must be transformed_keypoints, not keypoints.
        dict(type='PackPoseInputs', pack_transformed=True),
    ]
    ds = Rhd2DDataset(data_root=RHD_ROOT, ann_file=ANN_FILE, pipeline=pipeline)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=pseudo_collate)
    return ds, loader


# ── Per-system inference runners ────────────────────────────────────────────
# Each returns a dict of (N,21,2) gt, (N,21,2) pred, (N,) bool detected,
# (N,) float score (NaN where the system exposes no scalar score).
# Preprocessing intentionally replicates each system's own established
# eval script verbatim (mediapipe_vs_blazehand_eval.py for the first two,
# distilled_v2_ft_ep150_gt_comprehensive_eval.py for the student) so the
# detection-rate/PCK figures stay consistent with prior reported numbers.

def run_mediapipe_sdk(N, device):
    import mediapipe as mp

    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.01,
        model_complexity=1,
    )
    _, loader = get_loader()

    gt = np.zeros((N, 21, 2), dtype=np.float32)
    pred = np.zeros((N, 21, 2), dtype=np.float32)
    detected = np.zeros(N, dtype=bool)
    score = np.full(N, np.nan, dtype=np.float32)

    for idx, batch in enumerate(tqdm(loader, total=N, desc='MediaPipe SDK')):
        if idx >= N:
            break
        img_tensor = torch.stack(batch['inputs']).float() / 255.0
        gt_kpts = np.clip(np.array(batch['data_samples'][0].gt_instances.transformed_keypoints[0]),
                           0, 255).astype(np.float32)
        gt[idx] = gt_kpts[MP_TO_RHD]

        img_np = (img_tensor[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        result = mp_hands.process(img_rgb)
        if result.multi_hand_landmarks:
            detected[idx] = True
            lm = result.multi_hand_landmarks[0].landmark
            pred[idx] = np.array([[p.x * 256, p.y * 256] for p in lm])
            if result.multi_handedness:
                score[idx] = result.multi_handedness[0].classification[0].score

    mp_hands.close()
    return dict(gt=gt, pred=pred, detected=detected, score=score)


def run_zmurez_blazehand(N, device):
    # Import locally and prove BlazePalm never enters this path -- this is
    # the crux of the Section V-A contradiction (see module docstring).
    assert 'blazepalm' not in sys.modules, (
        'blazepalm was already imported before the zmurez BlazeHandLandmark '
        'run started -- invalidates the "BlazePalm absent from this path" claim.'
    )
    from blazehand_landmark import BlazeHandLandmark

    model = BlazeHandLandmark().to(device)
    model.load_weights(ZMUREZ_WEIGHTS)
    model.eval()

    assert 'blazepalm' not in sys.modules, (
        'blazepalm got imported as a side effect of loading BlazeHandLandmark '
        '-- invalidates the "BlazePalm absent from this path" claim.'
    )

    _, loader = get_loader()

    gt = np.zeros((N, 21, 2), dtype=np.float32)
    pred = np.zeros((N, 21, 2), dtype=np.float32)
    detected = np.zeros(N, dtype=bool)
    score = np.full(N, np.nan, dtype=np.float32)

    xc = torch.tensor([128.0])
    yc = torch.tensor([128.0])
    scale = torch.tensor([256.0])
    theta = torch.tensor([0.0])

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, total=N, desc='zmurez BlazeHandLandmark')):
            if idx >= N:
                break
            img_tensor = torch.stack(batch['inputs']).float() / 255.0
            gt_kpts = np.clip(np.array(batch['data_samples'][0].gt_instances.transformed_keypoints[0]),
                               0, 255).astype(np.float32)
            gt[idx] = gt_kpts[MP_TO_RHD]

            img_np = (img_tensor[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

            img_roi, affine, _ = model.extract_roi(img_rgb, xc, yc, theta, scale)
            flags, _, norm_lm = model(img_roi.to(device))

            score[idx] = flags[0].item()
            if flags[0].item() > 0.5:
                detected[idx] = True
                landmarks = model.denormalize_landmarks(norm_lm.cpu(), affine)
                pred[idx] = landmarks[0, :, :2].numpy()

    assert 'blazepalm' not in sys.modules, (
        'blazepalm got imported during zmurez BlazeHandLandmark inference '
        '-- invalidates the "BlazePalm absent from this path" claim.'
    )
    return dict(gt=gt, pred=pred, detected=detected, score=score)


def run_student(N, device):
    from blazehand_landmark import BlazeHandLandmark

    model = BlazeHandLandmark().to(device)
    model.load_state_dict(torch.load(STUDENT_WEIGHTS, map_location=device))
    model.eval()

    _, loader = get_loader()

    gt = np.zeros((N, 21, 2), dtype=np.float32)
    pred = np.zeros((N, 21, 2), dtype=np.float32)
    detected = np.zeros(N, dtype=bool)
    score = np.full(N, np.nan, dtype=np.float32)

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, total=N, desc='Our student')):
            if idx >= N:
                break
            img_tensor = torch.stack(batch['inputs']).float() / 255.0
            gt_kps = batch['data_samples'][0].gt_instances.transformed_keypoints
            gt[idx] = np.clip(np.array(gt_kps).reshape(21, 2), 0, 255).astype(np.float32)

            flag, _, out = model(img_tensor.to(device))
            score[idx] = flag.item()
            if flag.item() > 0.5:
                detected[idx] = True
            pred[idx] = out[0, :, :2].cpu().numpy() * 256

    return dict(gt=gt, pred=pred, detected=detected, score=score)


RUNNERS = {
    'mediapipe_sdk': run_mediapipe_sdk,
    'zmurez_blazehand': run_zmurez_blazehand,
    'student': run_student,
}


# ── Cache I/O ────────────────────────────────────────────────────────────────
def cache_path(name, n):
    return os.path.join(CACHE_DIR, f'{name}_n{n}.npz')


def load_or_run(name, N, device, force):
    path = cache_path(name, N)
    if os.path.exists(path) and not force:
        print(f'[{name}] loading cached predictions -> {path}')
        with np.load(path) as z:
            return dict(gt=z['gt'], pred=z['pred'], detected=z['detected'], score=z['score'])

    print(f'[{name}] running inference on {N} samples (no cache found, or --force)')
    data = RUNNERS[name](N, device)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(path, gt=data['gt'], pred=data['pred'], detected=data['detected'], score=data['score'])
    print(f'[{name}] cached raw predictions -> {path}')
    return data


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(dists):
    """dists: (n, 21) per-joint pixel distances. Returns PCK@0.2 / MPJPE / MSE / AUC."""
    n = dists.shape[0]
    if n == 0:
        return dict(pck=float('nan'), mpjpe=float('nan'), mse=float('nan'), auc=float('nan'), n=0)
    correct = (dists <= THRESH_PX).sum()
    pck = correct / (n * 21)
    mpjpe = float(dists.mean())
    mse = float(np.mean(np.sum(dists ** 2, axis=1) / 21))
    auc_vals = [(dists <= t).mean() for t in AUC_THRESHOLDS]
    auc = float(np.trapz(auc_vals, AUC_THRESHOLDS / (0.5 * 256)))
    return dict(pck=float(pck), mpjpe=mpjpe, mse=mse, auc=auc, n=n)


def score_system(name, data):
    gt, pred, detected = data['gt'], data['pred'], data['detected']
    n = gt.shape[0]
    dists = np.linalg.norm(pred - gt, axis=2)  # (N, 21)

    dists_penalized = dists.copy()
    dists_penalized[~detected] = FAIL_PENALTY_PX
    all_frames = compute_metrics(dists_penalized)

    detected_only = compute_metrics(dists[detected])

    return dict(
        system=name,
        n_total=n,
        n_detected=int(detected.sum()),
        n_failed=int((~detected).sum()),
        detection_rate_pct=float(detected.mean() * 100),
        criterion=FAILURE_CRITERIA[name],
        all_frames=all_frames,
        detected_only=detected_only,
    )


# ── Reporting ────────────────────────────────────────────────────────────────
def print_report(rows):
    SEP = '=' * 100
    print()
    print(SEP)
    print('  DETECTION-AWARE RESCORING  (same saved predictions -> two metric sets)')
    print(SEP)
    for r in rows:
        print(f"\n  {r['system']}")
        print(f"    Detection rate : {r['detection_rate_pct']:.2f}%  "
              f"({r['n_detected']}/{r['n_total']} detected, {r['n_failed']} failed)")
        print(f"    Criterion      : {r['criterion']}")
        af, do = r['all_frames'], r['detected_only']
        print(f"    {'':16s} {'PCK@0.2':>10} {'MPJPE(px)':>11} {'MSE(px2)':>12} {'AUC':>8}   n")
        print(f"    {'all-frames':16s} {af['pck']:>10.4f} {af['mpjpe']:>11.4f} "
              f"{af['mse']:>12.4f} {af['auc']:>8.4f}   {af['n']}")
        print(f"    {'detected-only':16s} {do['pck']:>10.4f} {do['mpjpe']:>11.4f} "
              f"{do['mse']:>12.4f} {do['auc']:>8.4f}   {do['n']}")
    print()
    print(SEP)

    zmurez = next((r for r in rows if r['system'] == 'zmurez_blazehand'), None)
    if zmurez:
        print('  CONTRADICTION RESOLUTION (for Section V-A)')
        print(SEP)
        print(
            "  Section V-A currently says the zmurez weights run directly on pre-cropped\n"
            "  256x256 inputs \"bypassing the BlazePalm stage\", then defines detection\n"
            "  failure as \"BlazePalm rejecting a crop\". If BlazePalm is not in that path,\n"
            "  it cannot reject anything -- yet Table V assigns those weights a "
            f"{100 - zmurez['detection_rate_pct']:.2f}%\n"
            "  failure rate. Verified by direct inspection of the eval code path\n"
            "  (run_zmurez_blazehand in this script == mediapipe_vs_blazehand_eval.py):\n"
            "  BlazePalm is never imported or constructed (asserted at runtime above);\n"
            "  only BlazeHandLandmark runs, fed through an IDENTITY roi. The failure\n"
            f"  count ({zmurez['n_failed']}/{zmurez['n_total']} = "
            f"{100 - zmurez['detection_rate_pct']:.2f}%) comes entirely from "
            "BlazeHandLandmark's own\n"
            "  hand_flag presence output.\n"
        )
        print('  Criterion string to quote verbatim in Section V-A:')
        print(f"  \"{zmurez['criterion']}\"")
        print(SEP)


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        'system', 'n_total', 'n_detected', 'n_failed', 'detection_rate_pct',
        'failure_criterion',
        'all_frames_pck', 'all_frames_mpjpe_px', 'all_frames_mse_px2', 'all_frames_auc',
        'detected_only_pck', 'detected_only_mpjpe_px', 'detected_only_mse_px2', 'detected_only_auc',
        'detected_only_n',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            af, do = r['all_frames'], r['detected_only']
            writer.writerow({
                'system': r['system'],
                'n_total': r['n_total'],
                'n_detected': r['n_detected'],
                'n_failed': r['n_failed'],
                'detection_rate_pct': f"{r['detection_rate_pct']:.4f}",
                'failure_criterion': r['criterion'],
                'all_frames_pck': f"{af['pck']:.6f}",
                'all_frames_mpjpe_px': f"{af['mpjpe']:.6f}",
                'all_frames_mse_px2': f"{af['mse']:.6f}",
                'all_frames_auc': f"{af['auc']:.6f}",
                'detected_only_pck': f"{do['pck']:.6f}",
                'detected_only_mpjpe_px': f"{do['mpjpe']:.6f}",
                'detected_only_mse_px2': f"{do['mse']:.6f}",
                'detected_only_auc': f"{do['auc']:.6f}",
                'detected_only_n': do['n'],
            })
    print(f'\nSaved: {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--samples', type=int, default=None,
                     help='limit to first N samples (default: full 2727-sample RHD test set)')
    ap.add_argument('--force', action='store_true', help='ignore cached predictions and re-run inference')
    ap.add_argument('--systems', nargs='+', choices=list(RUNNERS), default=list(RUNNERS),
                     help='subset of systems to (re)score')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    ds, _ = get_loader()
    N = len(ds) if args.samples is None else min(args.samples, len(ds))
    print(f'Samples: {N} (full test set = {len(ds)})\n')

    torch.set_grad_enabled(False)

    rows = []
    for name in args.systems:
        data = load_or_run(name, N, device, args.force)
        rows.append(score_system(name, data))

    print_report(rows)
    write_csv(rows, OUT_CSV)


if __name__ == '__main__':
    main()
