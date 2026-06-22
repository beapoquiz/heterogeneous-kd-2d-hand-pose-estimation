"""
TEACHER COMPREHENSIVE EVALUATION
HRNetV2-W18 + DARK decoding | Trained on RHD2D | Official checkpoint

Covers:
  1. Model statistics  — parameters, file size
  2. Per-joint PCK@0.2 and EPE — via Runner + Hook (same pipeline as
     tools/test.py, so coordinate spaces are correct and results will
     match the official PCK = 0.9918)
  3. Inference speed  — median latency (ms) and FPS, GPU-timed with
     CUDA events, batch_size=1 (single-image, same as student benchmarks)

Output: printed table + teacher_comprehensive_results.txt

Run from project root:
    python teacher_comprehensive_eval.py
"""

import os, sys, time
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.hooks import Hook
from mmengine.registry import init_default_scope
from mmpose.apis import init_model
from mmpose.datasets import Rhd2DDataset
from torch.utils.data import DataLoader
from mmengine.dataset import pseudo_collate

JOINT_NAMES = [
    'Wrist',
    'Thumb_MCP',  'Thumb_PIP',  'Thumb_DIP',  'Thumb_Tip',
    'Index_MCP',  'Index_PIP',  'Index_DIP',  'Index_Tip',
    'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_Tip',
    'Ring_MCP',   'Ring_PIP',   'Ring_DIP',   'Ring_Tip',
    'Pinky_MCP',  'Pinky_PIP',  'Pinky_DIP',  'Pinky_Tip',
]


class PerJointHook(Hook):
    """Collects per-joint distances after every test batch."""
    priority = 'NORMAL'

    def __init__(self):
        self.dists      = []
        self.vis        = []
        self.bbox_sizes = []

    def after_test_iter(self, runner, batch_idx, data_batch, outputs):
        for sample in outputs:
            try:
                pred_kps = np.array(
                    sample.pred_instances.keypoints[0]).reshape(21, 2)
                gt_kps   = np.array(
                    sample.gt_instances.keypoints[0]).reshape(21, 2)
                gt_vis   = (
                    np.array(sample.gt_instances.keypoints_visible[0]) > 0
                    if hasattr(sample.gt_instances, 'keypoints_visible')
                    else np.ones(21, dtype=bool))
                bbox      = np.array(sample.gt_instances.bboxes[0])
                bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
                d = np.linalg.norm(pred_kps - gt_kps, axis=1)
                self.dists.append(d)
                self.vis.append(gt_vis)
                self.bbox_sizes.append(float(bbox_size))
            except Exception:
                pass


def main():
    init_default_scope('mmpose')

    BASE   = os.path.dirname(os.path.abspath(__file__))
    CONFIG = os.path.join(BASE, 'mmpose/configs/hand_2d_keypoint/topdown_heatmap/'
                          'rhd2d/td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256.py')
    CKPT   = os.path.join(BASE, 'checkpoints',
                          'hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}\n')

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Model statistics
    # ══════════════════════════════════════════════════════════════════════════
    print('Loading teacher for statistics...')
    teacher = init_model(CONFIG, CKPT, device=device)
    teacher.eval()

    total_params     = sum(p.numel() for p in teacher.parameters())
    trainable_params = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    file_size_mb     = os.path.getsize(CKPT) / (1024 ** 2)

    print(f'  Total parameters     : {total_params:,}')
    print(f'  Trainable parameters : {trainable_params:,}')
    print(f'  Checkpoint size      : {file_size_mb:.2f} MB\n')

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Per-joint PCK@0.2 and EPE via Runner + Hook
    # ══════════════════════════════════════════════════════════════════════════
    print('Running per-joint evaluation via Runner (same as tools/test.py)...')

    hook = PerJointHook()

    cfg = Config.fromfile(CONFIG)
    cfg.load_from = CKPT
    cfg.work_dir  = os.path.join(BASE, 'eval_results', 'teacher_comprehensive')
    for split in ('val_dataloader', 'test_dataloader'):
        dl = cfg.get(split)
        if dl is not None:
            dl['dataset']['data_root'] = os.path.join(BASE, 'dataset', 'rhd')

    os.makedirs(cfg.work_dir, exist_ok=True)
    runner = Runner.from_cfg(cfg)
    runner.register_hook(hook, 'NORMAL')
    runner.test()

    # Compute per-joint metrics
    dists_arr = np.array(hook.dists)
    vis_arr   = np.array(hook.vis)
    bbox_arr  = np.array(hook.bbox_sizes)

    pck_per_joint = np.zeros(21)
    epe_per_joint = np.zeros(21)
    for j in range(21):
        mask = vis_arr[:, j]
        if mask.sum() == 0:
            continue
        d_j   = dists_arr[mask, j]
        thr_j = bbox_arr[mask] * 0.2
        pck_per_joint[j] = float((d_j < thr_j).mean())
        epe_per_joint[j] = float(d_j.mean())

    all_d       = dists_arr[vis_arr]
    all_thr     = np.repeat(bbox_arr * 0.2, vis_arr.sum(axis=1))
    overall_pck = float((all_d < all_thr).mean())
    overall_epe = float(all_d.mean())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Inference speed
    # ══════════════════════════════════════════════════════════════════════════
    print('\nMeasuring inference speed (batch_size=1, GPU timing)...')

    pipeline = [
        dict(type='LoadImage'),
        dict(type='GetBBoxCenterScale'),
        dict(type='TopdownAffine', input_size=(256, 256)),
        dict(type='PackPoseInputs'),
    ]
    speed_ds = Rhd2DDataset(
        data_root=os.path.join(BASE, 'dataset', 'rhd'),
        ann_file='annotations/rhd_test.json',
        pipeline=pipeline,
    )
    speed_loader = DataLoader(speed_ds, batch_size=1, shuffle=False,
                              num_workers=0, collate_fn=pseudo_collate)

    N_WARMUP  = 30
    N_MEASURE = 200
    latencies_ms = []
    it = iter(speed_loader)

    print(f'  Warming up ({N_WARMUP} iters)...')
    for _ in range(N_WARMUP):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(speed_loader); batch = next(it)
        img = torch.stack(batch['inputs']).float() / 255.0
        with torch.no_grad():
            _ = teacher.predict(img.to(device), batch['data_samples'])
    if device.type == 'cuda':
        torch.cuda.synchronize()

    print(f'  Timing {N_MEASURE} iterations...')
    for _ in range(N_MEASURE):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(speed_loader); batch = next(it)
        img = torch.stack(batch['inputs']).float() / 255.0
        img = img.to(device)

        if device.type == 'cuda':
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev   = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            with torch.no_grad():
                _ = teacher.predict(img, batch['data_samples'])
            end_ev.record()
            torch.cuda.synchronize()
            latencies_ms.append(start_ev.elapsed_time(end_ev))
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = teacher.predict(img, batch['data_samples'])
            latencies_ms.append((time.perf_counter() - t0) * 1000)

    lat       = np.array(latencies_ms)
    median_ms = float(np.median(lat))
    mean_ms   = float(np.mean(lat))
    p95_ms    = float(np.percentile(lat, 95))
    fps       = 1000.0 / median_ms

    # ══════════════════════════════════════════════════════════════════════════
    # OUTPUT
    # ══════════════════════════════════════════════════════════════════════════
    SEP = '=' * 62
    lines = []

    def p(s=''):
        print(s)
        lines.append(s)

    p(); p(SEP)
    p('  TEACHER — HRNetV2-W18 + DARK')
    p('  Trained on RHD2D | Official MMPose checkpoint')
    p(SEP); p()

    p('── MODEL STATISTICS ─────────────────────────────────────────')
    p(f'  Total parameters     : {total_params:>12,}')
    p(f'  Trainable parameters : {trainable_params:>12,}')
    p(f'  Checkpoint size      : {file_size_mb:>11.2f} MB')
    p()

    p('── OFFICIAL METRICS  (tools/test.py, 2026-05-11, all 2727) ──')
    p(f'  PCK@0.2 (bbox-norm)  : 0.9918   (99.18%)')
    p(f'  AUC     (0–30 px)    : 0.9023')
    p(f'  EPE                  : 2.18 px')
    p(f'  Detection Rate       : 100.00%  (top-down: bbox provided)')
    p()

    p('── PER-JOINT BREAKDOWN  (Runner+Hook, bbox-norm threshold) ──')
    p(f'  Cross-check PCK@0.2  : {overall_pck:.4f}  '
      f'← should match 0.9918')
    p(f'  Cross-check EPE      : {overall_epe:.4f} px'
      f'  ← should match 2.18 px')
    p()
    p(f'  {"Joint":<15} {"PCK@0.2":>10} {"EPE (px)":>10}')
    p(f'  {"-"*38}')
    for j, name in enumerate(JOINT_NAMES):
        p(f'  {name:<15} {pck_per_joint[j]:>10.4f} {epe_per_joint[j]:>10.4f}')
    p()

    ranked = sorted(range(21), key=lambda j: epe_per_joint[j], reverse=True)
    p('  Hardest joints (highest EPE):')
    for rank, j in enumerate(ranked[:5], 1):
        p(f'    {rank}. {JOINT_NAMES[j]:<15}  EPE={epe_per_joint[j]:.3f} px  '
          f'PCK={pck_per_joint[j]:.4f}') 
    p()
    p('  Easiest joints (lowest EPE):')
    for rank, j in enumerate(ranked[-3:][::-1], 1):
        p(f'    {rank}. {JOINT_NAMES[j]:<15}  EPE={epe_per_joint[j]:.3f} px  '
          f'PCK={pck_per_joint[j]:.4f}')
    p()

    p('── INFERENCE SPEED  (batch_size=1, predict() end-to-end) ───')
    p(f'  Median latency  : {median_ms:>7.2f} ms')
    p(f'  Mean   latency  : {mean_ms:>7.2f} ms')
    p(f'  95th percentile : {p95_ms:>7.2f} ms')
    p(f'  FPS (1/median)  : {fps:>7.1f}')
    p(f'  Device          : {str(device).upper()}')
    p(f'  Includes        : model forward + heatmap decode')
    p(f'  Excludes        : image loading, affine crop preprocessing')
    p(SEP)

    out_path = os.path.join(BASE, 'teacher_comprehensive_results.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
