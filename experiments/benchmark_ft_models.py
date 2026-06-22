"""
benchmark_ft_models.py
======================
Compares two fine-tuned distilled student (BlazeHandLandmark) checkpoints:

  Model A — RHD FT (ep.150)   : checkpoints/distilled_v2_ft_epoch_150.pth
              Distillation (100 ep.) → fine-tuned on RHD for 50 ep. → epoch 150
  Model B — FreiHAND FT (ep.50): checkpoints/freihand_ft_epoch_50.pth
              Continues from Model A → fine-tuned on FreiHAND for 50 ep.

Metrics reported
----------------
  1. Parameter count     — sum(p.numel()) over all / trainable params
  2. Model size (disk)   — .pth file size in MB
  3. Median infer. time  — median over N_RUNS timed single-image passes
  4. Inference speed     — 1 / median_time  (FPS)
  5. Detection rate      — % samples where hand_flag >= 0.5
                           Model A → RHD test set
                           Model B → FreiHAND validation split (5 %)

Timing notes
------------
  Warm-up (N_WARMUP passes on a dummy tensor) is done before timing to avoid
  cold-start bias from JIT/CUDA kernel compilation.  GPU synchronisation
  (torch.cuda.synchronize) surrounds each timed call so GPU-side latency is
  captured rather than just CPU-dispatch time.
  Median is used instead of mean because it is robust to occasional OS
  scheduling spikes that inflate the tail of the distribution.

No external citation is required for these standard metrics; the timing
methodology follows common PyTorch micro-benchmarking practice (see also the
PyTorch documentation for torch.utils.benchmark.Timer if finer analysis is
needed in future work).
"""

import sys, os, time
import numpy as np
import torch

# ── student model ─────────────────────────────────────────────────────────────
sys.path.append(os.path.join(os.getcwd(), 'student_model'))
from blazehand_landmark import BlazeHandLandmark

# ── MMPose (for RHD dataset) ──────────────────────────────────────────────────
from mmengine.registry import init_default_scope
from mmengine.dataset import pseudo_collate
from mmpose.datasets import Rhd2DDataset
from torch.utils.data import DataLoader
init_default_scope('mmpose')

# ── FreiHAND dataset ──────────────────────────────────────────────────────────
from freihand_dataset import make_train_val_loaders

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CKPT_RHD       = 'checkpoints/distilled_v2_ft_epoch_150.pth'
CKPT_FREIHAND  = 'checkpoints/freihand_ft_epoch_50.pth'

RHD_ROOT       = r'C:\Users\Bea Juliana Poquiz\Desktop\mmpose_thesis\dataset\rhd'
FREIHAND_ROOT  = os.path.join(os.getcwd(), 'freihand')

N_WARMUP = 50    # dummy passes before timing (GPU warm-up / JIT)
N_RUNS   = 500   # timed passes per model

DET_THRESHOLD = 0.5   # hand_flag threshold for "detected"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def file_size_mb(path):
    return os.path.getsize(path) / (1024 ** 2)


def benchmark_speed(model, device, n_warmup=N_WARMUP, n_runs=N_RUNS):
    """Returns (median_ms, fps, all_times_array)."""
    dummy = torch.randn(1, 3, 256, 256, device=device)
    use_cuda = (device.type == 'cuda')

    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)
            if use_cuda:
                torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy)
            if use_cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    times = np.array(times)
    median_ms = float(np.median(times)) * 1000.0
    fps       = 1000.0 / median_ms
    return median_ms, fps, times


def detection_rate_rhd(model, device):
    """% of RHD test images where model outputs hand_flag >= DET_THRESHOLD."""
    pipeline = [
        dict(type='LoadImage'),
        dict(type='GetBBoxCenterScale'),
        dict(type='TopdownAffine', input_size=(256, 256)),
        dict(type='PackPoseInputs'),
    ]
    ds     = Rhd2DDataset(data_root=RHD_ROOT,
                          ann_file='annotations/rhd_test.json',
                          pipeline=pipeline)
    loader = DataLoader(ds, batch_size=1, collate_fn=pseudo_collate)
    N      = len(ds)
    flags  = []
    with torch.no_grad():
        for batch in loader:
            img = torch.stack(batch['inputs']).float().to(device) / 255.0
            flag, _, _ = model(img)
            flags.append(float(flag.item()))
    flags = np.array(flags)
    return float((flags >= DET_THRESHOLD).mean() * 100.0), N


def detection_rate_freihand(model, device):
    """% of FreiHAND val images where model outputs hand_flag >= DET_THRESHOLD."""
    _, val_loader = make_train_val_loaders(
        FREIHAND_ROOT, input_size=256, val_frac=0.05, batch_size=1)
    N     = len(val_loader.dataset)
    flags = []
    with torch.no_grad():
        for img, _ in val_loader:
            img = img.to(device)           # already in [0, 1]
            flag, _, _ = model(img)
            flags.append(float(flag.item()))
    flags = np.array(flags)
    return float((flags >= DET_THRESHOLD).mean() * 100.0), N

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice : {device}')
print(f'PyTorch: {torch.__version__}\n')

model_rhd = BlazeHandLandmark().to(device)
model_rhd.load_state_dict(torch.load(CKPT_RHD, map_location=device))
model_rhd.eval()

model_freihand = BlazeHandLandmark().to(device)
model_freihand.load_state_dict(torch.load(CKPT_FREIHAND, map_location=device))
model_freihand.eval()

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER COUNT & MODEL SIZE
# ─────────────────────────────────────────────────────────────────────────────

total_a, train_a = count_params(model_rhd)
total_b, train_b = count_params(model_freihand)

size_a = file_size_mb(CKPT_RHD)
size_b = file_size_mb(CKPT_FREIHAND)

# ─────────────────────────────────────────────────────────────────────────────
# SPEED BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

print(f'Speed benchmark  ({N_WARMUP} warm-up + {N_RUNS} timed runs, single image)...')
print('  Running Model A (RHD FT ep.150)...')
med_a, fps_a, times_a = benchmark_speed(model_rhd, device)

print('  Running Model B (FreiHAND FT ep.50)...')
med_b, fps_b, times_b = benchmark_speed(model_freihand, device)

# ─────────────────────────────────────────────────────────────────────────────
# DETECTION RATE
# ─────────────────────────────────────────────────────────────────────────────

print('\nDetection rate — Model A on RHD test set...')
det_a, n_a = detection_rate_rhd(model_rhd, device)

print('Detection rate — Model B on FreiHAND val split...')
det_b, n_b = detection_rate_freihand(model_freihand, device)

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────

W = 60
print()
print('=' * W)
print(' BENCHMARK — Distilled Student Fine-Tuned Models')
print('=' * W)
print(f'  {"Metric":<30}  {"RHD FT (ep.150)":>12}  {"FreiHAND FT (ep.50)":>19}')
print('-' * W)
print(f'  {"Checkpoint":<30}  {"ep.150 (RHD)":>12}  {"ep.50 (FreiHAND)":>19}')
print(f'  {"Architecture":<30}  {"BlazeHandLandmark":>12}  {"BlazeHandLandmark":>19}')
print('-' * W)
print(f'  {"Total parameters":<30}  {total_a:>12,}  {total_b:>19,}')
print(f'  {"Trainable parameters":<30}  {train_a:>12,}  {train_b:>19,}')
print(f'  {"Model size (disk)":<30}  {size_a:>11.2f}M  {size_b:>18.2f}M')
print('-' * W)
print(f'  {"Median inference time":<30}  {med_a:>10.3f}ms  {med_b:>17.3f}ms')
print(f'  {"Inference speed":<30}  {fps_a:>10.1f}fps  {fps_b:>17.1f}fps')
print(f'  {"Min inference time":<30}  {np.min(times_a)*1000:>10.3f}ms  {np.min(times_b)*1000:>17.3f}ms')
print(f'  {"Max inference time":<30}  {np.max(times_a)*1000:>10.3f}ms  {np.max(times_b)*1000:>17.3f}ms')
print(f'  {"Std dev inference time":<30}  {np.std(times_a)*1000:>10.3f}ms  {np.std(times_b)*1000:>17.3f}ms')
print('-' * W)
print(f'  {"Detection rate":<30}  {det_a:>11.2f}%  {det_b:>18.2f}%')
print(f'  {"  (eval dataset)":<30}  {"RHD test":>12}  {"FreiHAND val":>19}')
print(f'  {"  (# samples evaluated)":<30}  {n_a:>12,}  {n_b:>19,}')
print(f'  {"  (threshold)":<30}  {"hand_flag≥0.5":>12}  {"hand_flag≥0.5":>19}')
print('=' * W)
print()
print('Notes:')
print(f'  * Timing device  : {device}')
print(f'  * Timing input   : random tensor (1 × 3 × 256 × 256)')
print(f'  * Warm-up passes : {N_WARMUP}  |  Timed passes: {N_RUNS}')
print( '  * Median used for robustness against OS scheduling outliers.')
print( '  * Both models share the same BlazeHandLandmark architecture;')
print( '    parameter counts and model sizes are expected to be identical.')
print( '  * Detection rate reflects the model\'s hand_flag output confidence')
print( '    (>= 0.5) and is evaluated on each model\'s primary test set.')
print()
