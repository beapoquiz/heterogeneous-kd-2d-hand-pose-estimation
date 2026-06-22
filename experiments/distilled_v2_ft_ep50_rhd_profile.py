"""
distilled_v2_ft_ep50_rhd_profile.py
=====================================
Comprehensive profile of the distilled student model finetuned on RHD for 50
epochs (checkpoint: distilled_v2_ft_epoch_150.pth — distillation ran to ep.100,
then 50 RHD fine-tuning epochs to reach ep.150).

Metrics reported
----------------
  1.  Architecture summary   — class name, input size, output heads
  2.  Parameter count        — total / trainable
  3.  Checkpoint file size   — .pth on disk (MB)
  4.  Model weight size      — float32 parameter tensor footprint (MB)
  5.  GPU inference speed    — mean, median, min, max, std, FPS (N_RUNS timed
                               passes after N_WARMUP warm-up)
  6.  CPU inference speed    — same statistics
  7.  Multi-batch throughput — FPS at batch sizes 1, 4, 8, 16 (GPU if available)
  8.  GPU memory usage       — peak allocated / reserved during one forward pass
  9.  Output shape report    — shapes of all three output tensors
  10. Detection rate (RHD)   — % of 2 727 test samples with hand_flag >= 0.5

Results are printed to stdout and saved to:
  results_distilled_v2_ft_ep50_rhd_profile.txt
"""

import sys, os, time
import numpy as np
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE, 'student_model'))
from blazehand_landmark import BlazeHandLandmark

from mmengine.registry import init_default_scope
from mmengine.dataset import pseudo_collate
from mmpose.datasets import Rhd2DDataset
from torch.utils.data import DataLoader
init_default_scope('mmpose')

# ── config ────────────────────────────────────────────────────────────────────
CKPT          = os.path.join(BASE, 'checkpoints', 'distilled_v2_ft_epoch_150.pth')
RHD_ROOT      = os.path.join(BASE, 'dataset', 'rhd')
INPUT_SIZE    = (1, 3, 256, 256)
N_WARMUP      = 50
N_RUNS        = 500
DET_THRESHOLD = 0.5
BATCH_SIZES   = [1, 4, 8, 16]
OUT_FILE      = os.path.join(BASE, 'results_distilled_v2_ft_ep50_rhd_profile.txt')
SEP           = '=' * 62
SEP2          = '-' * 62

# ── helpers ───────────────────────────────────────────────────────────────────
def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()

def timed_runs(model, dummy, device, n_warmup=N_WARMUP, n_runs=N_RUNS):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)
            sync(device)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            sync(device)
            t0 = time.perf_counter()
            model(dummy)
            sync(device)
            times.append((time.perf_counter() - t0) * 1000.0)
    return np.array(times)

def batch_fps(model, device, batch_size, n_warmup=20, n_runs=100):
    dummy = torch.randn(batch_size, 3, 256, 256, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)
            sync(device)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            sync(device)
            t0 = time.perf_counter()
            model(dummy)
            sync(device)
            times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return batch_size * 1000.0 / np.median(arr)  # images/sec

def detection_rate_rhd(model, device):
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
    model.eval()
    with torch.no_grad():
        for batch in loader:
            img = torch.stack(batch['inputs']).float().to(device) / 255.0
            flag, _, _ = model(img)
            flags.append(float(flag.item()))
            if (len(flags)) % 500 == 0:
                print(f'    [{len(flags)}/{N}] running...')
    flags = np.array(flags)
    rate  = float((flags >= DET_THRESHOLD).mean() * 100.0)
    return rate, N, flags

# ─────────────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(SEP)
print(' Distilled Student — RHD Fine-Tune (50 ep.) — Comprehensive Profile')
print(SEP)
print(f'  Checkpoint : {CKPT}')
print(f'  Device     : {device}')
if device.type == 'cuda':
    print(f'  GPU        : {torch.cuda.get_device_name(0)}')
    print(f'  CUDA       : {torch.version.cuda}')
print(f'  PyTorch    : {torch.__version__}')
print()

# ── load model ────────────────────────────────────────────────────────────────
print('Loading checkpoint...')
model = BlazeHandLandmark().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

# ── 1. architecture summary ───────────────────────────────────────────────────
print(SEP)
print('  1. ARCHITECTURE')
print(SEP2)
arch_name = model.__class__.__name__
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params    = total_params - trainable_params
weight_mb        = total_params * 4 / 1024 / 1024     # float32
disk_mb          = os.path.getsize(CKPT) / 1024 / 1024

# probe output shapes
dummy1 = torch.zeros(*INPUT_SIZE, device=device)
with torch.no_grad():
    out = model(dummy1)
hand_flag, landmarks_2d, landmarks_3d = out
out_shapes = {
    'hand_flag':     tuple(hand_flag.shape),
    'landmarks_2d':  tuple(landmarks_2d.shape),
    'landmarks_3d':  tuple(landmarks_3d.shape),
}

print(f'  Architecture         : {arch_name}')
print(f'  Input shape          : {tuple(INPUT_SIZE)}  (B × C × H × W)')
print(f'  Output — hand_flag   : {out_shapes["hand_flag"]}')
print(f'  Output — landmarks_2d: {out_shapes["landmarks_2d"]}')
print(f'  Output — landmarks_3d: {out_shapes["landmarks_3d"]}')

# ── 2–4. parameter count & size ───────────────────────────────────────────────
print()
print(SEP)
print('  2–4. PARAMETERS & SIZE')
print(SEP2)
print(f'  Total parameters     : {total_params:>12,}')
print(f'  Trainable parameters : {trainable_params:>12,}')
print(f'  Frozen parameters    : {frozen_params:>12,}')
print(f'  Parameter count (M)  : {total_params/1e6:>12.4f} M')
print(f'  Weight size (float32): {weight_mb:>12.4f} MB')
print(f'  Checkpoint file size : {disk_mb:>12.4f} MB')

# ── 5. GPU inference speed ────────────────────────────────────────────────────
print()
print(SEP)
print('  5. GPU INFERENCE SPEED')
print(SEP2)
dummy_gpu = torch.randn(*INPUT_SIZE, device=device)
if device.type == 'cuda':
    print(f'  Warming up ({N_WARMUP} passes) then timing {N_RUNS} passes...')
    gpu_times = timed_runs(model, dummy_gpu, device)
    print(f'  Mean   : {gpu_times.mean():>9.3f} ms   ({1000/gpu_times.mean():>8.1f} FPS)')
    print(f'  Median : {np.median(gpu_times):>9.3f} ms   ({1000/np.median(gpu_times):>8.1f} FPS)')
    print(f'  Min    : {gpu_times.min():>9.3f} ms   ({1000/gpu_times.min():>8.1f} FPS)')
    print(f'  Max    : {gpu_times.max():>9.3f} ms   ({1000/gpu_times.max():>8.1f} FPS)')
    print(f'  Std    : {gpu_times.std():>9.3f} ms')
    print(f'  p95    : {np.percentile(gpu_times, 95):>9.3f} ms   ({1000/np.percentile(gpu_times, 95):>8.1f} FPS)')
    print(f'  p99    : {np.percentile(gpu_times, 99):>9.3f} ms   ({1000/np.percentile(gpu_times, 99):>8.1f} FPS)')
else:
    gpu_times = None
    print('  (no GPU — skipped)')

# ── 6. CPU inference speed ────────────────────────────────────────────────────
print()
print(SEP)
print('  6. CPU INFERENCE SPEED')
print(SEP2)
cpu_runs = min(N_RUNS, 100)
print(f'  Warming up (10 passes) then timing {cpu_runs} passes...')
model_cpu = BlazeHandLandmark().cpu()
model_cpu.load_state_dict(torch.load(CKPT, map_location='cpu'))
model_cpu.eval()
dummy_cpu = torch.randn(*INPUT_SIZE)
cpu_times = timed_runs(model_cpu, dummy_cpu, torch.device('cpu'),
                       n_warmup=10, n_runs=cpu_runs)
print(f'  Mean   : {cpu_times.mean():>9.3f} ms   ({1000/cpu_times.mean():>8.1f} FPS)')
print(f'  Median : {np.median(cpu_times):>9.3f} ms   ({1000/np.median(cpu_times):>8.1f} FPS)')
print(f'  Min    : {cpu_times.min():>9.3f} ms   ({1000/cpu_times.min():>8.1f} FPS)')
print(f'  Max    : {cpu_times.max():>9.3f} ms   ({1000/cpu_times.max():>8.1f} FPS)')
print(f'  Std    : {cpu_times.std():>9.3f} ms')
del model_cpu

# ── 7. multi-batch throughput (GPU) ──────────────────────────────────────────
print()
print(SEP)
print('  7. MULTI-BATCH THROUGHPUT (GPU)')
print(SEP2)
batch_results = {}
if device.type == 'cuda':
    for bs in BATCH_SIZES:
        fps = batch_fps(model, device, bs)
        batch_results[bs] = fps
        print(f'  Batch size {bs:>2} : {fps:>8.1f} images/sec')
else:
    print('  (no GPU — skipped)')

# ── 8. GPU memory usage ───────────────────────────────────────────────────────
print()
print(SEP)
print('  8. GPU MEMORY USAGE')
print(SEP2)
if device.type == 'cuda':
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        model(dummy_gpu)
    torch.cuda.synchronize()
    peak_alloc_mb   = torch.cuda.max_memory_allocated(device)   / 1024 / 1024
    peak_reserv_mb  = torch.cuda.max_memory_reserved(device)    / 1024 / 1024
    print(f'  Peak allocated  : {peak_alloc_mb:>8.2f} MB')
    print(f'  Peak reserved   : {peak_reserv_mb:>8.2f} MB')
else:
    peak_alloc_mb = peak_reserv_mb = None
    print('  (no GPU — skipped)')

# ── 9. output shape report ────────────────────────────────────────────────────
print()
print(SEP)
print('  9. OUTPUT SHAPE REPORT')
print(SEP2)
print(f'  hand_flag     shape : {out_shapes["hand_flag"]}   — scalar confidence [0,1]')
print(f'  landmarks_2d  shape : {out_shapes["landmarks_2d"]}  — (B, 21, 3) normalized [0,1] (x,y,z)')
print(f'  landmarks_3d  shape : {out_shapes["landmarks_3d"]}  — (B, 21, 3) 3-D coords')

# ── 10. detection rate on RHD test set ────────────────────────────────────────
print()
print(SEP)
print('  10. DETECTION RATE — RHD TEST SET')
print(SEP2)
print('  Running inference on all 2 727 RHD test samples...')
det_rate, n_rhd, det_flags = detection_rate_rhd(model, device)
det_count = int((det_flags >= DET_THRESHOLD).sum())
print(f'  Samples evaluated : {n_rhd:,}')
print(f'  Detected (flag≥{DET_THRESHOLD}) : {det_count:,}  /  {n_rhd:,}')
print(f'  Detection rate    : {det_rate:.4f} %')
print(f'  Mean hand_flag    : {det_flags.mean():.4f}')
print(f'  Std  hand_flag    : {det_flags.std():.4f}')

# ── final summary ─────────────────────────────────────────────────────────────
print()
print(SEP)
print('  SUMMARY')
print(SEP)
print(f'  Model             : {arch_name}')
print(f'  Checkpoint        : distilled_v2_ft_epoch_150.pth')
print(f'  Fine-tune epochs  : 50 (ep.101–150) on RHD')
print(f'  Parameters        : {total_params:,}  ({total_params/1e6:.4f} M)')
print(f'  Checkpoint size   : {disk_mb:.4f} MB  (disk)')
print(f'  Weight size       : {weight_mb:.4f} MB  (float32 tensors)')
if gpu_times is not None:
    print(f'  GPU latency       : {np.median(gpu_times):.3f} ms median  '
          f'({1000/np.median(gpu_times):.1f} FPS)')
print(f'  CPU latency       : {np.median(cpu_times):.3f} ms median  '
      f'({1000/np.median(cpu_times):.1f} FPS)')
if device.type == 'cuda' and peak_alloc_mb is not None:
    print(f'  GPU peak memory   : {peak_alloc_mb:.2f} MB allocated')
print(f'  Detection rate    : {det_rate:.4f} %  (RHD test, flag≥{DET_THRESHOLD})')
print(SEP)

# ── save to file ──────────────────────────────────────────────────────────────
lines = []
lines.append('Distilled Student — RHD Fine-Tune (50 ep.) — Comprehensive Profile')
lines.append(f'Checkpoint : {CKPT}')
lines.append(f'Device     : {device}')
if device.type == 'cuda':
    lines.append(f'GPU        : {torch.cuda.get_device_name(0)}')
    lines.append(f'CUDA       : {torch.version.cuda}')
lines.append(f'PyTorch    : {torch.__version__}')
lines.append('')
lines.append('--- Architecture ---')
lines.append(f'Class              : {arch_name}')
lines.append(f'Input shape        : {tuple(INPUT_SIZE)}')
lines.append(f'Output hand_flag   : {out_shapes["hand_flag"]}')
lines.append(f'Output landmarks_2d: {out_shapes["landmarks_2d"]}')
lines.append(f'Output landmarks_3d: {out_shapes["landmarks_3d"]}')
lines.append('')
lines.append('--- Parameters & Size ---')
lines.append(f'Total parameters   : {total_params:,}')
lines.append(f'Trainable params   : {trainable_params:,}')
lines.append(f'Frozen params      : {frozen_params:,}')
lines.append(f'Parameters (M)     : {total_params/1e6:.4f} M')
lines.append(f'Weight size        : {weight_mb:.4f} MB (float32)')
lines.append(f'Checkpoint size    : {disk_mb:.4f} MB (disk)')
lines.append('')
if gpu_times is not None:
    lines.append('--- GPU Inference Speed ---')
    lines.append(f'Warm-up passes     : {N_WARMUP}')
    lines.append(f'Timed passes       : {N_RUNS}')
    lines.append(f'Mean               : {gpu_times.mean():.4f} ms  ({1000/gpu_times.mean():.2f} FPS)')
    lines.append(f'Median             : {np.median(gpu_times):.4f} ms  ({1000/np.median(gpu_times):.2f} FPS)')
    lines.append(f'Min                : {gpu_times.min():.4f} ms  ({1000/gpu_times.min():.2f} FPS)')
    lines.append(f'Max                : {gpu_times.max():.4f} ms  ({1000/gpu_times.max():.2f} FPS)')
    lines.append(f'Std                : {gpu_times.std():.4f} ms')
    lines.append(f'p95                : {np.percentile(gpu_times, 95):.4f} ms  ({1000/np.percentile(gpu_times, 95):.2f} FPS)')
    lines.append(f'p99                : {np.percentile(gpu_times, 99):.4f} ms  ({1000/np.percentile(gpu_times, 99):.2f} FPS)')
    lines.append('')
lines.append('--- CPU Inference Speed ---')
lines.append(f'Timed passes       : {cpu_runs}')
lines.append(f'Mean               : {cpu_times.mean():.4f} ms  ({1000/cpu_times.mean():.2f} FPS)')
lines.append(f'Median             : {np.median(cpu_times):.4f} ms  ({1000/np.median(cpu_times):.2f} FPS)')
lines.append(f'Min                : {cpu_times.min():.4f} ms  ({1000/cpu_times.min():.2f} FPS)')
lines.append(f'Max                : {cpu_times.max():.4f} ms  ({1000/cpu_times.max():.2f} FPS)')
lines.append(f'Std                : {cpu_times.std():.4f} ms')
lines.append('')
if device.type == 'cuda' and batch_results:
    lines.append('--- Multi-Batch GPU Throughput ---')
    for bs, fps in batch_results.items():
        lines.append(f'Batch size {bs:>2}      : {fps:.2f} images/sec')
    lines.append('')
if peak_alloc_mb is not None:
    lines.append('--- GPU Memory Usage (single forward pass) ---')
    lines.append(f'Peak allocated     : {peak_alloc_mb:.4f} MB')
    lines.append(f'Peak reserved      : {peak_reserv_mb:.4f} MB')
    lines.append('')
lines.append('--- Detection Rate (RHD test set) ---')
lines.append(f'Samples evaluated  : {n_rhd:,}')
lines.append(f'Detected (flag>={DET_THRESHOLD}) : {det_count:,} / {n_rhd:,}')
lines.append(f'Detection rate     : {det_rate:.4f} %')
lines.append(f'Mean hand_flag     : {det_flags.mean():.4f}')
lines.append(f'Std  hand_flag     : {det_flags.std():.4f}')

with open(OUT_FILE, 'w') as f:
    f.write('\n'.join(lines) + '\n')

print(f'\nSaved: {OUT_FILE}')
