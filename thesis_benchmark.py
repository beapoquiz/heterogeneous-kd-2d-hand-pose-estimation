"""
thesis_benchmark.py
====================
Full benchmark for the thesis report.

Measures and compares:
  Student  — BlazeHandLandmark (distilled_v2_ft_epoch_150.pth)
             Heterogeneous KD, 100 distillation + 50 fine-tuning epochs on RHD
  Teacher  — TopdownPoseEstimator / HRNetV2-W18 + DARK
             (hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth)

What is reported
----------------
  STUDENT
    1.  Architecture name, input/output shapes
    2.  Parameter count (total / trainable)
    3.  Checkpoint file size (disk, MB)
    4.  Float32 weight footprint (MB)
    5.  GPU inference speed — mean, median, min, max, std, p95, p99, FPS
        (50 warm-up + 500 timed single-image passes)
    6.  CPU inference speed — same statistics (100 timed passes)
    7.  Multi-batch GPU throughput at BS = 1, 4, 8, 16
    8.  GPU peak memory (allocated / reserved)
    9.  Detection rate on full RHD test set (2 728 images, flag >= 0.5)

  TEACHER
    10. Parameter count
    11. Checkpoint file size
    12. GPU full-pipeline speed (forward + heatmap decode, same methodology)
    13. GPU forward-only speed (backbone + head, no decode)

  COMPARISON TABLE
    14. Side-by-side: params, size, FPS, speedup ratio, compression ratio

All output is printed to the terminal AND saved to:
  thesis_benchmark_results.txt

Run with:
  python thesis_benchmark.py
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
from mmpose.apis import init_model
from mmpose.codecs import MSRAHeatmap
from torch.utils.data import DataLoader
init_default_scope('mmpose')

# ── paths ─────────────────────────────────────────────────────────────────────
CKPT_STUDENT   = os.path.join(BASE, 'checkpoints', 'distilled_v2_ft_epoch_150.pth')
CKPT_TEACHER   = os.path.join(BASE, 'checkpoints',
                               'hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth')
CONFIG_TEACHER = os.path.join(BASE, 'mmpose', 'configs', 'hand_2d_keypoint',
                               'topdown_heatmap', 'rhd2d',
                               'td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256.py')
RHD_ROOT       = os.path.join(BASE, 'dataset', 'rhd')
OUT_FILE       = os.path.join(BASE, 'thesis_benchmark_results.txt')

# ── config ────────────────────────────────────────────────────────────────────
INPUT_SHAPE    = (1, 3, 256, 256)
N_WARMUP       = 50
N_RUNS         = 500
N_RUNS_CPU     = 100
BATCH_SIZES    = [1, 4, 8, 16]
DET_THRESHOLD  = 0.5

SEP  = '=' * 65
SEP2 = '-' * 65

# ── helpers ───────────────────────────────────────────────────────────────────
lines = []   # accumulated for file save

def log(msg=''):
    print(msg)
    lines.append(msg)

def sync(dev):
    if dev.type == 'cuda':
        torch.cuda.synchronize()

def timed_runs(fn, dev, n_warmup=N_WARMUP, n_runs=N_RUNS):
    """fn() is a callable that performs one inference. Returns ms array."""
    for _ in range(n_warmup):
        fn()
        sync(dev)
    times = []
    for _ in range(n_runs):
        sync(dev)
        t0 = time.perf_counter()
        fn()
        sync(dev)
        times.append((time.perf_counter() - t0) * 1000.0)
    return np.array(times)

def speed_stats(times_ms, label, dev_label):
    log(f'  Device         : {dev_label}')
    log(f'  Warm-up passes : {N_WARMUP}   Timed passes: {len(times_ms)}')
    log(f'  Mean           : {times_ms.mean():>9.3f} ms  →  {1000/times_ms.mean():>8.1f} FPS')
    log(f'  Median         : {np.median(times_ms):>9.3f} ms  →  {1000/np.median(times_ms):>8.1f} FPS')
    log(f'  Min            : {times_ms.min():>9.3f} ms  →  {1000/times_ms.min():>8.1f} FPS')
    log(f'  Max            : {times_ms.max():>9.3f} ms  →  {1000/times_ms.max():>8.1f} FPS')
    log(f'  Std            : {times_ms.std():>9.3f} ms')
    log(f'  p95            : {np.percentile(times_ms,95):>9.3f} ms  →  {1000/np.percentile(times_ms,95):>8.1f} FPS')
    log(f'  p99            : {np.percentile(times_ms,99):>9.3f} ms  →  {1000/np.percentile(times_ms,99):>8.1f} FPS')

def batch_throughput(model_fn, dev, batch_size, n_warmup=20, n_runs=100):
    dummy = torch.randn(batch_size, 3, 256, 256, device=dev)
    for _ in range(n_warmup):
        model_fn(dummy)
        sync(dev)
    times = []
    for _ in range(n_runs):
        sync(dev)
        t0 = time.perf_counter()
        model_fn(dummy)
        sync(dev)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return batch_size * 1000.0 / np.median(arr)   # images / sec

# ── device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(SEP)
log(' THESIS BENCHMARK — Heterogeneous Knowledge Distillation')
log(' Student: BlazeHandLandmark   Teacher: HRNetV2-W18 + DARK')
log(SEP)
log(f'  Device  : {device}')
if device.type == 'cuda':
    log(f'  GPU     : {torch.cuda.get_device_name(0)}')
    log(f'  CUDA    : {torch.version.cuda}')
log(f'  PyTorch : {torch.__version__}')
log()

# ─────────────────────────────────────────────────────────────────────────────
# STUDENT
# ─────────────────────────────────────────────────────────────────────────────
log(SEP)
log('  STUDENT — BlazeHandLandmark  (distilled_v2_ft_epoch_150.pth)')
log('            100 ep. distillation + 50 ep. RHD fine-tuning')
log(SEP)

log('  Loading checkpoint...')
student = BlazeHandLandmark().to(device)
student.load_state_dict(torch.load(CKPT_STUDENT, map_location=device))
student.eval()

# 1. architecture / output shapes
dummy_s = torch.zeros(*INPUT_SHAPE, device=device)
with torch.no_grad():
    hf, hd, lm = student(dummy_s)

log()
log('  [1] Architecture & Output Shapes')
log(SEP2)
log(f'  Class              : {student.__class__.__name__}')
log(f'  Input shape        : {tuple(INPUT_SHAPE)}  (B × C × H × W)')
log(f'  hand_flag  shape   : {tuple(hf.shape)}   — detection confidence [0,1]')
log(f'  handed     shape   : {tuple(hd.shape)}   — handedness (not used in KD)')
log(f'  landmarks  shape   : {tuple(lm.shape)}   — (B,21,3) normalised [0,1]')
log(f'  For KD/eval: landmarks[:,:,:2] * 256  →  (B,21,2) pixel coords')

# 2–4. params & size
s_total  = sum(p.numel() for p in student.parameters())
s_train  = sum(p.numel() for p in student.parameters() if p.requires_grad)
s_weight = s_total * 4 / 1024 / 1024
s_disk   = os.path.getsize(CKPT_STUDENT) / 1024 / 1024

log()
log('  [2-4] Parameters & Size')
log(SEP2)
log(f'  Total parameters   : {s_total:>12,}')
log(f'  Trainable params   : {s_train:>12,}')
log(f'  Parameters (M)     : {s_total/1e6:>12.4f} M')
log(f'  Weight size        : {s_weight:>12.4f} MB  (float32 tensors)')
log(f'  Checkpoint on disk : {s_disk:>12.4f} MB')

# 5. GPU speed
log()
log('  [5] GPU Inference Speed  (single image, BS=1)')
log(SEP2)
if device.type == 'cuda':
    dummy_sg = torch.randn(*INPUT_SHAPE, device=device)
    with torch.no_grad():
        s_gpu_times = timed_runs(lambda: student(dummy_sg), device)
    speed_stats(s_gpu_times, 'student_gpu', str(device))
    s_gpu_median = float(np.median(s_gpu_times))
    s_gpu_fps    = 1000.0 / s_gpu_median
else:
    log('  (no GPU available — skipped)')
    s_gpu_times  = None
    s_gpu_median = None
    s_gpu_fps    = None

# 6. CPU speed
log()
log('  [6] CPU Inference Speed  (single image, BS=1)')
log(SEP2)
student_cpu = BlazeHandLandmark().cpu()
student_cpu.load_state_dict(torch.load(CKPT_STUDENT, map_location='cpu'))
student_cpu.eval()
dummy_sc = torch.randn(*INPUT_SHAPE)
cpu_dev  = torch.device('cpu')
with torch.no_grad():
    s_cpu_times = timed_runs(lambda: student_cpu(dummy_sc), cpu_dev,
                              n_warmup=10, n_runs=N_RUNS_CPU)
speed_stats(s_cpu_times, 'student_cpu', 'cpu')
s_cpu_median = float(np.median(s_cpu_times))
s_cpu_fps    = 1000.0 / s_cpu_median
del student_cpu

# 7. multi-batch throughput
log()
log('  [7] Multi-Batch GPU Throughput')
log(SEP2)
batch_results = {}
if device.type == 'cuda':
    with torch.no_grad():
        for bs in BATCH_SIZES:
            fps = batch_throughput(lambda x: student(x), device, bs)
            batch_results[bs] = fps
            log(f'  Batch size {bs:>2}       : {fps:>8.1f} images / sec')
else:
    log('  (no GPU — skipped)')

# 8. GPU memory
log()
log('  [8] GPU Memory Usage  (single forward pass, BS=1)')
log(SEP2)
if device.type == 'cuda':
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        student(dummy_sg)
    torch.cuda.synchronize()
    s_alloc_mb  = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    s_reserv_mb = torch.cuda.max_memory_reserved(device)  / 1024 / 1024
    log(f'  Peak allocated     : {s_alloc_mb:>8.2f} MB')
    log(f'  Peak reserved      : {s_reserv_mb:>8.2f} MB')
else:
    s_alloc_mb = s_reserv_mb = None
    log('  (no GPU — skipped)')

# 9. detection rate on full RHD test set
log()
log('  [9] Detection Rate — Full RHD Test Set (2 728 images)')
log(SEP2)
log('  Running inference on every test sample (this takes a few minutes)...')

pipeline_val = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(type='PackPoseInputs'),
]
rhd_ds  = Rhd2DDataset(data_root=RHD_ROOT,
                        ann_file='annotations/rhd_test.json',
                        pipeline=pipeline_val)
rhd_dl  = DataLoader(rhd_ds, batch_size=1, collate_fn=pseudo_collate)
N_RHD   = len(rhd_ds)
flags   = []
student.eval()
with torch.no_grad():
    for idx, batch in enumerate(rhd_dl):
        img = torch.stack(batch['inputs']).float().to(device) / 255.0
        flag, _, _ = student(img)
        flags.append(float(flag.item()))
        if (idx + 1) % 500 == 0:
            print(f'    [{idx+1}/{N_RHD}] ...', flush=True)

flags     = np.array(flags)
det_count = int((flags >= DET_THRESHOLD).sum())
det_rate  = det_count / N_RHD * 100.0

log(f'  Samples evaluated  : {N_RHD:,}')
log(f'  Detected (≥{DET_THRESHOLD})    : {det_count:,}  /  {N_RHD:,}')
log(f'  Detection rate     : {det_rate:.4f} %')
log(f'  Mean hand_flag     : {flags.mean():.4f}')
log(f'  Std  hand_flag     : {flags.std():.4f}')
log(f'  Min  hand_flag     : {flags.min():.4f}')
log(f'  Max  hand_flag     : {flags.max():.4f}')

# ─────────────────────────────────────────────────────────────────────────────
# TEACHER
# ─────────────────────────────────────────────────────────────────────────────
log()
log(SEP)
log('  TEACHER — HRNetV2-W18 + DARK Decoding')
log('            (hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth)')
log(SEP)

log('  Loading teacher model...')
teacher = init_model(CONFIG_TEACHER, CKPT_TEACHER, device=device)
teacher.cfg.model.test_cfg.output_heatmaps = True
if hasattr(teacher, 'head') and hasattr(teacher.head, 'test_cfg'):
    if isinstance(teacher.head.test_cfg, dict):
        teacher.head.test_cfg['output_heatmaps'] = True
    else:
        teacher.head.test_cfg.output_heatmaps = True

# Disable flip_test (TTA) for the speed benchmark:
# flip_test=True requires flip_indices in metainfo, which a synthetic dummy
# sample does not carry.  Real-time deployment also never uses TTA.
if isinstance(teacher.test_cfg, dict):
    teacher.test_cfg['flip_test'] = False
else:
    teacher.test_cfg.flip_test = False
if hasattr(teacher, 'head') and hasattr(teacher.head, 'test_cfg'):
    if isinstance(teacher.head.test_cfg, dict):
        teacher.head.test_cfg['flip_test'] = False
    else:
        teacher.head.test_cfg.flip_test = False

teacher.eval()
log('  (flip_test disabled for benchmark — TTA not used in real-time inference)')

codec = MSRAHeatmap(input_size=(256,256), heatmap_size=(64,64), sigma=2, unbiased=True)

# 10. params & size
t_total  = sum(p.numel() for p in teacher.parameters())
t_train  = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
t_weight = t_total * 4 / 1024 / 1024
t_disk   = os.path.getsize(CKPT_TEACHER) / 1024 / 1024

log()
log('  [10] Parameters & Size')
log(SEP2)
log(f'  Total parameters   : {t_total:>12,}')
log(f'  Trainable params   : {t_train:>12,}')
log(f'  Parameters (M)     : {t_total/1e6:>12.4f} M')
log(f'  Weight size        : {t_weight:>12.4f} MB  (float32 tensors)')
log(f'  Checkpoint on disk : {t_disk:>12.4f} MB')

dummy_t = torch.randn(*INPUT_SHAPE, device=device)

# Teacher is benchmarked by calling submodules directly
# (backbone → neck → head → codec.decode) to avoid needing metainfo-populated
# PoseDataSample objects that teacher.predict() requires.
# This measures the same computation that runs during distillation training.

# 11. teacher GPU — full pipeline (backbone + neck + head + heatmap decode)
log()
log('  [11] Teacher GPU Speed — Full Pipeline (backbone+neck+head + decode)')
log('       (submodule calls — same ops used during distillation training)')
log(SEP2)
if device.type == 'cuda':
    def teacher_full():
        feats = teacher.backbone(dummy_t)
        if teacher.with_neck:
            feats = teacher.neck(feats)
        hms = teacher.head.forward(feats)          # (B, 21, 64, 64)
        hms_np = hms[0].detach().cpu().numpy()
        codec.decode(hms_np)

    t_full_times = timed_runs(teacher_full, device)
    speed_stats(t_full_times, 'teacher_full', str(device))
    t_full_median = float(np.median(t_full_times))
    t_full_fps    = 1000.0 / t_full_median
else:
    log('  (no GPU — skipped)')
    t_full_times  = None
    t_full_median = None
    t_full_fps    = None

# 12. teacher GPU — forward only (backbone + neck + head, no decode)
log()
log('  [12] Teacher GPU Speed — Forward Only (backbone+neck+head, no decode)')
log(SEP2)
if device.type == 'cuda':
    def teacher_fwd_only():
        feats = teacher.backbone(dummy_t)
        if teacher.with_neck:
            feats = teacher.neck(feats)
        teacher.head.forward(feats)

    t_fwd_times = timed_runs(teacher_fwd_only, device)
    speed_stats(t_fwd_times, 'teacher_fwd_only', str(device))
    t_fwd_median = float(np.median(t_fwd_times))
    t_fwd_fps    = 1000.0 / t_fwd_median
else:
    log('  (no GPU — skipped)')
    t_fwd_times  = None
    t_fwd_median = None
    t_fwd_fps    = None

# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
log()
log(SEP)
log('  COMPARISON TABLE — Student vs Teacher')
log(SEP)

W1, W2, W3 = 32, 18, 18

def row(label, sv, tv):
    log(f'  {label:<{W1}} {sv:>{W2}} {tv:>{W3}}')

row('Metric', 'Student (BlazeHand)', 'Teacher (HRNetV2-W18)')
log(f'  {"-"*(W1+W2+W3+4)}')
row('Architecture',          'BlazeHandLandmark',     'TopdownPoseEstimator')
row('Output type',           'Coordinates (21×3)',    'Heatmaps (21×64×64)')
row('Parameters',            f'{s_total/1e6:.4f} M',  f'{t_total/1e6:.4f} M')
row('Weight size (float32)', f'{s_weight:.2f} MB',    f'{t_weight:.2f} MB')
row('Checkpoint on disk',    f'{s_disk:.2f} MB',      f'{t_disk:.2f} MB')

if s_gpu_fps is not None and t_full_fps is not None:
    row('GPU FPS (full pipeline)', f'{s_gpu_fps:.1f} FPS', f'{t_full_fps:.1f} FPS')
    row('GPU latency (median)',    f'{s_gpu_median:.3f} ms', f'{t_full_median:.3f} ms')

row('CPU FPS', f'{s_cpu_fps:.1f} FPS', 'N/A')
row('CPU latency (median)', f'{s_cpu_median:.3f} ms', 'N/A')

if s_gpu_fps is not None and t_full_fps is not None:
    speedup_gpu = t_full_median / s_gpu_median
    log()
    log(f'  GPU speedup (full pipeline): {speedup_gpu:.1f}x  faster than teacher')
    if t_fwd_fps is not None:
        speedup_fwd = t_fwd_median / s_gpu_median
        log(f'  GPU speedup (vs fwd-only)  : {speedup_fwd:.1f}x  faster than teacher backbone+head')
    param_ratio = t_total / s_total
    size_ratio  = t_weight / s_weight
    log(f'  Parameter compression      : {param_ratio:.1f}x  fewer parameters')
    log(f'  Size compression           : {size_ratio:.1f}x  smaller (float32 weights)')

log()
log('  Note: detection rate / accuracy is from eval scripts, not this benchmark.')
log(f'  Teacher RHD test: PCK@0.2=0.992, AUC=0.902, EPE=2.21 px (from paper)')
log()
log(f'  Real-time threshold: >= 30 FPS')
if s_gpu_fps is not None:
    rt_gpu = 'YES' if s_gpu_fps >= 30 else 'NO'
    log(f'  Student real-time (GPU): {rt_gpu}  ({s_gpu_fps:.1f} FPS)')
rt_cpu = 'YES' if s_cpu_fps >= 30 else 'NO'
log(f'  Student real-time (CPU): {rt_cpu}  ({s_cpu_fps:.1f} FPS)')

log()
log(SEP)
log('  BENCHMARK COMPLETE')
log(SEP)

# ── save to file ──────────────────────────────────────────────────────────────
with open(OUT_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')

print(f'\nAll results saved to: {OUT_FILE}')
print('Please paste the contents of that file (or this terminal output) back.')
