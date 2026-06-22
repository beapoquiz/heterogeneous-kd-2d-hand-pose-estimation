import torch, sys, os, numpy as np, time
from mmpose.apis import init_model
from mmpose.codecs import MSRAHeatmap
from mmengine.registry import init_default_scope
init_default_scope('mmpose')

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE, 'student_model'))
from blazehand_landmark import BlazeHandLandmark

SEP = '=' * 55

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── Load student ───────────────────────────────────────────
print('\nLoading student (Distilled v2 Epoch 100)...')
student = BlazeHandLandmark().to(device)
student.load_state_dict(torch.load(
    os.path.join(BASE, 'checkpoints', 'distilled_v2_epoch_100.pth'),
    map_location=device))
student.eval()

# ── Load teacher ───────────────────────────────────────────
print('Loading teacher (HRNetV2-W18)...')
CONFIG     = os.path.join(BASE, 'mmpose/configs/hand_2d_keypoint/topdown_heatmap/rhd2d/td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256.py')
CHECKPOINT = os.path.join(BASE, 'checkpoints/hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth')
teacher = init_model(CONFIG, CHECKPOINT, device=device)
teacher.cfg.model.test_cfg.output_heatmaps = True
if hasattr(teacher, 'head') and hasattr(teacher.head, 'test_cfg'):
    teacher.head.test_cfg['output_heatmaps'] = True
teacher.eval()

codec = MSRAHeatmap(input_size=(256,256), heatmap_size=(64,64),
                    sigma=2, unbiased=True)

WARMUP = 20
RUNS   = 200
dummy  = torch.rand(1, 3, 256, 256, device=device)

def sync():
    if device.type == 'cuda':
        torch.cuda.synchronize()

def benchmark(fn, label, warmup=WARMUP, runs=RUNS):
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(runs):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    print(f'\n  {label}')
    print(f'    Mean   : {times.mean():.3f} ms')
    print(f'    Median : {np.median(times):.3f} ms')
    print(f'    Min    : {times.min():.3f} ms')
    print(f'    Max    : {times.max():.3f} ms')
    print(f'    FPS    : {1000/times.mean():.1f} frames/sec')
    return times.mean()

# ── Student benchmark ──────────────────────────────────────
print(f'\n{SEP}')
print('  STUDENT — BlazeHandLandmark (Distilled v2 Ep.100)')
print(SEP)

def student_forward():
    with torch.no_grad():
        student(dummy)

student_ms = benchmark(student_forward, 'Forward pass (full inference)')

# ── Teacher benchmark ──────────────────────────────────────
print(f'\n{SEP}')
print('  TEACHER — HRNetV2-W18')
print(SEP)

# Teacher forward pass only (backbone + head → heatmaps)
def teacher_forward_only():
    with torch.no_grad():
        teacher.backbone(dummy)

# Teacher full pipeline: forward + heatmap decode → keypoint coords
# (this is what's actually used during training and evaluation)
from mmpose.structures import PoseDataSample
from mmengine.structures import InstanceData
import torch.nn.functional as F

# Build a minimal data_sample so teacher.predict() can run
def make_dummy_sample():
    ds = PoseDataSample()
    gt = InstanceData()
    gt.bboxes        = torch.tensor([[0., 0., 256., 256.]])
    gt.bbox_centers  = torch.tensor([[128., 128.]])
    gt.bbox_scales   = torch.tensor([[256., 256.]])
    ds.gt_instances  = gt
    ds.set_metainfo({'img_shape': (256, 256), 'ori_shape': (256, 256)})
    return ds

dummy_sample = [make_dummy_sample()]

def teacher_full_pipeline():
    with torch.no_grad():
        out  = teacher.predict(dummy, dummy_sample)
        hms  = out[0].pred_fields.heatmaps.cpu().numpy()
        codec.decode(hms)

teacher_fwd_ms  = benchmark(teacher_forward_only,  'Forward pass only (backbone + head)')
teacher_full_ms = benchmark(teacher_full_pipeline, 'Full pipeline (forward + heatmap decode)')

# ── CPU benchmark (student only) ──────────────────────────
print(f'\n{SEP}')
print('  STUDENT ON CPU — BlazeHandLandmark')
print(SEP)

student_cpu = BlazeHandLandmark().cpu()
student_cpu.load_state_dict(torch.load(
    os.path.join(BASE, 'checkpoints', 'distilled_v2_epoch_100.pth'),
    map_location='cpu'))
student_cpu.eval()
dummy_cpu = torch.rand(1, 3, 256, 256)   # CPU tensor

def student_cpu_forward():
    with torch.no_grad():
        student_cpu(dummy_cpu)

student_cpu_ms = benchmark(student_cpu_forward, 'Forward pass (CPU)', warmup=5, runs=50)

# ── Parameter counts ───────────────────────────────────────
student_params = sum(p.numel() for p in student.parameters())
teacher_params = sum(p.numel() for p in teacher.parameters())

# ── Summary table ──────────────────────────────────────────
print(f'\n{SEP}')
print('  COMPARISON SUMMARY')
print(SEP)
print(f'  {"":30} {"Stu (GPU)":>10} {"Stu (CPU)":>10} {"Teacher":>10}')
print(f'  {"-"*60}')
print(f'  {"Model":30} {"BlazeHand":>10} {"BlazeHand":>10} {"HRNetV2":>10}')
print(f'  {"Parameters":30} {student_params/1e6:>9.2f}M {student_params/1e6:>9.2f}M {teacher_params/1e6:>9.2f}M')
print(f'  {"Inference time (ms)":30} {student_ms:>10.3f} {student_cpu_ms:>10.3f} {teacher_full_ms:>10.3f}')
print(f'  {"Throughput (FPS)":30} {1000/student_ms:>10.1f} {1000/student_cpu_ms:>10.1f} {1000/teacher_full_ms:>10.1f}')
print(f'  {"Model size (MB, float32)":30} {student_params*4/1024/1024:>9.2f}  {student_params*4/1024/1024:>9.2f}  {teacher_params*4/1024/1024:>9.2f}')
print(f'  {"-"*60}')

speedup_gpu = teacher_full_ms / student_ms
speedup_cpu = teacher_full_ms / student_cpu_ms
size_ratio  = teacher_params / student_params
print(f'\n  Student GPU is {speedup_gpu:.1f}x faster than teacher')
print(f'  Student CPU is {speedup_cpu:.1f}x faster than teacher (full pipeline)')
print(f'  Student is {size_ratio:.1f}x smaller than teacher (parameter count)')
print(SEP)

# ── Save results ───────────────────────────────────────────
out_path = os.path.join(BASE, 'results_speed_comparison.txt')
with open(out_path, 'w') as f:
    f.write('Inference Speed Comparison\n')
    f.write(f'Device: {device}\n\n')
    f.write(f'Student (BlazeHandLandmark — Distilled v2 Ep.100)\n')
    f.write(f'  Parameters : {student_params:,} ({student_params/1e6:.2f}M)\n')
    f.write(f'  Model size : {student_params*4/1024/1024:.2f} MB\n')
    f.write(f'  GPU inference : {student_ms:.3f} ms  ({1000/student_ms:.1f} FPS)\n')
    f.write(f'  CPU inference : {student_cpu_ms:.3f} ms  ({1000/student_cpu_ms:.1f} FPS)\n\n')
    f.write(f'Teacher (HRNetV2-W18)\n')
    f.write(f'  Parameters   : {teacher_params:,} ({teacher_params/1e6:.2f}M)\n')
    f.write(f'  Model size   : {teacher_params*4/1024/1024:.2f} MB\n')
    f.write(f'  Forward only : {teacher_fwd_ms:.3f} ms\n')
    f.write(f'  Full pipeline: {teacher_full_ms:.3f} ms  ({1000/teacher_full_ms:.1f} FPS)\n\n')
    f.write(f'Speedup (GPU): {speedup_gpu:.1f}x faster than teacher\n')
    f.write(f'Speedup (CPU): {speedup_cpu:.1f}x faster than teacher\n')
    f.write(f'Size ratio   : {size_ratio:.1f}x smaller\n')

print(f'\nSaved: results_speed_comparison.txt')
