import torch
import sys, os, subprocess

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE, 'student_model'))
from blazehand_landmark import BlazeHandLandmark

SEP = '=' * 55

def mb(bytes_val):
    return bytes_val / 1024 / 1024

def section(title):
    print(f'\n{SEP}')
    print(f'  {title}')
    print(SEP)

# ── 1. CUDA availability ───────────────────────────────────
section('1. CUDA / GPU AVAILABILITY')
cuda_ok = torch.cuda.is_available()
print(f'  torch.cuda.is_available() : {cuda_ok}')
print(f'  PyTorch version           : {torch.__version__}')

if not cuda_ok:
    print('\n  [!] No CUDA GPU detected.')
    print('      Training is running on CPU.')
    print('      To use GPU: install a CUDA-enabled PyTorch build.')
    sys.exit(0)

gpu_count = torch.cuda.device_count()
print(f'  Number of GPUs            : {gpu_count}')
for i in range(gpu_count):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

device = torch.device('cuda:0')
props  = torch.cuda.get_device_properties(device)
print(f'\n  Total VRAM  : {mb(props.total_memory):.0f} MB')
print(f'  CUDA version: {torch.version.cuda}')
print(f'  cuDNN version: {torch.backends.cudnn.version()}')

# ── 2. Model on GPU ───────────────────────────────────────
section('2. MODEL PLACEMENT CHECK')

model = BlazeHandLandmark().to(device)
model.train()

param_devices = {str(p.device) for p in model.parameters()}
print(f'  All parameter devices: {param_devices}')

all_on_gpu = all('cuda' in d for d in param_devices)
print(f'  All parameters on GPU? : {"YES ✓" if all_on_gpu else "NO ✗"}')

total_params = sum(p.numel() for p in model.parameters())
print(f'  Total parameters       : {total_params:,}')

# ── 3. Memory before forward pass ────────────────────────
section('3. GPU MEMORY — BEFORE FORWARD PASS')
torch.cuda.reset_peak_memory_stats(device)
mem_before = torch.cuda.memory_allocated(device)
print(f'  Allocated : {mb(mem_before):.2f} MB')
print(f'  Reserved  : {mb(torch.cuda.memory_reserved(device)):.2f} MB')

# ── 4. Real forward + backward pass ──────────────────────
section('4. FORWARD + BACKWARD PASS (simulating one training step)')

batch_size = 32
dummy_input = torch.randn(batch_size, 3, 256, 256, device=device)
dummy_target = torch.rand(batch_size, 21, 2, device=device) * 256

print(f'  Input tensor device  : {dummy_input.device}')
print(f'  Target tensor device : {dummy_target.device}')
print(f'  Batch size           : {batch_size}')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer.zero_grad()

_, _, pred = model(dummy_input)
pred_coords = pred[:, :, :2] * 256

loss = torch.nn.functional.mse_loss(pred_coords, dummy_target)
loss.backward()
optimizer.step()

print(f'\n  Loss value (dummy)   : {loss.item():.6f}')
print(f'  Loss device          : {loss.device}')

mem_after = torch.cuda.memory_allocated(device)
mem_peak  = torch.cuda.max_memory_allocated(device)
print(f'\n  Allocated after pass : {mb(mem_after):.2f} MB')
print(f'  Peak during pass     : {mb(mem_peak):.2f} MB')
print(f'  Memory used by pass  : {mb(mem_after - mem_before):.2f} MB')

# ── 5. Gradient check ────────────────────────────────────
section('5. GRADIENT CHECK')
grads_exist  = [(n, p.grad is not None) for n, p in model.named_parameters()]
has_grad     = [x for x in grads_exist if x[1]]
no_grad      = [x for x in grads_exist if not x[1]]
print(f'  Parameters with gradients    : {len(has_grad)}')
print(f'  Parameters without gradients : {len(no_grad)}')
print(f'  Backprop reached GPU? : {"YES ✓" if len(has_grad) > 0 else "NO ✗"}')

# ── 6. nvidia-smi snapshot ───────────────────────────────
section('6. nvidia-smi SNAPSHOT')
try:
    result = subprocess.run(
        ['nvidia-smi',
         '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=5
    )
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        headers = ['GPU Name', 'Util %', 'Mem Used (MB)', 'Mem Total (MB)', 'Temp (C)']
        print('  ' + ' | '.join(f'{h:<16}' for h in headers))
        print('  ' + '-' * 75)
        for line in lines:
            cols = [c.strip() for c in line.split(',')]
            print('  ' + ' | '.join(f'{c:<16}' for c in cols))
    else:
        print('  nvidia-smi returned an error.')
except FileNotFoundError:
    print('  nvidia-smi not found in PATH (normal on some Windows installs).')
except Exception as e:
    print(f'  nvidia-smi error: {e}')

# ── 7. Final verdict ─────────────────────────────────────
section('7. VERDICT')
if all_on_gpu and len(has_grad) > 0:
    print('  GPU IS ACTIVELY USED FOR TRAINING ✓')
    print(f'  model and all tensors are on: {torch.cuda.get_device_name(0)}')
    print(f'  Peak VRAM used in one batch: {mb(mem_peak):.2f} MB')
else:
    print('  Something is NOT on GPU — check sections above.')
print(SEP + '\n')
