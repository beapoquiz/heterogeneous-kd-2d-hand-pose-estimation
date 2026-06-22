import torch, sys, os, numpy as np, time
sys.path.append(os.path.join(os.getcwd(), 'student_model'))
from blazehand_landmark import BlazeHandLandmark

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BlazeHandLandmark().to(device)
model.load_state_dict(torch.load('checkpoints/distilled_v2_best.pth', map_location=device))
model.eval()

dummy = torch.rand(1, 3, 256, 256).to(device)

# Warmup
for _ in range(10):
    with torch.no_grad():
        model(dummy)

# Measure
times = []
for _ in range(100):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        model(dummy)
    torch.cuda.synchronize()
    end = time.perf_counter()
    times.append((end-start)*1000)

times = np.array(times)
print(f'Inference time (GPU):')
print(f'  Mean:   {times.mean():.3f} ms')
print(f'  Median: {np.median(times):.3f} ms')
print(f'  Min:    {times.min():.3f} ms')
print(f'  Max:    {times.max():.3f} ms')
print(f'  FPS:    {1000/times.mean():.1f} frames/second')