import torch, sys, os, numpy as np
from mmpose.apis import init_model
from mmpose.codecs import MSRAHeatmap
from mmpose.datasets import Rhd2DDataset
from torch.utils.data import DataLoader
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
init_default_scope('mmpose')

sys.path.append(os.path.join(os.getcwd(), 'student_model'))
from blazehand_landmark import BlazeHandLandmark

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load all three students ---
untrained = BlazeHandLandmark().to(device)
untrained.eval()

mediapipe_original = BlazeHandLandmark().to(device)
mediapipe_original.load_state_dict(torch.load(
    'student_model/blazehand_landmark.pth', map_location=device))
mediapipe_original.eval()

distilled_v2_ft = BlazeHandLandmark().to(device)
distilled_v2_ft.load_state_dict(torch.load(
    'checkpoints/distilled_v2_ft_epoch_150.pth', map_location=device))
distilled_v2_ft.eval()

students = {
    'Untrained'          : untrained,
    'MediaPipe_Original' : mediapipe_original,
    'Distilled_v2_FT'    : distilled_v2_ft,
}

# --- Teacher ---
CONFIG = 'mmpose/configs/hand_2d_keypoint/topdown_heatmap/rhd2d/td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256.py'
CHECKPOINT = 'checkpoints/hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth'
teacher = init_model(CONFIG, CHECKPOINT, device=device)
teacher.cfg.model.test_cfg.output_heatmaps = True
if hasattr(teacher, 'head') and hasattr(teacher.head, 'test_cfg'):
    teacher.head.test_cfg['output_heatmaps'] = True
teacher.eval()

codec = MSRAHeatmap(input_size=(256,256), heatmap_size=(64,64), sigma=2, unbiased=True)

pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256,256)),
    dict(type='PackPoseInputs')
]
ds = Rhd2DDataset(
    data_root=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'rhd'),
    ann_file='annotations/rhd_test.json',
    pipeline=pipeline
)
loader = DataLoader(ds, batch_size=1, collate_fn=pseudo_collate)

SAMPLES = 500
threshold = 0.2 * 256
auc_thresholds = np.linspace(0, 0.5*256, 20)

# Accumulators per student
results = {name: {
    'dists': [], 'correct': np.zeros(21), 'flags': []
} for name in students}

print(f'Evaluating 3 students on {SAMPLES} RHD test samples...')
print('This compares: Untrained vs Original MediaPipe vs Distilled v2 Fine-Tuned')
print()

for idx, batch in enumerate(loader):
    if idx >= SAMPLES: break

    img_tensor = torch.stack(batch['inputs']).float() / 255.0

    # Teacher reference
    with torch.no_grad():
        out = teacher.predict(img_tensor.to(device), batch['data_samples'])
        hms = out[0].pred_fields.heatmaps.cpu().numpy()
        decoded = codec.decode(hms)
        if isinstance(decoded, dict):
            tc = np.array(decoded['keypoints']).reshape(21, 2)
        else:
            tc = np.array(decoded[0]).reshape(21, 2)

    # Evaluate each student
    for name, model in students.items():
        with torch.no_grad():
            flag, _, pred = model(img_tensor.to(device))
            sc = pred[0, :, :2].cpu().numpy() * 256
            results[name]['flags'].append(flag.item())

        dist = np.sqrt(np.sum((sc - tc)**2, axis=1))
        results[name]['dists'].append(dist)
        results[name]['correct'] += (dist <= threshold).astype(float)

    if (idx+1) % 100 == 0:
        print(f'  [{idx+1}/{SAMPLES}]')

# --- Print Results ---
print()
print('='*65)
print(f'  COMPARISON: Untrained vs MediaPipe vs Distilled v2 FT ({SAMPLES} RHD test samples)')
print('='*65)
print(f'  {"Model":<22} {"PCK@0.2":>8} {"MPJPE":>10} {"EPE":>10} {"Det%":>8}')
print('  ' + '-'*60)

for name, r in results.items():
    dists = np.array(r['dists'])
    flags = np.array(r['flags'])
    pck   = r['correct'].sum() / (SAMPLES * 21)
    mpjpe = dists.mean()
    det   = (flags >= 0.5).mean() * 100

    # AUC
    auc_vals = [(dists <= t).mean() for t in auc_thresholds]
    auc = np.trapz(auc_vals, auc_thresholds / (0.5*256))

    print(f'  {name:<22} {pck:>8.4f} {mpjpe:>10.2f}px {mpjpe:>10.2f}px {det:>7.1f}%')
    results[name]['pck']   = pck
    results[name]['mpjpe'] = mpjpe
    results[name]['auc']   = auc
    results[name]['det']   = det

print('='*65)
print()

# Per-joint breakdown
joint_names = ['Wrist',
               'Thumb_MCP','Thumb_PIP','Thumb_DIP','Thumb_Tip',
               'Index_MCP','Index_PIP','Index_DIP','Index_Tip',
               'Middle_MCP','Middle_PIP','Middle_DIP','Middle_Tip',
               'Ring_MCP','Ring_PIP','Ring_DIP','Ring_Tip',
               'Pinky_MCP','Pinky_PIP','Pinky_DIP','Pinky_Tip']

print(f'  {"Joint":<15} {"Untrained":>10} {"MediaPipe":>10} {"Distilled_v2":>12}')
print('  ' + '-'*50)
for i, jname in enumerate(joint_names):
    vals = [results[n]['correct'][i]/SAMPLES for n in students]
    print(f'  {jname:<15} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f}')

# Save results
with open('three_way_comparison.txt', 'w') as f:
    f.write('THREE-WAY STUDENT COMPARISON\n')
    f.write(f'Samples: {SAMPLES}, Dataset: RHD Test\n\n')
    for name, r in results.items():
        f.write(f'{name}:\n')
        f.write(f'  PCK@0.2: {r["pck"]:.4f}\n')
        f.write(f'  MPJPE:   {r["mpjpe"]:.4f} px\n')
        f.write(f'  AUC:     {r["auc"]:.4f}\n')
        f.write(f'  Det Rate:{r["det"]:.2f}%\n\n')
print('Saved: three_way_comparison.txt')