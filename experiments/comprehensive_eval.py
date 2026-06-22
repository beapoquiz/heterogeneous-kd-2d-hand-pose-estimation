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

# --- Load Student ---
student = BlazeHandLandmark().to(device)
student.load_state_dict(torch.load('checkpoints/distilled_student_epoch_210.pth', map_location=device))
student.eval()

# --- Load Teacher ---
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

# --- Test Dataset ---
ds = Rhd2DDataset(
    data_root=r'C:\Users\Bea Juliana Poquiz\Desktop\mmpose_thesis\dataset\rhd',
    ann_file='annotations/rhd_test.json',
    pipeline=pipeline
)
loader = DataLoader(ds, batch_size=1, collate_fn=pseudo_collate)
N = len(ds)
print(f'Evaluating on {N} test samples...')

threshold_02 = 0.2 * 256  # PCK@0.2
auc_thresholds = np.linspace(0, 0.5 * 256, 20)  # for AUC

# Accumulators
student_dists  = []  # per-sample per-joint distances (student vs teacher)
teacher_dists  = []  # per-sample per-joint distances (teacher vs GT)
student_correct_02 = np.zeros(21)
teacher_correct_02 = np.zeros(21)

for idx, batch in enumerate(loader):
    img_tensor = torch.stack(batch['inputs']).float() / 255.0
    gt_raw = batch['data_samples'][0].gt_instances.keypoints[0]  # [21,2] original space

    # Teacher prediction
    with torch.no_grad():
        out = teacher.predict(img_tensor.to(device), batch['data_samples'])
        hms = out[0].pred_fields.heatmaps.cpu().numpy()
        decoded = codec.decode(hms)
        if isinstance(decoded, dict):
            tc = np.array(decoded['keypoints']).reshape(21, 2)
        else:
            tc = np.array(decoded[0]).reshape(21, 2)

    # Student prediction
    with torch.no_grad():
        _, _, pred = student(img_tensor.to(device))
        sc = pred[0, :, :2].cpu().numpy() * 256

    # Per-joint Euclidean distances
    s_dist = np.sqrt(np.sum((sc - tc)**2, axis=1))   # student vs teacher [21]
    student_dists.append(s_dist)

    # PCK@0.2 student vs teacher
    student_correct_02 += (s_dist <= threshold_02).astype(float)

    if (idx+1) % 500 == 0:
        print(f'  [{idx+1}/{N}]')

student_dists = np.array(student_dists)   # [N, 21]

# ── METRICS ──────────────────────────────────────────────────────────────────

# 1. MPJPE (pixels)
mpjpe_per_joint = student_dists.mean(axis=0)   # [21]
mpjpe_overall   = student_dists.mean()

# 2. EPE (same as MPJPE mean, reported overall)
epe = mpjpe_overall

# 3. PCK@0.2
pck_per_joint = student_correct_02 / N
pck_overall   = pck_per_joint.mean()

# 4. AUC (area under PCK curve from threshold 0 to 0.5*256)
auc_values = []
for thr in auc_thresholds:
    pck_at_thr = (student_dists <= thr).mean()
    auc_values.append(pck_at_thr)
auc = np.trapz(auc_values, auc_thresholds / (0.5 * 256)) / 1.0  # normalized 0-1

joint_names = ['Wrist',
               'Thumb_MCP','Thumb_PIP','Thumb_DIP','Thumb_Tip',
               'Index_MCP','Index_PIP','Index_DIP','Index_Tip',
               'Middle_MCP','Middle_PIP','Middle_DIP','Middle_Tip',
               'Ring_MCP','Ring_PIP','Ring_DIP','Ring_Tip',
               'Pinky_MCP','Pinky_PIP','Pinky_DIP','Pinky_Tip']

print()
print('=' * 55)
print('  COMPREHENSIVE EVALUATION — Student vs Teacher')
print(f'  Dataset: RHD Test Split ({N} samples)')
print('=' * 55)
print(f'  Overall PCK@0.2 : {pck_overall:.4f}  ({pck_overall*100:.2f}%)')
print(f'  Overall MPJPE   : {mpjpe_overall:.4f} px')
print(f'  EPE             : {epe:.4f} px')
print(f'  AUC (0-0.5*256) : {auc:.4f}')
print()
print(f'  {"Joint":<15} {"PCK@0.2":>10} {"MPJPE(px)":>12}')
print('  ' + '-'*40)
for i, name in enumerate(joint_names):
    print(f'  {name:<15} {pck_per_joint[i]:>10.4f} {mpjpe_per_joint[i]:>12.4f}')
print('=' * 55)

# Save to file for thesis
with open('evaluation_results.txt', 'w') as f:
    f.write('COMPREHENSIVE EVALUATION — Student vs Teacher\n')
    f.write(f'Dataset: RHD Test Split ({N} samples)\n\n')
    f.write(f'Overall PCK@0.2 : {pck_overall:.4f}\n')
    f.write(f'Overall MPJPE   : {mpjpe_overall:.4f} px\n')
    f.write(f'EPE             : {epe:.4f} px\n')
    f.write(f'AUC             : {auc:.4f}\n\n')
    f.write(f'{"Joint":<15} {"PCK@0.2":>10} {"MPJPE(px)":>12}\n')
    f.write('-'*40 + '\n')
    for i, name in enumerate(joint_names):
        f.write(f'{name:<15} {pck_per_joint[i]:>10.4f} {mpjpe_per_joint[i]:>12.4f}\n')
print('Results saved to evaluation_results.txt')