import torch, sys, os, numpy as np
from mmpose.apis import init_model
from mmpose.codecs import MSRAHeatmap
from mmengine.registry import init_default_scope
init_default_scope('mmpose')

sys.path.append(os.path.join(os.getcwd(), 'student_model'))
from blazehand_landmark import BlazeHandLandmark
from freihand_dataset import make_train_val_loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Student ──────────────────────────────────────────────────────────────────
student = BlazeHandLandmark().to(device)
student.load_state_dict(torch.load(
    'checkpoints/freihand_ft_epoch_50.pth', map_location=device))
student.eval()

# ── Teacher ───────────────────────────────────────────────────────────────────
CONFIG     = 'mmpose/configs/hand_2d_keypoint/topdown_heatmap/rhd2d/td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256.py'
CHECKPOINT = 'checkpoints/hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth'
teacher = init_model(CONFIG, CHECKPOINT, device=device)
teacher.eval()

codec = MSRAHeatmap(input_size=(256, 256), heatmap_size=(64, 64), sigma=2, unbiased=True)

# ── Data ──────────────────────────────────────────────────────────────────────
FREIHAND_ROOT = os.path.join(os.getcwd(), 'freihand')
_, val_loader = make_train_val_loaders(
    FREIHAND_ROOT,
    input_size=256,
    val_frac=0.05,
    batch_size=1,
)

N = len(val_loader.dataset)
threshold      = 0.2 * 256
auc_thresholds = np.linspace(0, 0.5 * 256, 20)
correct        = np.zeros(21)
dists          = []
flags          = []

print(f'Comprehensive Eval — freihand_ft_epoch_50.pth (FreiHAND Fine-Tuned, Epoch 50)')
print(f'Starting checkpoint : checkpoints/distilled_v2_ft_epoch_150.pth  (RHD Epoch 150)')
print(f'Reference           : Teacher (HRNetV2-W18, RHD-trained)')
print(f'FreiHAND validation split: {N} samples')
print()

for idx, (img, _) in enumerate(val_loader):
    img = img.to(device)  # (1, 3, 256, 256) in [0, 1]

    # teacher: backbone → head directly (avoids data_samples requirement)
    with torch.no_grad():
        feats = teacher.extract_feat(img)
        hm = teacher.head(feats)
        if isinstance(hm, (list, tuple)):
            hm = hm[-1]
        hms = hm[0].cpu().numpy()  # (21, 64, 64)
    decoded = codec.decode(hms)
    if isinstance(decoded, dict):
        tc = np.array(decoded['keypoints']).reshape(21, 2)
    else:
        tc = np.array(decoded[0]).reshape(21, 2)

    # student: (21, 2) in [0, 256)
    with torch.no_grad():
        flag, _, pred = student(img)
        sc = pred[0, :, :2].cpu().numpy() * 256
        flags.append(flag.item())

    dist = np.sqrt(np.sum((sc - tc) ** 2, axis=1))  # (21,)
    dists.append(dist)
    correct += (dist <= threshold).astype(float)

    if (idx + 1) % 500 == 0:
        print(f'  [{idx+1}/{N}] Running PCK@0.2 (vs teacher): {correct.sum()/((idx+1)*21):.4f}')

dists  = np.array(dists)
flags  = np.array(flags)
pck    = correct.sum() / (N * 21)
mpjpe  = dists.mean()
mse    = np.mean(np.sum(dists ** 2, axis=1) / 21)
det    = (flags >= 0.5).mean() * 100
auc_v  = [(dists <= t).mean() for t in auc_thresholds]
auc    = np.trapz(auc_v, auc_thresholds / (0.5 * 256))

joint_names = ['Wrist',
               'Thumb_MCP', 'Thumb_PIP', 'Thumb_DIP', 'Thumb_Tip',
               'Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_Tip',
               'Middle_MCP','Middle_PIP','Middle_DIP','Middle_Tip',
               'Ring_MCP',  'Ring_PIP',  'Ring_DIP',  'Ring_Tip',
               'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP', 'Pinky_Tip']

pck_per_joint   = correct / N
mpjpe_per_joint = dists.mean(axis=0)

print()
print('=' * 55)
print('FREIHAND FT EPOCH 50 — TEACHER COORD EVALUATION')
print('Fine-tuned on FreiHAND | 50 Epochs | Last Checkpoint (Epoch 50)')
print(f'Dataset: FreiHAND validation split ({N} samples)')
print('Reference: Teacher predictions (HRNetV2-W18, RHD-trained)')
print('=' * 55)
print(f'PCK@0.2      : {pck:.4f} ({pck*100:.2f}%)')
print(f'MPJPE        : {mpjpe:.4f} px')
print(f'MSE          : {mse:.4f} px²')
print(f'AUC          : {auc:.4f}')
print(f'Det. Rate    : {det:.2f}%')
print()
print(f'{"Joint":<15} {"PCK@0.2":>10} {"MPJPE (px)":>12}')
print('-' * 40)
for i, name in enumerate(joint_names):
    print(f'{name:<15} {pck_per_joint[i]:>10.4f} {mpjpe_per_joint[i]:>12.4f}')
print('=' * 55)

with open('results_freihand_ft_epoch_50_tc.txt', 'w') as f:
    f.write('FreiHAND Fine-Tuned Epoch 50 — Teacher Coord Evaluation\n')
    f.write('Fine-tuned on FreiHAND | 50 Epochs | Last Checkpoint (Epoch 50)\n')
    f.write('Starting checkpoint : checkpoints/distilled_v2_ft_epoch_150.pth  (RHD Epoch 150)\n')
    f.write(f'Dataset: FreiHAND validation split ({N} samples)\n')
    f.write('Reference: Teacher predictions (HRNetV2-W18, RHD-trained)\n\n')
    f.write(f'PCK@0.2:    {pck:.4f}\n')
    f.write(f'MPJPE:      {mpjpe:.4f} px\n')
    f.write(f'MSE:        {mse:.4f} px2\n')
    f.write(f'AUC:        {auc:.4f}\n')
    f.write(f'Det Rate:   {det:.2f}%\n\n')
    f.write(f'{"Joint":<15} {"PCK@0.2":>10} {"MPJPE":>12}\n')
    f.write('-' * 40 + '\n')
    for i, name in enumerate(joint_names):
        f.write(f'{name:<15} {pck_per_joint[i]:>10.4f} {mpjpe_per_joint[i]:>12.4f}\n')

print('Saved: results_freihand_ft_epoch_50_tc.txt')
