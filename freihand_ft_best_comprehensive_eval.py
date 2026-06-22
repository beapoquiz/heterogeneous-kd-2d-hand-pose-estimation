import torch, sys, os, numpy as np

sys.path.append(os.path.join(os.getcwd(), 'student_model'))
from blazehand_landmark import BlazeHandLandmark
from freihand_dataset import make_train_val_loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

student = BlazeHandLandmark().to(device)
student.load_state_dict(torch.load(
    'checkpoints/freihand_ft_best.pth', map_location=device))
student.eval()

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

print(f'Comprehensive Eval — freihand_ft_best.pth (FreiHAND Fine-Tuned, Best @ Epoch 50)')
print(f'Starting checkpoint : checkpoints/distilled_v2_ft_epoch_150.pth  (RHD Epoch 150)')
print(f'FreiHAND validation split: {N} samples')
print()

for idx, (img, kps_gt) in enumerate(val_loader):
    img    = img.to(device)              # (1, 3, 256, 256)  already /255
    kps_gt = kps_gt.squeeze(0).numpy()  # (21, 2) in [0, 256)

    with torch.no_grad():
        flag, _, pred = student(img)
        sc = pred[0, :, :2].cpu().numpy() * 256  # (21, 2) in [0, 256)
        flags.append(flag.item())

    dist = np.sqrt(np.sum((sc - kps_gt) ** 2, axis=1))  # (21,)
    dists.append(dist)
    correct += (dist <= threshold).astype(float)

    if (idx + 1) % 500 == 0:
        print(f'  [{idx+1}/{N}] Running PCK@0.2: {correct.sum()/((idx+1)*21):.4f}')

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
print('FREIHAND FT BEST — FULL EVALUATION')
print('Fine-tuned on FreiHAND | 50 Epochs | Best Checkpoint (Epoch 50)')
print(f'Dataset: FreiHAND validation split ({N} samples)')
print('Ground truth: FreiHAND 3D → 2D projected keypoints')
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

with open('results_freihand_ft_best.txt', 'w') as f:
    f.write('FreiHAND Fine-Tuned Best — Full Evaluation\n')
    f.write('Fine-tuned on FreiHAND | 50 Epochs | Best Checkpoint (Epoch 50)\n')
    f.write('Starting checkpoint : checkpoints/distilled_v2_ft_epoch_150.pth  (RHD Epoch 150)\n')
    f.write(f'Dataset: FreiHAND validation split ({N} samples)\n')
    f.write('Ground truth: FreiHAND 3D -> 2D projected keypoints\n\n')
    f.write(f'PCK@0.2:    {pck:.4f}\n')
    f.write(f'MPJPE:      {mpjpe:.4f} px\n')
    f.write(f'MSE:        {mse:.4f} px2\n')
    f.write(f'AUC:        {auc:.4f}\n')
    f.write(f'Det Rate:   {det:.2f}%\n\n')
    f.write(f'{"Joint":<15} {"PCK@0.2":>10} {"MPJPE":>12}\n')
    f.write('-' * 40 + '\n')
    for i, name in enumerate(joint_names):
        f.write(f'{name:<15} {pck_per_joint[i]:>10.4f} {mpjpe_per_joint[i]:>12.4f}\n')

print('Saved: results_freihand_ft_best.txt')
