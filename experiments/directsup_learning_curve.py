import torch, sys, os, numpy as np
from mmpose.datasets import Rhd2DDataset
from torch.utils.data import DataLoader
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
init_default_scope('mmpose')

sys.path.append(os.path.join(os.getcwd(), 'student_model'))
from blazehand_landmark import BlazeHandLandmark

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Pipeline — same as direct supervision training
pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256,256)),
    dict(type='PackPoseInputs')
]
ds = Rhd2DDataset(
    data_root=r'C:\Users\Bea Juliana Poquiz\Desktop\mmpose_thesis\dataset\rhd',
    ann_file='annotations/rhd_test.json',
    pipeline=pipeline
)
loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=pseudo_collate)

SAMPLES = 200  # same as training validation
threshold = 0.2 * 256
auc_thresholds = np.linspace(0, 0.5*256, 20)

joint_names = ['Wrist',
               'Thumb_MCP','Thumb_PIP','Thumb_DIP','Thumb_Tip',
               'Index_MCP','Index_PIP','Index_DIP','Index_Tip',
               'Middle_MCP','Middle_PIP','Middle_DIP','Middle_Tip',
               'Ring_MCP','Ring_PIP','Ring_DIP','Ring_Tip',
               'Pinky_MCP','Pinky_PIP','Pinky_DIP','Pinky_Tip']

# All available direct supervision checkpoints
checkpoints = [
    ('DS_Best_Ep25',  'checkpoints/direct_sup_best.pth'),
    ('DS_Epoch_10',   'checkpoints/direct_sup_epoch_10.pth'),
    ('DS_Epoch_20',   'checkpoints/direct_sup_epoch_20.pth'),
    ('DS_Epoch_30',   'checkpoints/direct_sup_epoch_30.pth'),
    ('DS_Epoch_40',   'checkpoints/direct_sup_epoch_40.pth'),
    ('DS_Epoch_50',   'checkpoints/direct_sup_epoch_50.pth'),
    ('DS_Epoch_60',   'checkpoints/direct_sup_epoch_60.pth'),
    ('DS_Epoch_70',   'checkpoints/direct_sup_epoch_70.pth'),
    ('DS_Epoch_80',   'checkpoints/direct_sup_epoch_80.pth'),
    ('DS_Epoch_90',   'checkpoints/direct_sup_epoch_90.pth'),
    ('DS_Epoch_100',  'checkpoints/direct_sup_epoch_100.pth'),
]

print()
print('Direct Supervision — Learning Curve (All Checkpoints)')
print(f'Evaluated on: {SAMPLES}-sample RHD test subset')
print(f'Reference:    Raw GT annotations')
print()
print(f'{"Checkpoint":<18} {"PCK@0.2":>8} {"MPJPE":>9} {"MSE":>10} {"AUC":>8} {"Det%":>7}')
print('-'*65)

all_results = []

for name, ckpt_path in checkpoints:
    if not os.path.exists(ckpt_path):
        print(f'{name:<18} NOT FOUND — skipping')
        continue

    student = BlazeHandLandmark().to(device)
    student.load_state_dict(torch.load(ckpt_path, map_location=device))
    student.eval()

    correct = np.zeros(21)
    dists   = []
    flags   = []

    for idx, batch in enumerate(loader):
        if idx >= SAMPLES: break
        img_tensor = torch.stack(batch['inputs']).float() / 255.0

        # Raw GT reference
        gt_raw = batch['data_samples'][0].gt_instances.keypoints[0]
        tc = np.clip(gt_raw, 0, 255).astype(np.float32)

        with torch.no_grad():
            flag, _, pred = student(img_tensor.to(device))
            sc = pred[0, :, :2].cpu().numpy() * 256
            flags.append(flag.item())

        dist = np.sqrt(np.sum((sc - tc)**2, axis=1))
        dists.append(dist)
        correct += (dist <= threshold).astype(float)

    dists_arr = np.array(dists)
    flags_arr = np.array(flags)

    pck   = correct.sum() / (SAMPLES * 21)
    mpjpe = dists_arr.mean()
    mse   = np.mean(np.sum(dists_arr**2, axis=1) / 21)
    det   = (flags_arr >= 0.5).mean() * 100
    auc_v = [(dists_arr <= t).mean() for t in auc_thresholds]
    auc   = np.trapz(auc_v, auc_thresholds / (0.5*256))

    print(f'{name:<18} {pck:>8.4f} {mpjpe:>9.2f}px {mse:>10.2f} {auc:>8.4f} {det:>6.1f}%')

    all_results.append({
        'name': name, 'pck': pck, 'mpjpe': mpjpe,
        'mse': mse, 'auc': auc, 'det': det
    })

# Save CSV
with open('direct_sup_learning_curve.csv', 'w') as f:
    f.write('Checkpoint,PCK@0.2,MPJPE,MSE,AUC,Det_Rate\n')
    for r in all_results:
        f.write(f'{r["name"]},{r["pck"]:.4f},{r["mpjpe"]:.4f},'
                f'{r["mse"]:.4f},{r["auc"]:.4f},{r["det"]:.2f}\n')

print()
print('Saved: direct_sup_learning_curve.csv')

# Best checkpoint summary
if all_results:
    best = max(all_results, key=lambda x: x['pck'])
    print()
    print('='*55)
    print(f'BEST CHECKPOINT: {best["name"]}')
    print(f'  PCK@0.2 : {best["pck"]:.4f}')
    print(f'  MPJPE   : {best["mpjpe"]:.2f} px')
    print(f'  MSE     : {best["mse"]:.2f} px²')
    print(f'  AUC     : {best["auc"]:.4f}')
    print(f'  Det Rate: {best["det"]:.2f}%')
    print('='*55)

# Per-joint breakdown for best checkpoint
print()
print(f'Running per-joint breakdown on best checkpoint: {best["name"]}...')

ckpt_path = dict(checkpoints)[best["name"]]
student = BlazeHandLandmark().to(device)
student.load_state_dict(torch.load(ckpt_path, map_location=device))
student.eval()

correct_joint = np.zeros(21)
dists_joint   = []

for idx, batch in enumerate(loader):
    if idx >= SAMPLES: break
    img_tensor = torch.stack(batch['inputs']).float() / 255.0
    gt_raw = batch['data_samples'][0].gt_instances.keypoints[0]
    tc = np.clip(gt_raw, 0, 255).astype(np.float32)
    with torch.no_grad():
        _, _, pred = student(img_tensor.to(device))
        sc = pred[0, :, :2].cpu().numpy() * 256
    dist = np.sqrt(np.sum((sc - tc)**2, axis=1))
    dists_joint.append(dist)
    correct_joint += (dist <= threshold).astype(float)

dists_joint = np.array(dists_joint)
pck_j  = correct_joint / SAMPLES
mpjpe_j = dists_joint.mean(axis=0)

print()
print(f'Per-joint breakdown — {best["name"]}')
print(f'{"Joint":<15} {"PCK@0.2":>10} {"MPJPE (px)":>12}')
print('-'*40)
for i, name in enumerate(joint_names):
    print(f'{name:<15} {pck_j[i]:>10.4f} {mpjpe_j[i]:>12.4f}')