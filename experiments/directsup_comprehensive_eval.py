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
print(f'Device: {device}')

student = BlazeHandLandmark().to(device)
student.load_state_dict(torch.load(
    'checkpoints/direct_sup_best.pth', map_location=device))
student.eval()
print('Loaded: checkpoints/direct_sup_best.pth')

CONFIG = 'mmpose/configs/hand_2d_keypoint/topdown_heatmap/rhd2d/td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256.py'
CHECKPOINT = 'checkpoints/hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth'
teacher = init_model(CONFIG, CHECKPOINT, device=device)
teacher.cfg.model.test_cfg.output_heatmaps = True
if hasattr(teacher, 'head') and hasattr(teacher.head, 'test_cfg'):
    teacher.head.test_cfg['output_heatmaps'] = True
teacher.eval()
print('Loaded teacher: HRNetV2-W18 + DARK')

codec = MSRAHeatmap(input_size=(256,256), heatmap_size=(64,64), sigma=2, unbiased=True)

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

N = len(ds)
threshold = 0.2 * 256
auc_thresholds = np.linspace(0, 0.5*256, 20)
correct = np.zeros(21)
dists = []
flags = []

print(f'Direct Supervision Comprehensive Eval — Teacher Coords Reference')
print(f'Full RHD test set: {N} samples')
print()

for idx, batch in enumerate(loader):
    img_tensor = torch.stack(batch['inputs']).float() / 255.0

    with torch.no_grad():
        out = teacher.predict(img_tensor.to(device), batch['data_samples'])
        hms = out[0].pred_fields.heatmaps.cpu().numpy()
        decoded = codec.decode(hms)
        if isinstance(decoded, dict):
            tc = np.array(decoded['keypoints']).reshape(21, 2)
        else:
            tc = np.array(decoded[0]).reshape(21, 2)
        tc = np.clip(tc, 0, 255)

    with torch.no_grad():
        flag, _, pred = student(img_tensor.to(device))
        sc = pred[0, :, :2].cpu().numpy() * 256
        flags.append(flag.item())

    dist = np.sqrt(np.sum((sc - tc)**2, axis=1))
    dists.append(dist)
    correct += (dist <= threshold).astype(float)

    if (idx+1) % 500 == 0:
        print(f'  [{idx+1}/{N}] Running PCK@0.2: {correct.sum()/((idx+1)*21):.4f}')

dists = np.array(dists)
flags = np.array(flags)
pck   = correct.sum() / (N * 21)
mpjpe = dists.mean()
mse   = np.mean(np.sum(dists**2, axis=1) / 21)
det   = (flags >= 0.5).mean() * 100
auc_v = [(dists <= t).mean() for t in auc_thresholds]
auc   = np.trapz(auc_v, auc_thresholds / (0.5*256))

joint_names = ['Wrist',
               'Thumb_MCP','Thumb_PIP','Thumb_DIP','Thumb_Tip',
               'Index_MCP','Index_PIP','Index_DIP','Index_Tip',
               'Middle_MCP','Middle_PIP','Middle_DIP','Middle_Tip',
               'Ring_MCP','Ring_PIP','Ring_DIP','Ring_Tip',
               'Pinky_MCP','Pinky_PIP','Pinky_DIP','Pinky_Tip']

print()
print('='*60)
print('DIRECT SUPERVISION BEST (Epoch 25) — TEACHER COORDS REFERENCE')
print(f'Dataset: RHD test split ({N} samples)')
print('Reference: Teacher (HRNetV2-W18 + DARK) decoded coordinates')
print('='*60)
print(f'PCK@0.2      : {pck:.4f} ({pck*100:.2f}%)')
print(f'MPJPE        : {mpjpe:.4f} px')
print(f'MSE          : {mse:.4f} px²')
print(f'AUC          : {auc:.4f}')
print(f'Det. Rate    : {det:.2f}%')
print()
print(f'{"Joint":<15} {"PCK@0.2":>10} {"MPJPE (px)":>12}')
print('-'*40)
pck_per_joint   = correct / N
mpjpe_per_joint = dists.mean(axis=0)
for i, name in enumerate(joint_names[:21]):
    print(f'{name:<15} {pck_per_joint[i]:>10.4f} {mpjpe_per_joint[i]:>12.4f}')
print('='*60)

with open('results_direct_sup_TC.txt', 'w') as f:
    f.write('Direct Supervision Best (Epoch 25) — Teacher Coords Reference\n')
    f.write(f'Dataset: RHD test split ({N} samples)\n')
    f.write('Reference: Teacher (HRNetV2-W18 + DARK) decoded coordinates\n\n')
    f.write(f'PCK@0.2:    {pck:.4f}\n')
    f.write(f'MPJPE:      {mpjpe:.4f} px\n')
    f.write(f'MSE:        {mse:.4f} px2\n')
    f.write(f'AUC:        {auc:.4f}\n')
    f.write(f'Det Rate:   {det:.2f}%\n\n')
    f.write(f'{"Joint":<15} {"PCK@0.2":>10} {"MPJPE":>12}\n')
    f.write('-'*40 + '\n')
    for i, name in enumerate(joint_names[:21]):
        f.write(f'{name:<15} {pck_per_joint[i]:>10.4f} {mpjpe_per_joint[i]:>12.4f}\n')

print('Saved: results_direct_sup_TC.txt')
