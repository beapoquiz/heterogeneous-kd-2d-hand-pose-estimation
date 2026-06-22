import torch, sys, os, numpy as np
from mmpose.apis import init_model
from mmpose.codecs import MSRAHeatmap
from mmpose.datasets import Rhd2DDataset
from torch.utils.data import DataLoader
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope

init_default_scope('mmpose')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG     = 'mmpose/configs/hand_2d_keypoint/topdown_heatmap/rhd2d/td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256.py'
CHECKPOINT = 'checkpoints/hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth'

print('Loading Teacher (HRNetV2-W18 DARK)...')
teacher = init_model(CONFIG, CHECKPOINT, device=device)
teacher.cfg.model.test_cfg.output_heatmaps = True
if hasattr(teacher, 'head') and hasattr(teacher.head, 'test_cfg'):
    if isinstance(teacher.head.test_cfg, dict):
        teacher.head.test_cfg['output_heatmaps'] = True
    else:
        teacher.head.test_cfg.output_heatmaps = True
teacher.eval()

codec = MSRAHeatmap(
    input_size=(256, 256),
    heatmap_size=(64, 64),
    sigma=2,
    unbiased=True
)

pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(type='PackPoseInputs')
]
ds = Rhd2DDataset(
    data_root=r'C:\Users\Bea Juliana Poquiz\Desktop\mmpose_thesis\dataset\rhd',
    ann_file='annotations/rhd_test.json',
    pipeline=pipeline
)
loader = DataLoader(ds, batch_size=1, collate_fn=pseudo_collate)

N              = len(ds)
threshold      = 0.2 * 256
auc_thresholds = np.linspace(0, 0.5 * 256, 20)
correct        = np.zeros(21)
dists          = []

print(f'Teacher GT Eval — HRNetV2-W18 DARK (pretrained on RHD2D)')
print(f'Reference  : Ground Truth annotations')
print(f'Dataset    : RHD test split ({N} samples)')
print(f'Threshold  : PCK@0.2 = {threshold:.1f} px (fixed, 0.2 × 256)')
print()

for idx, batch in enumerate(loader):
    img_tensor = torch.stack(batch['inputs']).float() / 255.0

    gt_kps = batch['data_samples'][0].gt_instances.keypoints
    gc = np.array(gt_kps).reshape(21, 2)

    with torch.no_grad():
        output_samples = teacher.predict(img_tensor.to(device), batch['data_samples'])
        hms = output_samples[0].pred_fields.heatmaps
        if torch.is_tensor(hms):
            hms = hms.detach().cpu().numpy()
        decoded = codec.decode(hms)
        if isinstance(decoded, dict):
            tc = decoded.get('keypoints', decoded.get('pred_keypoints'))
            if isinstance(tc, list):
                tc = tc[0]
            tc = np.array(tc).reshape(21, 2)
        else:
            tc = np.array(decoded[0]).reshape(21, 2)

    dist = np.sqrt(np.sum((tc - gc) ** 2, axis=1))
    dists.append(dist)
    correct += (dist <= threshold).astype(float)

    if (idx + 1) % 500 == 0:
        print(f'  [{idx+1}/{N}] Running PCK@0.2 (Teacher vs GT): {correct.sum()/((idx+1)*21):.4f}')

dists = np.array(dists)
pck   = correct.sum() / (N * 21)
mpjpe = dists.mean()
mse   = np.mean(np.sum(dists ** 2, axis=1) / 21)
auc_v = [(dists <= t).mean() for t in auc_thresholds]
auc   = np.trapz(auc_v, auc_thresholds / (0.5 * 256))

joint_names = ['Wrist',
               'Thumb_MCP', 'Thumb_PIP', 'Thumb_DIP', 'Thumb_Tip',
               'Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_Tip',
               'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_Tip',
               'Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Ring_Tip',
               'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP', 'Pinky_Tip']

pck_per_joint   = correct / N
mpjpe_per_joint = dists.mean(axis=0)

print()
print('=' * 55)
print('TEACHER (HRNetV2-W18 DARK) — GT EVALUATION')
print('Pretrained on RHD2D | No finetuning | Official checkpoint')
print(f'Dataset: RHD test split ({N} samples)')
print('Reference: Ground Truth annotations')
print('Threshold: PCK@0.2 = 51.2 px fixed (0.2 × 256)')
print('=' * 55)
print(f'PCK@0.2      : {pck:.4f} ({pck*100:.2f}%)')
print(f'MPJPE        : {mpjpe:.4f} px')
print(f'MSE          : {mse:.4f} px²')
print(f'AUC          : {auc:.4f}')
print(f'Det. Rate    : 100.00%  (top-down: bbox always provided)')
print()
print(f'{"Joint":<15} {"PCK@0.2":>10} {"MPJPE (px)":>12}')
print('-' * 40)
for i, name in enumerate(joint_names):
    print(f'{name:<15} {pck_per_joint[i]:>10.4f} {mpjpe_per_joint[i]:>12.4f}')
print('=' * 55)

with open('results_teacher_gt.txt', 'w') as f:
    f.write('Teacher (HRNetV2-W18 DARK) — GT Evaluation\n')
    f.write('Pretrained on RHD2D | No finetuning | Official checkpoint\n')
    f.write(f'Dataset: RHD test split ({N} samples)\n')
    f.write('Reference: Ground Truth annotations\n')
    f.write('Threshold: PCK@0.2 = 51.2 px fixed (0.2 x 256)\n\n')
    f.write(f'PCK@0.2:    {pck:.4f}\n')
    f.write(f'MPJPE:      {mpjpe:.4f} px\n')
    f.write(f'MSE:        {mse:.4f} px2\n')
    f.write(f'AUC:        {auc:.4f}\n')
    f.write(f'Det Rate:   100.00%\n\n')
    f.write(f'{"Joint":<15} {"PCK@0.2":>10} {"MPJPE":>12}\n')
    f.write('-' * 40 + '\n')
    for i, name in enumerate(joint_names):
        f.write(f'{name:<15} {pck_per_joint[i]:>10.4f} {mpjpe_per_joint[i]:>12.4f}\n')

print('Saved: results_teacher_gt.txt')
