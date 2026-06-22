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

# UNTRAINED student - random weights, no checkpoint loaded
student = BlazeHandLandmark().to(device)
student.eval()

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
    data_root=r'C:\Users\Bea Juliana Poquiz\Desktop\mmpose_thesis\dataset\rhd',
    ann_file='annotations/rhd_test.json',
    pipeline=pipeline
)
loader = DataLoader(ds, batch_size=1, collate_fn=pseudo_collate)

threshold = 0.2 * 256
correct = np.zeros(21)
dists = []
total = 0

print('Evaluating UNTRAINED student on test split...')
for idx, batch in enumerate(loader):
    if idx >= 500: break
    img_tensor = torch.stack(batch['inputs']).float() / 255.0

    with torch.no_grad():
        out = teacher.predict(img_tensor.to(device), batch['data_samples'])
        hms = out[0].pred_fields.heatmaps.cpu().numpy()
        decoded = codec.decode(hms)
        if isinstance(decoded, dict):
            tc = np.array(decoded['keypoints']).reshape(21, 2)
        else:
            tc = np.array(decoded[0]).reshape(21, 2)

    with torch.no_grad():
        _, _, pred = student(img_tensor.to(device))
        sc = pred[0, :, :2].cpu().numpy() * 256

    dist = np.sqrt(np.sum((sc - tc)**2, axis=1))
    dists.append(dist)
    correct += (dist <= threshold).astype(float)
    total += 1

dists = np.array(dists)
print()
print('='*45)
print('UNTRAINED STUDENT BASELINE (500 test samples)')
print(f'PCK@0.2 : {correct.sum()/(total*21):.4f}')
print(f'MPJPE   : {dists.mean():.4f} px')
print(f'EPE     : {dists.mean():.4f} px')
print('='*45)