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

# Only use 200 samples for speed
SAMPLES = 200
threshold = 0.2 * 256

# All checkpoints in order
checkpoints = [
    ('Epoch_10',  'checkpoints/distilled_student_epoch_10.pth'),
    ('Epoch_20',  'checkpoints/distilled_student_epoch_20.pth'),
    ('Epoch_30',  'checkpoints/distilled_student_epoch_30.pth'),
    ('Epoch_40',  'checkpoints/distilled_student_epoch_40.pth'),
    ('Epoch_50',  'checkpoints/distilled_student_epoch_50.pth'),
    ('Epoch_60',  'checkpoints/distilled_student_epoch_60.pth'),
    ('Epoch_70',  'checkpoints/distilled_student_epoch_70.pth'),
    ('Epoch_80',  'checkpoints/distilled_student_epoch_80.pth'),
    ('Epoch_90',  'checkpoints/distilled_student_epoch_90.pth'),
    ('Epoch_100', 'checkpoints/distilled_student_epoch_100.pth'),
    ('Epoch_110', 'checkpoints/distilled_student_epoch_110.pth'),
    ('Epoch_120', 'checkpoints/distilled_student_epoch_120.pth'),
    ('Epoch_130', 'checkpoints/distilled_student_epoch_130.pth'),
    ('Epoch_140', 'checkpoints/distilled_student_epoch_140.pth'),
    ('Epoch_150', 'checkpoints/distilled_student_epoch_150.pth'),
    ('Epoch_160', 'checkpoints/distilled_student_epoch_160.pth'),
    ('Epoch_170', 'checkpoints/distilled_student_epoch_170.pth'),
    ('Epoch_180', 'checkpoints/distilled_student_epoch_180.pth'),
    ('Epoch_190', 'checkpoints/distilled_student_epoch_190.pth'),
    ('Epoch_200', 'checkpoints/distilled_student_epoch_200.pth'),
    ('Epoch_210', 'checkpoints/distilled_student_epoch_210.pth'),
]

print('Evaluating learning curve across all checkpoints...')
print(f'Using {SAMPLES} test samples per checkpoint')
print()
print(f'{"Checkpoint":<12} {"PCK@0.2":>10} {"MPJPE":>12}')
print('-' * 38)

results = []
for name, ckpt_path in checkpoints:
    student = BlazeHandLandmark().to(device)
    student.load_state_dict(torch.load(ckpt_path, map_location=device))
    student.eval()

    correct = np.zeros(21)
    dists = []

    for idx, batch in enumerate(loader):
        if idx >= SAMPLES: break
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

    pck   = correct.sum() / (SAMPLES * 21)
    mpjpe = np.array(dists).mean()
    results.append((name, pck, mpjpe))
    print(f'{name:<12} {pck:>10.4f} {mpjpe:>10.2f}px')

# Save learning curve
with open('learning_curve.txt', 'w') as f:
    f.write('Checkpoint,PCK@0.2,MPJPE\n')
    for name, pck, mpjpe in results:
        f.write(f'{name},{pck:.4f},{mpjpe:.4f}\n')

print()
print('Saved: learning_curve.txt')
best = max(results, key=lambda x: x[1])
print(f'Best checkpoint: {best[0]} with PCK@0.2={best[1]:.4f}, MPJPE={best[2]:.2f}px')