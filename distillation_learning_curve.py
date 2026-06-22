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

SAMPLES = 200
threshold = 0.2 * 256
auc_thresholds = np.linspace(0, 0.5 * 256, 20)

# ALL available v2 checkpoints in order
checkpoints = [
    ('Epoch_10',       'checkpoints/distilled_v2_epoch_10.pth'),
    ('Epoch_20',       'checkpoints/distilled_v2_epoch_20.pth'),
    ('Epoch_30',       'checkpoints/distilled_v2_epoch_30.pth'),
    ('Epoch_40',       'checkpoints/distilled_v2_epoch_40.pth'),
    ('Epoch_50',       'checkpoints/distilled_v2_epoch_50.pth'),
    ('Epoch_60',       'checkpoints/distilled_v2_epoch_60.pth'),
    ('Epoch_65_BEST',  'checkpoints/distilled_v2_best.pth'),
    ('Epoch_70',       'checkpoints/distilled_v2_epoch_70.pth'),
    ('Epoch_80',       'checkpoints/distilled_v2_epoch_80.pth'),
    ('Epoch_90',       'checkpoints/distilled_v2_epoch_90.pth'),
    ('Epoch_100',      'checkpoints/distilled_v2_epoch_100.pth'),
]

print('V2 Complete Learning Curve — All Checkpoints')
print(f'Samples per checkpoint: {SAMPLES}')
print()
print(f'{"Checkpoint":<16} {"PCK@0.2":>8} {"MPJPE":>9} {"EPE":>8} {"MSE":>10} {"AUC":>8}')
print('-'*65)

all_results = []

for name, ckpt_path in checkpoints:
    if not os.path.exists(ckpt_path):
        print(f'{name:<16} NOT FOUND — skipping')
        continue

    student = BlazeHandLandmark().to(device)
    student.load_state_dict(torch.load(ckpt_path, map_location=device))
    student.eval()

    correct = np.zeros(21)
    dists   = []

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

    dists_arr = np.array(dists)

    pck   = correct.sum() / (SAMPLES * 21)
    mpjpe = dists_arr.mean()
    epe   = mpjpe
    mse   = np.mean(np.sum(dists_arr**2, axis=1) / 21)
    auc_v = [(dists_arr <= t).mean() for t in auc_thresholds]
    auc   = np.trapz(auc_v, auc_thresholds / (0.5 * 256))

    print(f'{name:<16} {pck:>8.4f} {mpjpe:>9.2f}px {epe:>8.2f}px {mse:>10.2f} {auc:>8.4f}')

    all_results.append({
        'name': name, 'pck': pck, 'mpjpe': mpjpe,
        'epe': epe, 'mse': mse, 'auc': auc
    })

# Save CSV
with open('distillation_learning_curve.csv', 'w') as f:
    f.write('Checkpoint,Epoch,PCK@0.2,MPJPE,EPE,MSE,AUC\n')
    epoch_map = {
        'Epoch_10': 10, 'Epoch_20': 20, 'Epoch_30': 30,
        'Epoch_40': 40, 'Epoch_50': 50, 'Epoch_60': 60,
        'Epoch_65_BEST': 65, 'Epoch_70': 70, 'Epoch_80': 80,
        'Epoch_90': 90, 'Epoch_100': 100
    }
    for r in all_results:
        ep = epoch_map.get(r['name'], 0)
        f.write(f'{r["name"]},{ep},{r["pck"]:.4f},{r["mpjpe"]:.4f},'
                f'{r["epe"]:.4f},{r["mse"]:.4f},{r["auc"]:.4f}\n')

print()
print('Saved: distillation_learning_curve.csv')

# Best summary
best = max(all_results, key=lambda x: x['pck'])
print()
print('='*55)
print(f'BEST CHECKPOINT: {best["name"]}')
print(f'  PCK@0.2 : {best["pck"]:.4f}')
print(f'  MPJPE   : {best["mpjpe"]:.2f} px')
print(f'  EPE     : {best["epe"]:.2f} px')
print(f'  MSE     : {best["mse"]:.2f} px²')
print(f'  AUC     : {best["auc"]:.4f}')
print('='*55)