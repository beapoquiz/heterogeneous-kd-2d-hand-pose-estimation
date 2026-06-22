import torch
import torch.nn as nn
import numpy as np
import sys
import os
from mmpose.apis import init_model
from mmpose.codecs import MSRAHeatmap
from mmpose.datasets import Rhd2DDataset
from torch.utils.data import DataLoader
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope

def wing_loss(pred, target, w=10.0, epsilon=2.0):
    delta = (pred - target).abs()
    C = w - w * np.log(1 + w / epsilon)
    loss = torch.where(
        delta < w,
        w * torch.log(1 + delta / epsilon),
        delta - C
    )
    return loss.mean()

BONES = [(0,1),(1,2),(2,3),(3,4),
         (0,5),(5,6),(6,7),(7,8),
         (0,9),(9,10),(10,11),(11,12),
         (0,13),(13,14),(14,15),(15,16),
         (0,17),(17,18),(18,19),(19,20)]

def bone_loss(pred, target):
    loss = 0
    for i, j in BONES:
        pred_len = torch.norm(pred[:,i] - pred[:,j], dim=1)
        tgt_len  = torch.norm(target[:,i] - target[:,j], dim=1)
        loss += torch.nn.functional.mse_loss(pred_len, tgt_len)
    return loss / len(BONES)

def main():
    init_default_scope('mmpose')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    STUDENT_FOLDER = os.path.join(os.getcwd(), 'student_model')
    if STUDENT_FOLDER not in sys.path:
        sys.path.append(STUDENT_FOLDER)
    from blazehand_landmark import BlazeHandLandmark

    # --- Teacher ---
    print('Loading Teacher...')
    CONFIG = 'mmpose/configs/hand_2d_keypoint/topdown_heatmap/rhd2d/td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256.py'
    CHECKPOINT = 'checkpoints/hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth'
    teacher = init_model(CONFIG, CHECKPOINT, device=device)
    teacher.cfg.model.test_cfg.output_heatmaps = True
    if hasattr(teacher, 'head') and hasattr(teacher.head, 'test_cfg'):
        if isinstance(teacher.head.test_cfg, dict):
            teacher.head.test_cfg['output_heatmaps'] = True
        else:
            teacher.head.test_cfg.output_heatmaps = True
    teacher.eval()
    print('Teacher loaded OK')

    # --- Student from MediaPipe weights ---
    print('Loading Student from MediaPipe weights...')
    student = BlazeHandLandmark().to(device)
    student.load_state_dict(torch.load(
        'student_model/blazehand_landmark.pth', map_location=device))
    student.train()
    print('Student loaded OK')

    # --- Codec ---
    codec = MSRAHeatmap(
        input_size=(256,256), heatmap_size=(64,64), sigma=2, unbiased=True)

    # --- Augmented pipeline ---
    pipeline = [
        dict(type='LoadImage'),
        dict(type='GetBBoxCenterScale', padding=1.25),
        dict(type='RandomFlip', direction='horizontal'),
        dict(type='RandomBBoxTransform',
             shift_factor=0.1,
             shift_prob=0.3,
             scale_factor=(0.75, 1.25),
             scale_prob=1.0,
             rotate_factor=45,
             rotate_prob=0.6),
        dict(type='TopdownAffine', input_size=(256,256)),
        dict(type='PhotometricDistortion'),
        dict(type='PackPoseInputs')
    ]

    DATA_ROOT = r'C:\Users\Bea Juliana Poquiz\Desktop\mmpose_thesis\dataset\rhd'
    ds = Rhd2DDataset(
        data_root=DATA_ROOT,
        ann_file='annotations/rhd_train.json',
        pipeline=pipeline
    )
    loader = DataLoader(ds, batch_size=4, shuffle=True,
                        collate_fn=pseudo_collate, num_workers=0)

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-6)

    print()
    print('--- DRY RUN: 2 batches only ---')
    print('Testing full pipeline end to end...')
    print()

    for i, batch in enumerate(loader):
        if i >= 2: break

        images = torch.stack(batch['inputs']).to(device).float() / 255.0
        print(f'Batch {i+1} — images shape: {images.shape}')

        with torch.no_grad():
            output_samples = teacher.predict(images, batch['data_samples'])
            batch_coords = []
            for sample in output_samples:
                hms = None
                if hasattr(sample, 'pred_fields'):
                    hms = sample.pred_fields.get(
                        'heatmaps',
                        sample.pred_fields.get('pred_heatmaps', None))
                if hms is None:
                    raise RuntimeError('Teacher produced no heatmaps')
                if torch.is_tensor(hms):
                    hms = hms.detach().cpu().numpy()
                decoded = codec.decode(hms)
                if isinstance(decoded, dict):
                    coords = decoded['keypoints']
                    if isinstance(coords, list):
                        coords = coords[0]
                    coords = np.array(coords).reshape(21, 2)
                else:
                    coords = np.array(decoded[0]).reshape(21, 2)
                coords = np.clip(coords, 0, 255)
                batch_coords.append(coords)

            teacher_targets = torch.from_numpy(
                np.stack(batch_coords)).to(device).float()
            if teacher_targets.dim() == 4:
                teacher_targets = teacher_targets.squeeze(1)

        print(f'  Teacher targets shape: {teacher_targets.shape}')
        print(f'  Teacher targets range: {teacher_targets.min():.1f} to {teacher_targets.max():.1f}')

        optimizer.zero_grad()
        _, _, pred_landmarks = student(images)
        student_coords = pred_landmarks[:, :, :2] * 256
        teacher_targets = teacher_targets.view(student_coords.shape)

        print(f'  Student coords shape: {student_coords.shape}')
        print(f'  Student coords range: {student_coords.min():.1f} to {student_coords.max():.1f}')

        loss_w = wing_loss(student_coords, teacher_targets)
        loss_b = bone_loss(student_coords, teacher_targets)
        loss   = loss_w + 0.01 * loss_b

        loss.backward()
        optimizer.step()

        print(f'  Wing loss: {loss_w.item():.4f}')
        print(f'  Bone loss: {loss_b.item():.4f}')
        print(f'  Total loss: {loss.item():.4f}')
        print()

    scheduler.step()
    print('='*45)
    print('DRY RUN COMPLETE — all systems working')
    print('Safe to start full distillation_v2.py')
    print('='*45)

if __name__ == '__main__':
    main()