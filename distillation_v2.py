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
    print(f'Using device: {device}')

    STUDENT_FOLDER = os.path.join(os.getcwd(), 'student_model')
    if STUDENT_FOLDER not in sys.path:
        sys.path.append(STUDENT_FOLDER)
    from blazehand_landmark import BlazeHandLandmark

    # --- Teacher ---
    print('Loading Teacher (HRNet)...')
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

    # --- Student: Start from MediaPipe weights ---
    print('Loading Student from MediaPipe pretrained weights...')
    student = BlazeHandLandmark().to(device)
    student.load_state_dict(torch.load(
        'student_model/blazehand_landmark.pth', map_location=device))
    print('MediaPipe weights loaded successfully')
    student.train()

    # --- Codec ---
    codec = MSRAHeatmap(
        input_size=(256,256),
        heatmap_size=(64,64),
        sigma=2,
        unbiased=True
    )

    DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'rhd')

    # --- Augmented Training Pipeline ---
    train_pipeline = [
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

    # --- Validation Pipeline (no augmentation) ---
    val_pipeline = [
        dict(type='LoadImage'),
        dict(type='GetBBoxCenterScale'),
        dict(type='TopdownAffine', input_size=(256,256)),
        dict(type='PackPoseInputs')
    ]

    print('Preparing datasets...')
    train_dataset = Rhd2DDataset(
        data_root=DATA_ROOT,
        ann_file='annotations/rhd_train.json',
        pipeline=train_pipeline
    )
    val_dataset = Rhd2DDataset(
        data_root=DATA_ROOT,
        ann_file='annotations/rhd_test.json',
        pipeline=val_pipeline
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=pseudo_collate,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=pseudo_collate,
        num_workers=0
    )

    # --- Optimizer and scheduler ---
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-6
    )

    EPOCHS    = 100
    VAL_EVERY = 5
    VAL_SAMPLES = 200
    threshold = 0.2 * 256
    best_pck  = 0.0
    best_epoch = 0
    os.makedirs('checkpoints', exist_ok=True)

    print(f'\n{"="*55}')
    print('DISTILLATION V2 — FINAL CONFIGURATION')
    print('Start weights : MediaPipe blazehand_landmark.pth')
    print('Loss          : Wing(w=10) + 0.002 * Bone')
    print('Augmentation  : Flip + BBoxTransform + PhotoDistort')
    print('LR schedule   : Cosine 1e-4 -> 1e-6 over 100 epochs')
    print('Validation    : Every 5 epochs, 200 test samples')
    print('Best model    : Auto-saved to distilled_v2_best.pth')
    print(f'{"="*55}\n')

    for epoch in range(EPOCHS):
        student.train()
        epoch_loss = 0.0
        epoch_wing = 0.0
        epoch_bone = 0.0

        for i, batch in enumerate(train_loader):
            images = torch.stack(batch['inputs']).to(device).float() / 255.0

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
                        raise RuntimeError('Teacher produced no heatmaps.')
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

            optimizer.zero_grad()
            _, _, pred_landmarks = student(images)
            student_coords = pred_landmarks[:, :, :2] * 256
            teacher_targets = teacher_targets.view(student_coords.shape)

            loss_w = wing_loss(student_coords, teacher_targets)
            loss_b = bone_loss(student_coords, teacher_targets)
            loss   = loss_w + 0.002 * loss_b  # calibrated weight

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_wing += loss_w.item()
            epoch_bone += loss_b.item()

            if i % 20 == 0:
                print(f'Epoch [{epoch+1}/{EPOCHS}] '
                      f'Step [{i}/{len(train_loader)}] '
                      f'Loss:{loss.item():.4f} '
                      f'Wing:{loss_w.item():.4f} '
                      f'Bone:{loss_b.item():.4f}')

        scheduler.step()
        avg_loss  = epoch_loss / len(train_loader)
        avg_wing  = epoch_wing / len(train_loader)
        avg_bone  = epoch_bone / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        print(f'===> Epoch {epoch+1} | '
              f'Loss:{avg_loss:.4f} | '
              f'Wing:{avg_wing:.4f} | '
              f'Bone:{avg_bone:.4f} | '
              f'LR:{current_lr:.2e}')

        # Save every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt = f'checkpoints/distilled_v2_epoch_{epoch+1}.pth'
            torch.save(student.state_dict(), ckpt)
            print(f'    Checkpoint saved: {ckpt}')

        # Validate every 5 epochs
        if (epoch + 1) % VAL_EVERY == 0:
            student.eval()
            correct   = np.zeros(21)
            val_dists = []

            with torch.no_grad():
                for vidx, vbatch in enumerate(val_loader):
                    if vidx >= VAL_SAMPLES: break
                    vimg = torch.stack(vbatch['inputs']).float() / 255.0

                    vout = teacher.predict(
                        vimg.to(device), vbatch['data_samples'])
                    vhms = vout[0].pred_fields.heatmaps.cpu().numpy()
                    vdec = codec.decode(vhms)
                    if isinstance(vdec, dict):
                        vtc = np.array(vdec['keypoints']).reshape(21, 2)
                    else:
                        vtc = np.array(vdec[0]).reshape(21, 2)

                    _, _, vpred = student(vimg.to(device))
                    vsc = vpred[0, :, :2].cpu().numpy() * 256

                    vdist = np.sqrt(np.sum((vsc - vtc)**2, axis=1))
                    val_dists.append(vdist)
                    correct += (vdist <= threshold).astype(float)

            val_pck   = correct.sum() / (VAL_SAMPLES * 21)
            val_mpjpe = np.array(val_dists).mean()
            print(f'    >>> VAL PCK@0.2: {val_pck:.4f} | '
                  f'MPJPE: {val_mpjpe:.2f}px')

            if val_pck > best_pck:
                best_pck   = val_pck
                best_epoch = epoch + 1
                torch.save(student.state_dict(),
                           'checkpoints/distilled_v2_best.pth')
                print(f'    >>> NEW BEST: PCK@0.2={best_pck:.4f} '
                      f'at Epoch {best_epoch} — saved distilled_v2_best.pth')

            student.train()

    # Final save
    torch.save(student.state_dict(),
               f'checkpoints/distilled_v2_epoch_{EPOCHS}.pth')
    print(f'\n{"="*55}')
    print('TRAINING COMPLETE')
    print(f'Best PCK@0.2 : {best_pck:.4f} at Epoch {best_epoch}')
    print(f'Best model   : checkpoints/distilled_v2_best.pth')
    print(f'{"="*55}')

if __name__ == '__main__':
    main()