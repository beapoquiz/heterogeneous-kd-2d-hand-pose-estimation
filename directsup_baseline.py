import torch
import torch.nn as nn
import numpy as np
import sys
import os
from mmpose.datasets import Rhd2DDataset
from torch.utils.data import DataLoader
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
init_default_scope('mmpose')

def wing_loss(pred, target, w=10.0, epsilon=2.0):
    delta = (pred - target).abs()
    C = w - w * np.log(1 + w / epsilon)
    loss = torch.where(
        delta < w,
        w * torch.log(1 + delta / epsilon),
        delta - C
    )
    return loss.mean()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    STUDENT_FOLDER = os.path.join(os.getcwd(), 'student_model')
    if STUDENT_FOLDER not in sys.path:
        sys.path.append(STUDENT_FOLDER)
    from blazehand_landmark import BlazeHandLandmark

    # Start from MediaPipe weights — same as v2
    student = BlazeHandLandmark().to(device)
    student.load_state_dict(torch.load(
        'student_model/blazehand_landmark.pth', map_location=device))
    student.train()
    print('Loaded MediaPipe pretrained weights as starting point')

    DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'rhd')

    # Same augmented pipeline as v2 — identical conditions
    train_pipeline = [
        dict(type='LoadImage'),
        dict(type='GetBBoxCenterScale', padding=1.25),
        dict(type='RandomFlip', direction='horizontal'),
        dict(type='RandomBBoxTransform',
             shift_factor=0.1, shift_prob=0.3,
             scale_factor=(0.75, 1.25), scale_prob=1.0,
             rotate_factor=45, rotate_prob=0.6),
        dict(type='TopdownAffine', input_size=(256,256)),
        dict(type='PhotometricDistortion'),
        dict(type='PackPoseInputs')
    ]
    val_pipeline = [
        dict(type='LoadImage'),
        dict(type='GetBBoxCenterScale'),
        dict(type='TopdownAffine', input_size=(256,256)),
        dict(type='PackPoseInputs')
    ]

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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              collate_fn=pseudo_collate, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False,
                              collate_fn=pseudo_collate, num_workers=0)

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-6)

    EPOCHS      = 100
    VAL_EVERY   = 5
    VAL_SAMPLES = 200
    threshold   = 0.2 * 256
    best_pck    = 0.0
    best_epoch  = 0
    os.makedirs('checkpoints', exist_ok=True)

    print(f'\n{"="*55}')
    print('DIRECT SUPERVISION BASELINE')
    print('NO knowledge distillation — GT annotations only')
    print('Same augmentation + Wing Loss + LR schedule as v2')
    print('Starting weights: MediaPipe pretrained')
    print(f'{"="*55}\n')

    for epoch in range(EPOCHS):
        student.train()
        epoch_loss = 0.0

        for i, batch in enumerate(train_loader):
            images = torch.stack(batch['inputs']).to(device).float() / 255.0

            # Get ground truth keypoints directly from annotations
            # Already transformed to 256×256 space by the pipeline
            gt_list = []
            for ds in batch['data_samples']:
                kps = ds.gt_instances.keypoints  # [1, 21, 2]
                gt_list.append(kps[0])           # [21, 2]

            gt_targets = torch.from_numpy(
                np.stack(gt_list)).to(device).float()
            gt_targets = torch.clamp(gt_targets, 0, 255)

            optimizer.zero_grad()
            _, _, pred_landmarks = student(images)
            student_coords = pred_landmarks[:, :, :2] * 256
            gt_targets = gt_targets.view(student_coords.shape)

            loss = wing_loss(student_coords, gt_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if i % 20 == 0:
                print(f'Epoch [{epoch+1}/{EPOCHS}] '
                      f'Step [{i}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f}')

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f'===> Epoch {epoch+1} | Loss: {avg_loss:.6f} | '
              f'LR: {scheduler.get_last_lr()[0]:.2e}')

        if (epoch + 1) % 10 == 0:
            torch.save(student.state_dict(),
                       f'checkpoints/direct_sup_epoch_{epoch+1}.pth')
            print(f'    Checkpoint saved: direct_sup_epoch_{epoch+1}.pth')

        # Validate every 5 epochs on GT annotations
        if (epoch + 1) % VAL_EVERY == 0:
            student.eval()
            correct   = np.zeros(21)
            val_dists = []

            with torch.no_grad():
                for vidx, vbatch in enumerate(val_loader):
                    if vidx >= VAL_SAMPLES: break
                    vimg = torch.stack(vbatch['inputs']).float() / 255.0

                    # GT validation reference
                    vgt = vbatch['data_samples'][0].gt_instances.keypoints[0]
                    vgt_tensor = torch.clamp(
                        torch.from_numpy(vgt).float(), 0, 255)

                    _, _, vpred = student(vimg.to(device))
                    vsc  = vpred[0, :, :2].cpu().numpy() * 256
                    vtc  = vgt_tensor.numpy()

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
                           'checkpoints/direct_sup_best.pth')
                print(f'    >>> NEW BEST: {best_pck:.4f} '
                      f'at Epoch {best_epoch}')

            student.train()

    torch.save(student.state_dict(),
               f'checkpoints/direct_sup_epoch_{EPOCHS}.pth')
    print(f'\n{"="*55}')
    print('DIRECT SUPERVISION TRAINING COMPLETE')
    print(f'Best PCK@0.2: {best_pck:.4f} at Epoch {best_epoch}')
    print(f'Best model:   checkpoints/direct_sup_best.pth')
    print(f'{"="*55}')

if __name__ == '__main__':
    main()