import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import sys, os
from mmpose.apis import init_model
from mmpose.codecs import MSRAHeatmap
from mmpose.datasets import Rhd2DDataset
from torch.utils.data import DataLoader
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE, 'student_model'))
from blazehand_landmark import BlazeHandLandmark

def wing_loss(pred, target, w=10.0, epsilon=2.0):
    delta = (pred - target).abs()
    C = w - w * np.log(1 + w / epsilon)
    return torch.where(delta < w,
                       w * torch.log(1 + delta / epsilon),
                       delta - C).mean()

BONES = [(0,1),(1,2),(2,3),(3,4),
         (0,5),(5,6),(6,7),(7,8),
         (0,9),(9,10),(10,11),(11,12),
         (0,13),(13,14),(14,15),(15,16),
         (0,17),(17,18),(18,19),(19,20)]

def bone_loss(pred, target):
    losses = [torch.nn.functional.mse_loss(
                  torch.norm(pred[:,i] - pred[:,j], dim=1),
                  torch.norm(target[:,i] - target[:,j], dim=1))
              for i, j in BONES]
    return torch.stack(losses).mean()

def main():
    init_default_scope('mmpose')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ── Teacher ────────────────────────────────────────────
    print('Loading Teacher (HRNetV2-W18)...')
    CONFIG     = os.path.join(BASE, 'mmpose/configs/hand_2d_keypoint/topdown_heatmap/rhd2d/td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256.py')
    CHECKPOINT = os.path.join(BASE, 'checkpoints/hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth')
    teacher = init_model(CONFIG, CHECKPOINT, device=device)
    teacher.cfg.model.test_cfg.output_heatmaps = True
    if hasattr(teacher, 'head') and hasattr(teacher.head, 'test_cfg'):
        if isinstance(teacher.head.test_cfg, dict):
            teacher.head.test_cfg['output_heatmaps'] = True
        else:
            teacher.head.test_cfg.output_heatmaps = True
    teacher.eval()

    # ── Student — start from v2 epoch 100 ─────────────────
    print('Loading Student from distilled_v2_epoch_100.pth...')
    student = BlazeHandLandmark().to(device)
    student.load_state_dict(torch.load(
        os.path.join(BASE, 'checkpoints/distilled_v2_epoch_100.pth'),
        map_location=device))
    student.train()

    # ── Codec ──────────────────────────────────────────────
    codec = MSRAHeatmap(input_size=(256,256), heatmap_size=(64,64),
                        sigma=2, unbiased=True)

    DATA_ROOT = os.path.join(BASE, 'dataset', 'rhd')

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

    print('Preparing datasets...')
    train_dataset = Rhd2DDataset(
        data_root=DATA_ROOT,
        ann_file='annotations/rhd_train.json',
        pipeline=train_pipeline)
    val_dataset = Rhd2DDataset(
        data_root=DATA_ROOT,
        ann_file='annotations/rhd_test.json',
        pipeline=val_pipeline)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              collate_fn=pseudo_collate, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=1,  shuffle=False,
                              collate_fn=pseudo_collate, num_workers=0)

    # ── Optimizer — small LR reset for finetuning ─────────
    # Cosine 5e-5 → 1e-6 over 50 epochs gives the model a
    # second wind without overwriting what epoch 100 learned.
    optimizer = torch.optim.Adam(student.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6)

    EPOCHS      = 50
    START_EPOCH = 100
    VAL_EVERY   = 5
    VAL_SAMPLES = 200
    threshold   = 0.2 * 256
    best_pck    = 0.0
    best_epoch  = START_EPOCH
    patience    = 5          # stop if no PCK improvement for 5 val checks (= 25 epochs)
    no_improve  = 0
    os.makedirs(os.path.join(BASE, 'checkpoints'), exist_ok=True)

    print(f'\n{"="*55}')
    print('DISTILLATION V2 — FINETUNE FROM EPOCH 100')
    print('Start weights  : distilled_v2_epoch_100.pth')
    print('Loss           : Wing(w=10) + 0.002 * Bone')
    print('LR schedule    : Cosine 5e-5 -> 1e-6 over 50 epochs')
    print('Augmentation   : Flip + BBoxTransform + PhotoDistort')
    print('Validation     : Every 5 epochs, 200 test samples')
    print(f'{"="*55}\n')

    for epoch in range(EPOCHS):
        student.train()
        epoch_loss = epoch_wing = epoch_bone = 0.0

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
                        coords = decoded.get('keypoints', decoded.get('pred_keypoints'))
                        if isinstance(coords, list):
                            coords = coords[0]
                        coords = np.array(coords).reshape(21, 2)
                    else:
                        coords = np.array(decoded[0]).reshape(21, 2)
                    batch_coords.append(np.clip(coords, 0, 255))

                teacher_targets = torch.from_numpy(
                    np.stack(batch_coords)).to(device).float()
                if teacher_targets.dim() == 4:
                    teacher_targets = teacher_targets.squeeze(1)

            optimizer.zero_grad()
            _, _, pred_landmarks = student(images)
            student_coords  = pred_landmarks[:, :, :2] * 256
            teacher_targets = teacher_targets.view(student_coords.shape)

            loss_w = wing_loss(student_coords, teacher_targets)
            loss_b = bone_loss(student_coords, teacher_targets)
            loss   = loss_w + 0.002 * loss_b

            loss.backward()
            clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_wing += loss_w.item()
            epoch_bone += loss_b.item()

            if i % 20 == 0:
                print(f'Epoch [{START_EPOCH+epoch+1}/{START_EPOCH+EPOCHS}] '
                      f'Step [{i}/{len(train_loader)}] '
                      f'Loss:{loss.item():.4f} '
                      f'Wing:{loss_w.item():.4f} '
                      f'Bone:{loss_b.item():.4f}')

        scheduler.step()
        n = len(train_loader)
        print(f'===> Epoch {START_EPOCH+epoch+1} | '
              f'Loss:{epoch_loss/n:.4f} | '
              f'Wing:{epoch_wing/n:.4f} | '
              f'Bone:{epoch_bone/n:.4f} | '
              f'LR:{scheduler.get_last_lr()[0]:.2e}')

        # Save every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt = os.path.join(BASE, 'checkpoints',
                                f'distilled_v2_ft_epoch_{START_EPOCH+epoch+1}.pth')
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

                    vout = teacher.predict(vimg.to(device), vbatch['data_samples'])
                    vhms = vout[0].pred_fields.heatmaps.cpu().numpy()
                    vdec = codec.decode(vhms)
                    if isinstance(vdec, dict):
                        vtc = np.array(vdec.get('keypoints')).reshape(21, 2)
                    else:
                        vtc = np.array(vdec[0]).reshape(21, 2)

                    _, _, vpred = student(vimg.to(device))
                    vsc = vpred[0, :, :2].cpu().numpy() * 256

                    vdist = np.sqrt(np.sum((vsc - vtc)**2, axis=1))
                    val_dists.append(vdist)
                    correct += (vdist <= threshold).astype(float)

            val_pck   = correct.sum() / (VAL_SAMPLES * 21)
            val_mpjpe = np.array(val_dists).mean()
            print(f'    >>> VAL PCK@0.2: {val_pck:.4f} | MPJPE: {val_mpjpe:.2f}px')

            if val_pck > best_pck:
                best_pck   = val_pck
                best_epoch = START_EPOCH + epoch + 1
                no_improve = 0
                torch.save(student.state_dict(),
                           os.path.join(BASE, 'checkpoints/distilled_v2_ft_best.pth'))
                print(f'    >>> NEW BEST: PCK@0.2={best_pck:.4f} '
                      f'at Epoch {best_epoch} — saved distilled_v2_ft_best.pth')
            else:
                no_improve += 1
                print(f'    >>> No improvement ({no_improve}/{patience})')
                if no_improve >= patience:
                    print(f'\n    Early stopping: no improvement for '
                          f'{patience * VAL_EVERY} epochs.')
                    break

            student.train()

    # Final save
    torch.save(student.state_dict(),
               os.path.join(BASE, f'checkpoints/distilled_v2_ft_epoch_{START_EPOCH+EPOCHS}.pth'))
    print(f'\n{"="*55}')
    print('FINETUNING COMPLETE')
    print(f'Best PCK@0.2 : {best_pck:.4f} at Epoch {best_epoch}')
    print(f'Best model   : checkpoints/distilled_v2_ft_best.pth')
    print(f'{"="*55}')

if __name__ == '__main__':
    main()
