"""
Fine-tune BlazeHandLandmark on the FreiHAND dataset.

Starting point (in priority order):
  1. checkpoints/distilled_v2_ft_best.pth   (distilled + RHD-finetuned)
  2. checkpoints/direct_sup_best.pth         (direct supervision on RHD)
  3. student_model/blazehand_landmark.pth    (MediaPipe pretrained)

FreiHAND is a real-world dataset with 130,240 RGB images of hands
at 224×224.  We project 3-D joint annotations to 2-D using the
provided camera intrinsics and fine-tune with Wing Loss.

Usage:
    python finetune_freihand.py
    python finetune_freihand.py --epochs 30 --lr 3e-5 --batch 32
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn

STUDENT_FOLDER = os.path.join(os.getcwd(), 'student_model')
if STUDENT_FOLDER not in sys.path:
    sys.path.insert(0, STUDENT_FOLDER)

from freihand_dataset import make_train_val_loaders


# ─── loss ────────────────────────────────────────────────────────────────────

def wing_loss(pred, target, w=10.0, epsilon=2.0):
    delta = (pred - target).abs()
    C = w - w * np.log(1 + w / epsilon)
    loss = torch.where(delta < w,
                       w * torch.log(1 + delta / epsilon),
                       delta - C)
    return loss.mean()


# ─── checkpoint loading ───────────────────────────────────────────────────────

CHECKPOINT_CANDIDATES = [
    'checkpoints/distilled_v2_ft_epoch_150.pth',
    'checkpoints/distilled_v2_ft_best.pth',
    'checkpoints/direct_sup_best.pth',
    'student_model/blazehand_landmark.pth',
]

def load_best_available(model, device):
    for ckpt in CHECKPOINT_CANDIDATES:
        if os.path.isfile(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device))
            print(f'Loaded weights from: {ckpt}')
            return ckpt
    raise FileNotFoundError(
        f"No starting checkpoint found. Checked:\n" +
        "\n".join(f"  {c}" for c in CHECKPOINT_CANDIDATES)
    )


# ─── validation ───────────────────────────────────────────────────────────────

def validate(student, val_loader, device, threshold_px):
    student.eval()
    all_dists = []
    correct   = np.zeros(21)
    n_samples = 0

    with torch.no_grad():
        for img, kps_gt in val_loader:
            img    = img.to(device)             # (1, 3, 256, 256)
            kps_gt = kps_gt.squeeze(0).numpy()  # (21, 2)

            _, _, pred_lm = student(img)
            pred_kps = pred_lm[0, :, :2].cpu().numpy() * 256  # (21, 2)

            dists = np.sqrt(np.sum((pred_kps - kps_gt) ** 2, axis=1))  # (21,)
            all_dists.append(dists)
            correct  += (dists <= threshold_px).astype(float)
            n_samples += 1

    all_dists = np.array(all_dists)           # (N, 21)
    pck   = correct.sum() / (n_samples * 21)
    mpjpe = all_dists.mean()
    return pck, mpjpe


# ─── main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--freihand_root', default=
                   r'freihand',
                   help='Path to FreiHAND folder containing training_K.json and training/')
    p.add_argument('--epochs',    type=int,   default=50)
    p.add_argument('--lr',        type=float, default=5e-5,
                   help='Initial learning rate (lower than training from scratch)')
    p.add_argument('--batch',     type=int,   default=32)
    p.add_argument('--val_every', type=int,   default=5)
    p.add_argument('--save_every',type=int,   default=10)
    p.add_argument('--resume',    type=str,   default=None,
                   help='Path to checkpoint to resume from (overrides auto-select)')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── model ────────────────────────────────────────────────────────────────
    from blazehand_landmark import BlazeHandLandmark
    student = BlazeHandLandmark().to(device)

    if args.resume:
        student.load_state_dict(torch.load(args.resume, map_location=device))
        start_ckpt = args.resume
        print(f'Resuming from: {start_ckpt}')
    else:
        start_ckpt = load_best_available(student, device)

    student.train()

    # ── data ─────────────────────────────────────────────────────────────────
    freihand_root = args.freihand_root
    if not os.path.isabs(freihand_root):
        freihand_root = os.path.join(os.getcwd(), freihand_root)

    train_loader, val_loader = make_train_val_loaders(
        freihand_root,
        input_size=256,
        val_frac=0.05,
        batch_size=args.batch,
    )

    # ── optimiser & schedule ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7)

    threshold_px = 0.2 * 256   # PCK@0.2
    best_pck     = 0.0
    best_epoch   = 0
    os.makedirs('checkpoints', exist_ok=True)

    print(f'\n{"="*60}')
    print('FREIHAND FINE-TUNING')
    print(f'Starting checkpoint : {start_ckpt}  (RHD Epoch 150)')
    print(f'Epochs              : {args.epochs}')
    print(f'Learning rate       : {args.lr}')
    print(f'Batch size          : {args.batch}')
    print(f'Train batches/epoch : {len(train_loader):,}')
    print(f'Val samples         : {len(val_loader.dataset):,}')
    print(f'{"="*60}\n')

    for epoch in range(args.epochs):
        student.train()
        epoch_loss = 0.0

        for i, (imgs, kps_gt) in enumerate(train_loader):
            imgs   = imgs.to(device)                     # (B, 3, 256, 256)
            kps_gt = kps_gt.to(device)                   # (B, 21, 2) in [0,256)

            optimizer.zero_grad()
            _, _, pred_lm = student(imgs)                # (B, 21, 3) in [0,1]
            pred_kps = pred_lm[:, :, :2] * 256          # (B, 21, 2) in [0,256)

            loss = wing_loss(pred_kps, kps_gt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if i % 50 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}] '
                      f'Step [{i}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f}')

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f'===> Epoch {epoch+1} | Avg Loss: {avg_loss:.6f} | '
              f'LR: {scheduler.get_last_lr()[0]:.2e}')

        # periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = f'checkpoints/freihand_ft_epoch_{epoch+1}.pth'
            torch.save(student.state_dict(), ckpt_path)
            print(f'    Saved: {ckpt_path}')

        # validation
        if (epoch + 1) % args.val_every == 0:
            pck, mpjpe = validate(student, val_loader, device, threshold_px)
            print(f'    >>> VAL PCK@0.2: {pck:.4f} | MPJPE: {mpjpe:.2f}px')

            if pck > best_pck:
                best_pck   = pck
                best_epoch = epoch + 1
                torch.save(student.state_dict(), 'checkpoints/freihand_ft_best.pth')
                print(f'    >>> NEW BEST: {best_pck:.4f} at Epoch {best_epoch}')

            student.train()

    # final save
    torch.save(student.state_dict(),
               f'checkpoints/freihand_ft_epoch_{args.epochs}.pth')

    print(f'\n{"="*60}')
    print('FREIHAND FINE-TUNING COMPLETE')
    print(f'Best PCK@0.2 : {best_pck:.4f}  (Epoch {best_epoch})')
    print(f'Best model   : checkpoints/freihand_ft_best.pth')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
