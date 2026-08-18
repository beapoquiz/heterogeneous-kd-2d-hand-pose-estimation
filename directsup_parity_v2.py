#!/usr/bin/env python3
"""
directsup_parity_v2.py -- R2-05 parity-controlled Direct Supervision retrain.

Matches distillation_v2.py on every axis except the supervision target
(hard GT keypoints instead of teacher-decoded heatmap coordinates), per
APSIPA 2026 paper #1571328446, Reviewer 2 item R2-05. See
R2-05/R2-05_config_parity.md for the full audit this script implements.

Two defects in the original directsup_baseline.py are fixed here. Both
are prerequisites for a valid parity run, not new design choices:

  1. Coordinate-space bug: directsup_baseline.py trained against
     gt_instances.keypoints (original ~320x320 RHD image space) while
     scoring a 256x256-crop-space prediction against it (mean per-sample
     gap ~70px). Fixed by using PackPoseInputs(pack_transformed=True) and
     reading gt_instances.transformed_keypoints (crop space), matching
     every already-audited "clean" script in this repo
     (directsup_gt_comprehensive_eval.py, scripts/rerun_dual_space_eval.py).
  2. Missing Bone loss term: directsup_baseline.py trained with Wing loss
     only. distillation_v2.py uses Wing + 0.002*Bone. Added here,
     identical formula/weight, computed against the GT target.

Two training protocols, selected via --protocol (default v2match):

  v2match (Phase 2 -- the primary run that answers Reviewer 2):
    Trains on the FULL rhd_train.json (41,255 images, verified against
    distillation_v2.py:104-108 -- Rhd2DDataset(ann_file='rhd_train.json')
    with no Subset/slicing/sampling anywhere in that file, confirmed by
    grep, plus an empirical len()==41255 from this repo's own dataset
    loader). Monitoring reuses V2's exact checkpoint-selection instrument:
    the first 200 samples of rhd_test.json (batch_size=1, shuffle=False),
    scored every 5 epochs -- same file, same sample count, same cadence
    V2 itself uses, just scored against GT (transformed_keypoints) instead
    of teacher-decoded coords, since there is no teacher in this arm. This
    protocol never builds or touches the carved held-out split.

  heldout (Phase 3 -- remedy runs only, e.g. --early-stop):
    Trains on a seeded 90/10 carve of rhd_train.json (37,129 train /
    4,126 held-out val), disjoint from rhd_test.json. This is what
    Reviewer 2's "early stopping on a GT validation split" remedy needs
    to exist before it can be tried -- v2match's 200-sample monitoring is
    borrowed from V2's own test-set-derived instrument, not a real
    held-out set, so it cannot support a defensible early-stop claim.
    --early-stop requires --protocol heldout (enforced below).

Remedies (Phase 3, R2-05) are CLI flags on this same script so each run's
exact config delta is just its logged command line:
  --weight-decay FLOAT   (default 0.0, matches distillation_v2.py's Adam)
  --early-stop            (default off; requires --protocol heldout;
                            stalls when held-out-val PCK@0.2 hasn't
                            improved for --patience val-checks)
  --strong-aug            (default off; widens RandomBBoxTransform ranges)

Everything else (init weights, augmentation order/params absent
--strong-aug, optimizer type, LR schedule, batch size, epoch count,
absence of gradient clipping) is copied unchanged from distillation_v2.py.

Usage:
    python directsup_parity_v2.py --run-name parity_baseline
    python directsup_parity_v2.py --run-name remedy_early_stop --protocol heldout --early-stop
    python directsup_parity_v2.py --run-name remedy_weight_decay --protocol heldout --weight-decay 1e-4
    python directsup_parity_v2.py --run-name remedy_strong_aug --protocol heldout --strong-aug

Timing-probe-only flags (not for real runs):
    --max-iters-per-epoch N   cap steps per epoch
    --max-epochs-probe N      stop after N epochs regardless of --epochs

Writes under R2-05/runs/<run-name>/:
    command.txt, git_commit.txt, seed.txt   -- provenance
    log.csv                                  -- per-epoch train/val metrics
    checkpoints/parity_epoch_{N}.pth         -- every 10 epochs
    checkpoints/parity_best.pth              -- best held-out-val PCK@0.2
    checkpoints/parity_final.pth             -- final epoch (the reported
                                                 checkpoint for non-early-
                                                 stopped runs, matching how
                                                 Dist.V2-100's paper number
                                                 comes from the FINAL
                                                 epoch-100 checkpoint, not
                                                 a "best" checkpoint)
"""

import argparse
import csv
import os
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from mmpose.datasets import Rhd2DDataset
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope

init_default_scope('mmpose')

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'student_model'))
from blazehand_landmark import BlazeHandLandmark  # noqa: E402


def wing_loss(pred, target, w=10.0, epsilon=2.0):
    delta = (pred - target).abs()
    C = w - w * np.log(1 + w / epsilon)
    return torch.where(delta < w,
                        w * torch.log(1 + delta / epsilon),
                        delta - C).mean()


BONES = [(0, 1), (1, 2), (2, 3), (3, 4),
         (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16),
         (0, 17), (17, 18), (18, 19), (19, 20)]


def bone_loss(pred, target):
    losses = [torch.nn.functional.mse_loss(
                  torch.norm(pred[:, i] - pred[:, j], dim=1),
                  torch.norm(target[:, i] - target[:, j], dim=1))
              for i, j in BONES]
    return torch.stack(losses).mean()


def build_pipelines(strong_aug):
    if not strong_aug:
        train_pipeline = [
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale', padding=1.25),
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='RandomBBoxTransform',
                 shift_factor=0.1, shift_prob=0.3,
                 scale_factor=(0.75, 1.25), scale_prob=1.0,
                 rotate_factor=45, rotate_prob=0.6),
            dict(type='TopdownAffine', input_size=(256, 256)),
            dict(type='PhotometricDistortion'),
            dict(type='PackPoseInputs', pack_transformed=True),
        ]
    else:
        # Remedy: widen the geometric augmentation (bbox jitter, scale
        # range, rotation range/prob) -- the standard first lever against
        # overfitting on a texture-limited synthetic dataset like RHD.
        # NOTE: placeholder magnitudes; revisit once Phase 2's parity
        # curve shows whether/how fast overfitting actually appears.
        train_pipeline = [
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale', padding=1.25),
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='RandomBBoxTransform',
                 shift_factor=0.2, shift_prob=0.5,
                 scale_factor=(0.6, 1.4), scale_prob=1.0,
                 rotate_factor=90, rotate_prob=0.8),
            dict(type='TopdownAffine', input_size=(256, 256)),
            dict(type='PhotometricDistortion'),
            dict(type='PackPoseInputs', pack_transformed=True),
        ]
    val_pipeline = [
        dict(type='LoadImage'),
        dict(type='GetBBoxCenterScale'),
        dict(type='TopdownAffine', input_size=(256, 256)),
        dict(type='PackPoseInputs', pack_transformed=True),
    ]
    return train_pipeline, val_pipeline


def evaluate(student, loader, device, threshold, auc_thresholds, max_samples=None):
    student.eval()
    correct = np.zeros(21)
    dists = []
    n = 0
    with torch.no_grad():
        for vidx, vbatch in enumerate(loader):
            if max_samples is not None and n >= max_samples:
                break
            vimg = torch.stack(vbatch['inputs']).float().to(device) / 255.0
            vgt = np.stack([
                np.clip(np.array(ds.gt_instances.transformed_keypoints[0]), 0, 255)
                for ds in vbatch['data_samples']
            ]).astype(np.float32)
            _, _, vpred = student(vimg)
            vsc = vpred[:, :, :2].cpu().numpy() * 256
            d = np.sqrt(np.sum((vsc - vgt) ** 2, axis=2))
            dists.append(d)
            correct += (d <= threshold).sum(axis=0)
            n += d.shape[0]
    dists = np.concatenate(dists, axis=0)
    pck = correct.sum() / (n * 21)
    mpjpe = dists.mean()
    auc_v = [(dists <= t).mean() for t in auc_thresholds]
    auc = float(np.trapz(auc_v, auc_thresholds / (0.5 * 256)))
    student.train()
    return dict(pck=pck, mpjpe=mpjpe, auc=auc, n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--protocol', choices=['v2match', 'heldout'], default='v2match',
                     help='v2match: full 41,255-image rhd_train.json + V2-style '
                          '200-sample rhd_test.json monitoring (Phase 2, primary run). '
                          'heldout: seeded 90/10 carve of rhd_train.json with a real '
                          'held-out val split (Phase 3, remedy runs only).')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--val-fraction', type=float, default=0.10,
                     help='heldout protocol only')
    ap.add_argument('--val-samples', type=int, default=200,
                     help='v2match protocol only: how many leading rhd_test.json '
                          'samples to monitor on, matching distillation_v2.py\'s '
                          'VAL_SAMPLES=200 exactly at its default')
    ap.add_argument('--val-eval-samples', type=int, default=None,
                     help='heldout protocol only: cap held-out-val samples scored '
                          'per check; default: score the full held-out split')
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--early-stop', action='store_true')
    ap.add_argument('--patience', type=int, default=5,
                     help='val-checks without improvement before stopping '
                          '(x VAL_EVERY epochs each), matches '
                          'distillation_finetune.py convention')
    ap.add_argument('--strong-aug', action='store_true')
    ap.add_argument('--max-iters-per-epoch', type=int, default=None,
                     help='timing-probe only: cap steps per epoch')
    ap.add_argument('--max-epochs-probe', type=int, default=None,
                     help='timing-probe only: stop after N epochs '
                          'regardless of --epochs')
    args = ap.parse_args()

    if args.early_stop and args.protocol != 'heldout':
        ap.error('--early-stop requires --protocol heldout: a defensible early-stop '
                  'claim needs a real held-out validation split, not v2match\'s '
                  '200-sample rhd_test.json monitoring subset (that instrument is '
                  'borrowed from distillation_v2.py for parity, not built for '
                  'checkpoint selection against unseen data).')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = os.path.join(ROOT, 'R2-05', 'runs', args.run_name)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(run_dir, 'command.txt'), 'w') as f:
        f.write('python ' + ' '.join(sys.argv) + '\n')
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    except Exception:
        commit = 'unknown'
    with open(os.path.join(run_dir, 'git_commit.txt'), 'w') as f:
        f.write(commit + '\n')
    with open(os.path.join(run_dir, 'seed.txt'), 'w') as f:
        f.write(str(args.seed) + '\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    student = BlazeHandLandmark().to(device)
    student.load_state_dict(torch.load(
        os.path.join(ROOT, 'student_model/blazehand_landmark.pth'),
        map_location=device))
    student.train()
    print('Loaded MediaPipe pretrained weights as starting point (matches distillation_v2.py)')

    DATA_ROOT = os.path.join(ROOT, 'dataset', 'rhd')
    train_pipeline, val_pipeline = build_pipelines(args.strong_aug)

    if args.protocol == 'v2match':
        # Matches distillation_v2.py:104-108 exactly: full rhd_train.json,
        # no Subset/slicing. Monitoring reuses V2's own instrument -- the
        # first VAL_SAMPLES (default 200) of rhd_test.json, batch_size=1,
        # shuffle=False -- scored against GT since there is no teacher here.
        train_dataset = Rhd2DDataset(
            data_root=DATA_ROOT, ann_file='annotations/rhd_train.json',
            pipeline=train_pipeline)
        n_total = len(train_dataset)
        print(f'[v2match] train set: rhd_train.json, {n_total} images (full, '
              f'matches distillation_v2.py:104-108, no subsetting)')
        assert n_total == 41255, \
            f'expected 41,255 images from rhd_train.json (verified against ' \
            f'distillation_v2.py), got {n_total} -- STOP, do not silently proceed'

        val_dataset_full = Rhd2DDataset(
            data_root=DATA_ROOT, ann_file='annotations/rhd_test.json',
            pipeline=val_pipeline)
        val_dataset = Subset(val_dataset_full, list(range(min(args.val_samples, len(val_dataset_full)))))
        print(f'[v2match] monitoring set: first {len(val_dataset)} of rhd_test.json '
              f'({len(val_dataset_full)} total), matching distillation_v2.py\'s '
              f'VAL_SAMPLES={args.val_samples} default and its batch_size=1/'
              f'shuffle=False iteration order. NOTE: this is V2\'s own monitoring '
              f'instrument, reused for parity -- it is not a held-out set (it is '
              f'part of the file everything is finally scored on), so it does not '
              f'license early stopping. See heldout protocol for that.')
        val_batch_size = 1
    else:
        # heldout: seeded 90/10 carve of rhd_train.json, disjoint train/val,
        # rhd_test.json never touched. Required for a defensible early-stop
        # remedy (Phase 3) -- neither original script has this.
        full_train_aug = Rhd2DDataset(
            data_root=DATA_ROOT, ann_file='annotations/rhd_train.json',
            pipeline=train_pipeline)
        full_train_noaug = Rhd2DDataset(
            data_root=DATA_ROOT, ann_file='annotations/rhd_train.json',
            pipeline=val_pipeline)
        assert len(full_train_aug) == len(full_train_noaug), \
            'train/val pipeline variants of rhd_train.json disagree on sample count'
        n_total = len(full_train_aug)

        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(n_total)
        n_val = int(round(args.val_fraction * n_total))
        val_idx = perm[:n_val].tolist()
        train_idx = perm[n_val:].tolist()
        print(f'[heldout] RHD train split: {n_total} images -> {len(train_idx)} train / '
              f'{len(val_idx)} held-out val (seed={args.seed}, fraction={args.val_fraction})')
        print('[heldout] val is carved from rhd_train.json ONLY -- rhd_test.json '
              '(2,727 samples) is never read by this script.')

        train_dataset = Subset(full_train_aug, train_idx)
        val_dataset = Subset(full_train_noaug, val_idx)
        val_batch_size = 16

    # Coordinate-space sanity check (this exact bug has bitten this repo
    # before -- verify rather than trust the pipeline config).
    for k in range(min(3, len(val_dataset))):
        s = val_dataset[k]
        gt_t = np.array(s['data_samples'].gt_instances.transformed_keypoints[0])
        rng_ok = -5 <= gt_t[:, 0].mean() <= 260 and -5 <= gt_t[:, 1].mean() <= 260
        print(f'  sanity[{k}] transformed_keypoints mean={gt_t.mean(axis=0).round(1)} '
              f'in-range={rng_ok}')
        assert rng_ok, 'transformed_keypoints out of expected crop-space range -- STOP'

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                               collate_fn=pseudo_collate, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False,
                             collate_fn=pseudo_collate, num_workers=0)
    val_max_samples = args.val_samples if args.protocol == 'v2match' else args.val_eval_samples

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    VAL_EVERY = 5
    threshold = 0.2 * 256
    auc_thresholds = np.linspace(0, 0.5 * 256, 20)
    best_pck = 0.0
    best_epoch = 0
    no_improve = 0

    log_path = os.path.join(run_dir, 'log.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['epoch', 'train_loss', 'train_wing', 'train_bone', 'lr', 'epoch_seconds',
             'val_pck', 'val_auc', 'val_mpjpe', 'val_n'])

    print(f'\n{"="*55}')
    print('R2-05 PARITY-CONTROLLED DIRECT SUPERVISION RETRAIN')
    print(f'run_name={args.run_name}  weight_decay={args.weight_decay}  '
          f'early_stop={args.early_stop}  strong_aug={args.strong_aug}')
    print('Target: hard GT keypoints (gt_instances.transformed_keypoints, '
          'crop-space, bug-fixed)')
    print('Loss: Wing(w=10) + 0.002*Bone  |  Aug: matches distillation_v2.py'
          f'{" (widened, --strong-aug)" if args.strong_aug else ""}')
    print(f'LR: Cosine 1e-4 -> 1e-6 over {args.epochs} epochs  |  Adam wd={args.weight_decay}')
    print(f'{"="*55}\n')

    epochs_to_run = args.epochs
    if args.max_epochs_probe is not None:
        epochs_to_run = min(epochs_to_run, args.max_epochs_probe)

    t_start = time.time()
    stopped_early_at = None

    for epoch in range(epochs_to_run):
        student.train()
        epoch_loss = epoch_wing = epoch_bone = 0.0
        t_epoch0 = time.time()
        n_steps = 0

        for i, batch in enumerate(train_loader):
            if args.max_iters_per_epoch is not None and i >= args.max_iters_per_epoch:
                break
            images = torch.stack(batch['inputs']).to(device).float() / 255.0

            gt_list = [np.clip(np.array(ds.gt_instances.transformed_keypoints[0]), 0, 255)
                       for ds in batch['data_samples']]
            gt_targets = torch.from_numpy(np.stack(gt_list)).to(device).float()

            optimizer.zero_grad()
            _, _, pred_landmarks = student(images)
            student_coords = pred_landmarks[:, :, :2] * 256
            gt_targets = gt_targets.view(student_coords.shape)

            loss_w = wing_loss(student_coords, gt_targets)
            loss_b = bone_loss(student_coords, gt_targets)
            loss = loss_w + 0.002 * loss_b
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_wing += loss_w.item()
            epoch_bone += loss_b.item()
            n_steps += 1

            if i % 20 == 0:
                print(f'Epoch [{epoch+1}/{epochs_to_run}] Step [{i}/{len(train_loader)}] '
                      f'Loss:{loss.item():.4f} Wing:{loss_w.item():.4f} Bone:{loss_b.item():.4f}')

        scheduler.step()
        epoch_time = time.time() - t_epoch0
        avg_loss = epoch_loss / max(n_steps, 1)
        avg_wing = epoch_wing / max(n_steps, 1)
        avg_bone = epoch_bone / max(n_steps, 1)
        current_lr = scheduler.get_last_lr()[0]
        print(f'===> Epoch {epoch+1} | Loss:{avg_loss:.4f} | Wing:{avg_wing:.4f} | '
              f'Bone:{avg_bone:.4f} | LR:{current_lr:.2e} | {epoch_time:.1f}s/epoch '
              f'({n_steps} steps)')

        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f'    Peak CUDA memory allocated so far: {peak_mb:.1f} MB')

        val_row = dict(pck='', auc='', mpjpe='', n='')
        if (epoch + 1) % 10 == 0:
            ckpt = os.path.join(ckpt_dir, f'parity_epoch_{epoch+1}.pth')
            torch.save(student.state_dict(), ckpt)
            print(f'    Checkpoint saved: {ckpt}')

        if (epoch + 1) % VAL_EVERY == 0:
            t_val0 = time.time()
            m = evaluate(student, val_loader, device, threshold, auc_thresholds,
                         max_samples=val_max_samples)
            val_time = time.time() - t_val0
            tag = 'MONITOR (V2-style, rhd_test.json subset)' if args.protocol == 'v2match' \
                else 'HELD-OUT VAL'
            print(f'    >>> {tag} (n={m["n"]}, {val_time:.1f}s) PCK@0.2: {m["pck"]:.4f} | '
                  f'AUC: {m["auc"]:.4f} | MPJPE: {m["mpjpe"]:.2f}px')
            val_row = dict(pck=f'{m["pck"]:.6f}', auc=f'{m["auc"]:.6f}',
                            mpjpe=f'{m["mpjpe"]:.4f}', n=m['n'])

            if m['pck'] > best_pck:
                best_pck = m['pck']
                best_epoch = epoch + 1
                no_improve = 0
                torch.save(student.state_dict(),
                           os.path.join(ckpt_dir, 'parity_best.pth'))
                print(f'    >>> NEW BEST {tag} PCK@0.2: {best_pck:.4f} at epoch {best_epoch}')
            else:
                no_improve += 1
                print(f'    >>> No improvement ({no_improve}/{args.patience})')
                if args.early_stop and no_improve >= args.patience:
                    print(f'\n    EARLY STOP (--early-stop): no {tag} PCK '
                          f'improvement for {args.patience * VAL_EVERY} epochs.')
                    stopped_early_at = epoch + 1

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch + 1, f'{avg_loss:.6f}', f'{avg_wing:.6f}', f'{avg_bone:.6f}',
                f'{current_lr:.8e}', f'{epoch_time:.2f}', val_row['pck'], val_row['auc'],
                val_row['mpjpe'], val_row['n']])

        if stopped_early_at is not None:
            break

    total_time = time.time() - t_start
    final_epoch = stopped_early_at or epochs_to_run
    torch.save(student.state_dict(),
               os.path.join(ckpt_dir, 'parity_final.pth'))
    print(f'\n{"="*55}')
    print('R2-05 PARITY RUN COMPLETE' if stopped_early_at is None else 'R2-05 PARITY RUN EARLY-STOPPED')
    print(f'run_name={args.run_name}  final_epoch={final_epoch}  '
          f'best_held_out_val_pck={best_pck:.4f} at epoch {best_epoch}')
    print(f'Total wall-clock: {total_time:.1f}s ({total_time/3600:.2f}h)')
    print(f'Final checkpoint: {os.path.join(ckpt_dir, "parity_final.pth")}')
    print(f'{"="*55}')


if __name__ == '__main__':
    main()
