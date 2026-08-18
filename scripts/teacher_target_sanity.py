"""
teacher_target_sanity.py — coordinate-space sanity check.

Reproduces, on 20 RHD test samples, the exact keypoint-decoding logic used by:
  (b) the TRAINING pipeline  (distillation_v2.py, teacher_targets computation)
  (c) the EVAL pipeline      (experiments/teacher_gt_comprehensive_eval.py)

and prints, side by side with (a) ground truth:
  - the raw values each path produces (in whatever space it naturally outputs)
  - the same values after applying the model's own inverse-affine
    (input space -> original image space), i.e. the transform mmpose's
    TopdownPoseEstimator.add_pred_to_datasample() applies internally.

Mean L2 is reported for:
  (b) vs (c), raw                    -> tests whether train/eval decode agree
  (b) vs (c), after inverse-affine   -> same, in the space GT lives in
  (b)/(c) raw          vs GT         -> reproduces the ~83px "broken" number
  (b)/(c) inverse-affined vs GT      -> what the comparison SHOULD look like

Run from project root:
    python scripts/teacher_target_sanity.py
"""

import os
import numpy as np
import torch

from mmpose.apis import init_model
from mmpose.codecs import MSRAHeatmap
from mmpose.datasets import Rhd2DDataset
from torch.utils.data import DataLoader
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope

N_SAMPLES = 20

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(
    BASE, 'mmpose/configs/hand_2d_keypoint/topdown_heatmap/'
    'rhd2d/td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256.py')
CHECKPOINT = os.path.join(
    BASE, 'checkpoints', 'hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth')


def decode_manual(sample, codec):
    """Exactly what distillation_v2.py / teacher_gt_comprehensive_eval.py do:
    pull the raw heatmap off the returned PoseDataSample's pred_fields and
    run codec.decode() on it. Returns keypoints in INPUT-CROP (256x256)
    space, per MSRAHeatmap's own docstring ("decoded keypoint coordinates
    are in the input image space").
    """
    hms = None
    if hasattr(sample, 'pred_fields'):
        hms = sample.pred_fields.get(
            'heatmaps', sample.pred_fields.get('pred_heatmaps', None))
    if hms is None:
        raise RuntimeError('No heatmaps found on pred_fields.')
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
    return coords


def input_space_to_orig(coords_256, input_center, input_scale, input_size):
    """The inverse-affine mmpose applies in
    TopdownPoseEstimator.add_pred_to_datasample() (models/pose_estimators/
    topdown.py) to map input-crop-space keypoints back to original image
    space. Neither the training pipeline nor teacher_gt_comprehensive_eval.py
    ever calls this."""
    input_size = np.array(input_size, dtype=np.float64)
    input_scale = np.array(input_scale, dtype=np.float64)
    input_center = np.array(input_center, dtype=np.float64)
    return coords_256 / input_size * input_scale + input_center - 0.5 * input_scale


def main():
    init_default_scope('mmpose')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    print('Loading teacher...')
    teacher = init_model(CONFIG, CHECKPOINT, device=device)
    teacher.cfg.model.test_cfg.output_heatmaps = True
    if hasattr(teacher, 'head') and hasattr(teacher.head, 'test_cfg'):
        if isinstance(teacher.head.test_cfg, dict):
            teacher.head.test_cfg['output_heatmaps'] = True
        else:
            teacher.head.test_cfg.output_heatmaps = True
    teacher.eval()

    codec = MSRAHeatmap(
        input_size=(256, 256), heatmap_size=(64, 64), sigma=2, unbiased=True)

    pipeline = [
        dict(type='LoadImage'),
        dict(type='GetBBoxCenterScale'),
        dict(type='TopdownAffine', input_size=(256, 256)),
        dict(type='PackPoseInputs'),
    ]
    ds = Rhd2DDataset(
        data_root=os.path.join(BASE, 'dataset', 'rhd'),
        ann_file='annotations/rhd_test.json',
        pipeline=pipeline,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=pseudo_collate)

    rows = []  # per-sample dict of everything we print/aggregate

    it = iter(loader)
    for idx in range(N_SAMPLES):
        batch = next(it)
        img = torch.stack(batch['inputs']).float() / 255.0
        sample = batch['data_samples'][0]

        gt = np.array(sample.gt_instances.keypoints[0]).reshape(21, 2)

        input_center = sample.metainfo['input_center']
        input_scale = sample.metainfo['input_scale']
        input_size = sample.metainfo['input_size']

        with torch.no_grad():
            # teacher.predict() returns a list of PoseDataSample (one per
            # batch item), each carrying .pred_instances (already
            # inverse-affine'd to original space by add_pred_to_datasample)
            # AND .pred_fields.heatmaps (raw 64x64, since output_heatmaps
            # was requested). SAME forward pass feeds both the "training"
            # and "eval" decode paths below -- they must agree exactly
            # since they consume the same heatmaps. Two separate
            # teacher.predict() calls are made only to mirror how the two
            # source scripts actually invoke it.
            out_train = teacher.predict(img.to(device), batch['data_samples'])
            out_eval = teacher.predict(img.to(device), batch['data_samples'])

        # (b) TRAINING-pipeline decode (distillation_v2.py logic)
        b_256 = decode_manual(out_train[0], codec)
        b_256 = np.clip(b_256, 0, 255)  # distillation_v2.py clips to [0,255]

        # (c) EVAL-pipeline decode (teacher_gt_comprehensive_eval.py logic)
        c_256 = decode_manual(out_eval[0], codec)

        # bonus: the CORRECT, already-inverse-affine'd output mmpose itself
        # produces via TopdownPoseEstimator.add_pred_to_datasample() --
        # neither training nor eval script ever reads this field.
        official_orig = np.array(out_eval[0].pred_instances.keypoints[0]).reshape(21, 2)

        b_orig = input_space_to_orig(b_256, input_center, input_scale, input_size)
        c_orig = input_space_to_orig(c_256, input_center, input_scale, input_size)

        rows.append(dict(
            gt=gt, b_256=b_256, c_256=c_256, b_orig=b_orig, c_orig=c_orig,
            official_orig=official_orig,
        ))

    print(f'{"="*78}')
    print(f'  TEACHER TARGET SANITY CHECK — {N_SAMPLES} RHD test samples')
    print(f'{"="*78}\n')

    for i, r in enumerate(rows):
        d_bc_256 = np.linalg.norm(r['b_256'] - r['c_256'], axis=1).mean()
        d_bc_orig = np.linalg.norm(r['b_orig'] - r['c_orig'], axis=1).mean()
        d_b_gt_raw = np.linalg.norm(r['b_256'] - r['gt'], axis=1).mean()
        d_c_gt_raw = np.linalg.norm(r['c_256'] - r['gt'], axis=1).mean()
        d_b_gt_fixed = np.linalg.norm(r['b_orig'] - r['gt'], axis=1).mean()
        d_c_gt_fixed = np.linalg.norm(r['c_orig'] - r['gt'], axis=1).mean()
        d_official_gt = np.linalg.norm(r['official_orig'] - r['gt'], axis=1).mean()
        d_b_vs_official = np.linalg.norm(r['b_orig'] - r['official_orig'], axis=1).mean()

        print(f'[Sample {i:2d}]')
        print(f'  GT (orig space)         wrist={r["gt"][0]}')
        print(f'  (b) train-decode, 256sp wrist={r["b_256"][0]}')
        print(f'  (c) eval-decode,  256sp wrist={r["c_256"][0]}')
        print(f'  (b) train-decode, orig  wrist={r["b_orig"][0]}')
        print(f'  (c) eval-decode,  orig  wrist={r["c_orig"][0]}')
        print(f'  mean L2  (b)vs(c) raw (256sp)      : {d_bc_256:8.4f} px'
              f'   <- should be ~0 (same heatmap, same codec)')
        print(f'  mean L2  (b)vs(c) after inv-affine  : {d_bc_orig:8.4f} px'
              f'   <- should be ~0')
        print(f'  mean L2  (b) raw        vs GT       : {d_b_gt_raw:8.4f} px'
              f'   <- reproduces the ~83px "broken" number')
        print(f'  mean L2  (c) raw        vs GT       : {d_c_gt_raw:8.4f} px'
              f'   <- reproduces the ~83px "broken" number')
        print(f'  mean L2  (b) inv-affine vs GT       : {d_b_gt_fixed:8.4f} px'
              f'   <- true prediction error')
        print(f'  mean L2  (c) inv-affine vs GT       : {d_c_gt_fixed:8.4f} px'
              f'   <- true prediction error')
        print(f'  mean L2  official pred_instances vs GT : {d_official_gt:8.4f} px'
              f'   <- mmpose\'s own correct output')
        print(f'  mean L2  (b) inv-affine vs official : {d_b_vs_official:8.4f} px'
              f'   <- confirms our manual inverse-affine matches mmpose\'s')
        print()

    all_bc_256 = np.mean([np.linalg.norm(r['b_256'] - r['c_256'], axis=1).mean()
                          for r in rows])
    all_bc_orig = np.mean([np.linalg.norm(r['b_orig'] - r['c_orig'], axis=1).mean()
                           for r in rows])
    all_b_gt_raw = np.mean([np.linalg.norm(r['b_256'] - r['gt'], axis=1).mean()
                            for r in rows])
    all_c_gt_raw = np.mean([np.linalg.norm(r['c_256'] - r['gt'], axis=1).mean()
                            for r in rows])
    all_b_gt_fixed = np.mean([np.linalg.norm(r['b_orig'] - r['gt'], axis=1).mean()
                              for r in rows])
    all_c_gt_fixed = np.mean([np.linalg.norm(r['c_orig'] - r['gt'], axis=1).mean()
                              for r in rows])
    all_official_gt = np.mean([np.linalg.norm(r['official_orig'] - r['gt'], axis=1).mean()
                               for r in rows])
    all_b_vs_official = np.mean([np.linalg.norm(r['b_orig'] - r['official_orig'], axis=1).mean()
                                 for r in rows])

    print(f'{"="*78}')
    print(f'  AGGREGATE OVER {N_SAMPLES} SAMPLES')
    print(f'{"="*78}')
    print(f'  (b) vs (c), raw 256-space          : {all_bc_256:8.4f} px  (train/eval decode agreement)')
    print(f'  (b) vs (c), after inverse-affine   : {all_bc_orig:8.4f} px  (train/eval decode agreement)')
    print(f'  (b) raw   vs GT                    : {all_b_gt_raw:8.4f} px  (space-mismatched "error")')
    print(f'  (c) raw   vs GT                    : {all_c_gt_raw:8.4f} px  (space-mismatched "error")')
    print(f'  (b) inv-affine vs GT               : {all_b_gt_fixed:8.4f} px  (true teacher error)')
    print(f'  (c) inv-affine vs GT               : {all_c_gt_fixed:8.4f} px  (true teacher error)')
    print(f'  official pred_instances vs GT      : {all_official_gt:8.4f} px  (mmpose\'s own correct output)')
    print(f'  (b) inv-affine vs official         : {all_b_vs_official:8.4f} px  (manual inv-affine == mmpose\'s)')
    print()
    print('  Interpretation:')
    print('    - If (b) vs (c) is ~0 in BOTH rows: the training pipeline and')
    print('      the eval script decode identically -> the model/codec are fine,')
    print('      and the eval script is the thing that is broken (it never')
    print('      applies the inverse-affine that TopdownPoseEstimator.predict()')
    print('      applies automatically via add_pred_to_datasample()).')
    print('    - If "raw vs GT" is ~80px but "inv-affine vs GT" is ~2-3px: this')
    print('      CONFIRMS the bug is a missing coordinate-space conversion in')
    print('      the eval script, not a broken training target.')
    print(f'{"="*78}')


if __name__ == '__main__':
    main()
