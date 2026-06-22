import mediapipe as mp
import cv2, numpy as np, torch
from mmpose.datasets import Rhd2DDataset
from torch.utils.data import DataLoader
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
init_default_scope('mmpose')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# MediaPipe Full model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.01,
    model_complexity=1
)

# Dataset
pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256,256)),
    dict(type='PackPoseInputs')
]
ds = Rhd2DDataset(
    data_root=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'rhd'),
    ann_file='annotations/rhd_test.json',
    pipeline=pipeline
)
loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=pseudo_collate)

SAMPLES = len(ds)          # evaluate on full RHD test set (2,727 samples)
threshold = 0.2 * 256
auc_thresholds = np.linspace(0, 0.5*256, 20)

# ─────────────────────────────────────────────────────────
# Correct joint remapping — verified from MMPose RHD config
# RHD: tip→MCP order, MediaPipe: MCP→tip order
# For each MediaPipe joint, give the matching RHD index
# ─────────────────────────────────────────────────────────
MP_TO_RHD = [
    0,           # Wrist       → wrist
    4, 3, 2, 1,  # Thumb       CMC,MCP,IP,Tip  → thumb1,2,3,4
    8, 7, 6, 5,  # Index       MCP,PIP,DIP,Tip → forefinger1,2,3,4
    12,11,10, 9, # Middle      MCP,PIP,DIP,Tip → middle1,2,3,4
    16,15,14,13, # Ring        MCP,PIP,DIP,Tip → ring1,2,3,4
    20,19,18,17  # Pinky       MCP,PIP,DIP,Tip → pinky1,2,3,4
]

mp_correct        = np.zeros(21)
mp_dists          = []
mp_detected       = 0
mp_failed         = 0
total_visible     = 0
visible_per_joint = np.zeros(21)

print(f'Evaluating MediaPipe on {SAMPLES} RHD test samples...')
print('Reference: RHD Ground Truth annotations')
print('Model: Full (model_complexity=1, 1.98M params)')
print()

for idx, batch in enumerate(loader):
    if idx >= SAMPLES: break

    img_tensor = torch.stack(batch['inputs']).float() / 255.0

    # Ground truth from rhd_test.json
    gt_instances = batch['data_samples'][0].gt_instances
    gt_keypoints = gt_instances.keypoints[0]          # (21,2)
    gt_visible   = gt_instances.keypoints_visible[0]  # (21,)

    # Reorder GT to match MediaPipe joint order
    tc  = gt_keypoints[MP_TO_RHD]
    vis = gt_visible[MP_TO_RHD]

    # Accumulate visible joint counts regardless of detection outcome
    total_visible     += int((vis > 0).sum())
    visible_per_joint += (vis > 0).astype(float)

    # Cropped image to numpy RGB
    img_np  = (img_tensor[0].permute(1,2,0).numpy() * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        mp_detected += 1
        lm = result.multi_hand_landmarks[0].landmark
        mp_coords = np.array([[l.x * 256, l.y * 256] for l in lm])

        dist = np.sqrt(np.sum((mp_coords - tc)**2, axis=1))
        dist_vis = np.where(vis > 0, dist, np.nan)
        mp_dists.append(dist_vis)
        mp_correct += np.where(vis > 0,
                               (dist <= threshold).astype(float),
                               0.0)
    else:
        mp_failed += 1
        mp_dists.append(np.full(21, 256.0))

    if (idx+1) % 100 == 0:
        current_pck = mp_correct.sum() / max(total_visible, 1)
        det_rate    = mp_detected / (idx+1) * 100
        print(f'[{idx+1}/{SAMPLES}] PCK@0.2: {current_pck:.4f} | '
              f'Det rate: {det_rate:.1f}%')

mp_dists = np.array(mp_dists)
pck      = mp_correct.sum() / total_visible
mpjpe    = np.nanmean(mp_dists)
mse      = np.nanmean(np.nansum(mp_dists**2, axis=1) / 21)
det_rate = mp_detected / SAMPLES * 100
auc_v    = [np.nanmean(mp_dists <= t) for t in auc_thresholds]
auc      = np.trapz(auc_v, auc_thresholds / (0.5*256))

joint_names = ['Wrist',
               'Thumb_CMC','Thumb_MCP','Thumb_IP','Thumb_Tip',
               'Index_MCP','Index_PIP','Index_DIP','Index_Tip',
               'Middle_MCP','Middle_PIP','Middle_DIP','Middle_Tip',
               'Ring_MCP','Ring_PIP','Ring_DIP','Ring_Tip',
               'Pinky_MCP','Pinky_PIP','Pinky_DIP','Pinky_Tip']

print()
print('='*60)
print('MEDIAPIPE SDK — RHD Test Set vs Ground Truth')
print(f'Model: Full (model_complexity=1, 1.98M params)')
print(f'Samples: {SAMPLES}')
print('='*60)
print(f'PCK@0.2      : {pck:.4f} ({pck*100:.2f}%)')
print(f'MPJPE        : {mpjpe:.4f} px')
print(f'MSE          : {mse:.4f} px²')
print(f'AUC          : {auc:.4f}')
print(f'Det. Rate    : {det_rate:.2f}%')
print(f'Detected     : {mp_detected}/{SAMPLES}')
print(f'Not detected : {mp_failed}/{SAMPLES}')
print()
print(f'{"Joint":<15} {"PCK@0.2":>10}')
print('-'*28)
pck_per_joint = mp_correct / np.maximum(visible_per_joint, 1)
for i, name in enumerate(joint_names):
    print(f'{name:<15} {pck_per_joint[i]:>10.4f}')
print('='*60)

with open('mediapipe_gt_full2727.txt', 'w') as f:
    f.write('MediaPipe SDK — RHD Ground Truth Reference\n')
    f.write(f'Model: Full (model_complexity=1, 1.98M params)\n')
    f.write(f'Dataset: RHD test ({SAMPLES} samples)\n\n')
    f.write(f'PCK@0.2:    {pck:.4f}\n')
    f.write(f'MPJPE:      {mpjpe:.4f} px\n')
    f.write(f'MSE:        {mse:.4f} px2\n')
    f.write(f'AUC:        {auc:.4f}\n')
    f.write(f'Det Rate:   {det_rate:.2f}%\n')
    f.write(f'Detected:   {mp_detected}/{SAMPLES}\n\n')
    for i, name in enumerate(joint_names):
        f.write(f'{name:<15} {pck_per_joint[i]:.4f}\n')
print('Saved: mediapipe_gt_full2727.txt')