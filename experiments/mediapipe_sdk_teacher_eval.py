import mediapipe as mp
import cv2, numpy as np, torch, sys, os
from mmpose.apis import init_model
from mmpose.codecs import MSRAHeatmap
from mmpose.datasets import Rhd2DDataset
from torch.utils.data import DataLoader
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
init_default_scope('mmpose')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Official MediaPipe Hands SDK — Full model (1.98M params)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.01,
    model_complexity=1
)

# Teacher for reference coordinates
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
loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=pseudo_collate)

SAMPLES = 500
threshold = 0.2 * 256
auc_thresholds = np.linspace(0, 0.5*256, 20)

# MediaPipe joint order (MCP→tip) differs from RHD/teacher order (tip→MCP).
# MP_TO_RHD[i] = teacher index that corresponds to MediaPipe joint i.
MP_TO_RHD = [
    0,           # Wrist
    4, 3, 2, 1,  # Thumb  CMC,MCP,IP,Tip
    8, 7, 6, 5,  # Index  MCP,PIP,DIP,Tip
    12,11,10, 9, # Middle MCP,PIP,DIP,Tip
    16,15,14,13, # Ring   MCP,PIP,DIP,Tip
    20,19,18,17  # Pinky  MCP,PIP,DIP,Tip
]

mp_correct  = np.zeros(21)
mp_dists    = []
mp_detected = 0
mp_failed   = 0

print(f'Evaluating Official MediaPipe SDK on {SAMPLES} RHD test samples...')
print('Model: Full (model_complexity=1, 1.98M params)')
print()

for idx, batch in enumerate(loader):
    if idx >= SAMPLES: break

    img_tensor = torch.stack(batch['inputs']).float() / 255.0

    # Teacher reference coordinates
    with torch.no_grad():
        out = teacher.predict(img_tensor.to(device), batch['data_samples'])
        hms = out[0].pred_fields.heatmaps.cpu().numpy()
        decoded = codec.decode(hms)
        if isinstance(decoded, dict):
            tc = np.array(decoded['keypoints']).reshape(21, 2)
        else:
            tc = np.array(decoded[0]).reshape(21, 2)

    # Get cropped image as numpy RGB
    img_np  = (img_tensor[0].permute(1,2,0).numpy() * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    # Run official MediaPipe SDK
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        mp_detected += 1
        lm = result.multi_hand_landmarks[0].landmark
        mp_coords  = np.array([[l.x * 256, l.y * 256] for l in lm])
        tc_ordered = tc[MP_TO_RHD]   # reorder teacher to match MediaPipe joint order
        dist = np.sqrt(np.sum((mp_coords - tc_ordered)**2, axis=1))
        mp_dists.append(dist)
        mp_correct += (dist <= threshold).astype(float)
    else:
        mp_failed += 1
        mp_dists.append(np.full(21, 256.0))

    if (idx+1) % 100 == 0:
        current_pck = mp_correct.sum() / ((idx+1) * 21)
        det_rate    = mp_detected / (idx+1) * 100
        print(f'[{idx+1}/{SAMPLES}] PCK@0.2: {current_pck:.4f} | '
              f'Det rate: {det_rate:.1f}%')

mp_dists = np.array(mp_dists)
pck      = mp_correct.sum() / (SAMPLES * 21)
mpjpe    = mp_dists.mean()
mse      = np.mean(np.sum(mp_dists**2, axis=1) / 21)
det_rate = mp_detected / SAMPLES * 100
auc_v    = [(mp_dists <= t).mean() for t in auc_thresholds]
auc      = np.trapz(auc_v, auc_thresholds / (0.5*256))

joint_names = ['Wrist',
               'Thumb_CMC','Thumb_MCP','Thumb_IP','Thumb_Tip',
               'Index_MCP','Index_PIP','Index_DIP','Index_Tip',
               'Middle_MCP','Middle_PIP','Middle_DIP','Middle_Tip',
               'Ring_MCP','Ring_PIP','Ring_DIP','Ring_Tip',
               'Pinky_MCP','Pinky_PIP','Pinky_DIP','Pinky_Tip']

print()
print('='*60)
print('OFFICIAL MEDIAPIPE SDK — RHD Test Set (vs Teacher Coords)')
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
for i, name in enumerate(joint_names):
    print(f'{name:<15} {mp_correct[i]/SAMPLES:>10.4f}')
print('='*60)
print()
print('NOTE: PCK measured against HRNetV2 teacher predictions,')
print('not dataset ground truth. Low detection rate expected —')
print('NOTE: MediaPipe was trained on real hands, not synthetic RHD.')

with open('mediapipe_sdk_teacher_results.txt', 'w') as f:
    f.write('Official MediaPipe SDK Evaluation\n')
    f.write(f'Model: Full (model_complexity=1, 1.98M params)\n')
    f.write(f'Dataset: RHD test ({SAMPLES} samples)\n\n')
    f.write(f'PCK@0.2:    {pck:.4f}\n')
    f.write(f'MPJPE:      {mpjpe:.4f} px\n')
    f.write(f'MSE:        {mse:.4f} px2\n')
    f.write(f'AUC:        {auc:.4f}\n')
    f.write(f'Det Rate:   {det_rate:.2f}%\n')
    f.write(f'Detected:   {mp_detected}/{SAMPLES}\n\n')
    for i, name in enumerate(joint_names):
        f.write(f'{name:<15} {mp_correct[i]/SAMPLES:.4f}\n')
print('Saved: mediapipe_sdk_teacher_results.txt')
