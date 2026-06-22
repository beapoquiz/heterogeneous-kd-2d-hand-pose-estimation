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
student = BlazeHandLandmark().to(device)
student.load_state_dict(torch.load('checkpoints/distilled_student_epoch_210.pth', map_location=device))
student.eval()

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

# --- USE EVALUATION SPLIT ---
ds = Rhd2DDataset(
    data_root=r'C:\Users\Bea Juliana Poquiz\Desktop\mmpose_thesis\dataset\rhd',
    ann_file='annotations/rhd_test.json',
    pipeline=pipeline
)
loader = DataLoader(ds, batch_size=1, collate_fn=pseudo_collate)

threshold = 0.2 * 256
student_correct = np.zeros(21)
total = 0

print(f'Evaluating on EVALUATION split ({len(ds)} samples)...')

for idx, batch in enumerate(loader):
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

    for k in range(21):
        dist = np.sqrt((tc[k,0]-sc[k,0])**2 + (tc[k,1]-sc[k,1])**2)
        if dist <= threshold:
            student_correct[k] += 1

    total += 1
    if (total) % 500 == 0:
        print(f'  [{total}/{len(ds)}] Running PCK@0.2: {student_correct.sum()/(total*21):.4f}')

pck_per_joint = student_correct / total
overall_pck = pck_per_joint.mean()

print()
print('='*40)
print(f'FINAL EVALUATION SPLIT PCK@0.2: {overall_pck:.4f}')
print(f'Total samples evaluated: {total}')
joint_names = ['Wrist','Thumb1','Thumb2','Thumb3','Thumb4',
               'Index1','Index2','Index3','Index4',
               'Middle1','Middle2','Middle3','Middle4',
               'Ring1','Ring2','Ring3','Ring4',
               'Pinky1','Pinky2','Pinky3','Pinky4']
print()
print('Per-joint PCK@0.2:')
for i, name in enumerate(joint_names):
    print(f'  {name:10s}: {pck_per_joint[i]:.4f}')
print('='*40)