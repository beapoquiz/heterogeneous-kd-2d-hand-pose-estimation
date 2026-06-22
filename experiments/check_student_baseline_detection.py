import torch, sys, os, numpy as np
from mmpose.datasets import Rhd2DDataset
from torch.utils.data import DataLoader
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
init_default_scope('mmpose')

sys.path.append(os.path.join(os.getcwd(), 'student_model'))
from blazehand_landmark import BlazeHandLandmark

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Test TRAINED model detection rate
trained = BlazeHandLandmark().to(device)
trained.load_state_dict(torch.load(
    'checkpoints/distilled_student_epoch_210.pth', map_location=device))
trained.eval()

# Test UNTRAINED model detection rate
untrained = BlazeHandLandmark().to(device)
untrained.eval()

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
loader = DataLoader(ds, batch_size=1, collate_fn=pseudo_collate)

trained_flags   = []
untrained_flags = []
SAMPLES = 500

print(f'Checking hand presence detection rate on {SAMPLES} samples...')
for idx, batch in enumerate(loader):
    if idx >= SAMPLES: break
    img_tensor = torch.stack(batch['inputs']).float() / 255.0

    with torch.no_grad():
        flag_t, _, _ = trained(img_tensor.to(device))
        flag_u, _, _ = untrained(img_tensor.to(device))
        trained_flags.append(flag_t.item())
        untrained_flags.append(flag_u.item())

trained_flags   = np.array(trained_flags)
untrained_flags = np.array(untrained_flags)

# Threshold at 0.5 — above means hand detected
threshold = 0.5
trained_det_rate   = (trained_flags >= threshold).mean()
untrained_det_rate = (untrained_flags >= threshold).mean()

print()
print('='*50)
print('HAND PRESENCE DETECTION RATE')
print(f'  Threshold: flag >= {threshold}')
print()
print(f'  Untrained Student : {untrained_det_rate:.4f} ({untrained_det_rate*100:.2f}%)')
print(f'  Trained Student   : {trained_det_rate:.4f} ({trained_det_rate*100:.2f}%)')
print()
print('Flag score distribution (Trained):')
print(f'  Min  : {trained_flags.min():.4f}')
print(f'  Max  : {trained_flags.max():.4f}')
print(f'  Mean : {trained_flags.mean():.4f}')
print(f'  Std  : {trained_flags.std():.4f}')
print()
print('Flag score distribution (Untrained):')
print(f'  Min  : {untrained_flags.min():.4f}')
print(f'  Max  : {untrained_flags.max():.4f}')
print(f'  Mean : {untrained_flags.mean():.4f}')
print(f'  Std  : {untrained_flags.std():.4f}')
print('='*50)