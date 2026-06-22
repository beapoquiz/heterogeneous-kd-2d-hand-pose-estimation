import torch, sys, os, numpy as np
sys.path.append(os.path.join(os.getcwd(), 'student_model'))
from blazehand_landmark import BlazeHandLandmark
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BlazeHandLandmark().to(device)
model.load_state_dict(torch.load(
    'checkpoints/distilled_student_epoch_20.pth', map_location=device))
model.eval()
from mmpose.datasets import Rhd2DDataset
from torch.utils.data import DataLoader
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
init_default_scope('mmpose')
pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256,256)),
    dict(type='PackPoseInputs')
]
ds = Rhd2DDataset(
    data_root=r'C:\Users\Bea Juliana Poquiz\Desktop\mmpose_thesis\dataset\rhd',
    ann_file='annotations/rhd_test.json', pipeline=pipeline)
loader = DataLoader(ds, batch_size=1, collate_fn=pseudo_collate)
flags = []
for idx, batch in enumerate(loader):
    if idx >= 500: break
    img = torch.stack(batch['inputs']).float() / 255.0
    with torch.no_grad():
        flag, _, _ = model(img.to(device))
        flags.append(flag.item())
flags = np.array(flags)
print(f'Epoch 20 Detection Rate: {(flags>=0.5).mean()*100:.2f}%')
print(f'Flag mean: {flags.mean():.4f}, std: {flags.std():.4f}')