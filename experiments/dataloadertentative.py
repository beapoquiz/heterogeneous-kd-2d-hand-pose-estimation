import torch
from torch.utils.data import DataLoader
from mmpose.datasets import Rhd2DDataset
from mmengine.dataset import pseudo_collate # Required to handle MMPose data dictionaries

# --- 1. DATA PREPARATION ---
# We use the MMPose pipeline because it handles the RHD annotations correctly.
# This replaces 'data_transforms'.
pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(type='PackPoseInputs') # This organizes the data for the model
]

# Initialize the Training Dataset
train_dataset = Rhd2DDataset(
    data_root='C:/Users/Bea Juliana Poquiz/Desktop/mmpose_thesis/dataset/rhd',
    ann_file='annotations/rhd_train.json',
    pipeline=pipeline
)

# --- 2. DATALOADER ---
rhd_dataloader = DataLoader(
    train_dataset, 
    batch_size=32,   
    shuffle=True,    
    num_workers=0,   # Set to 0 on Windows to avoid "BrokenPipe" errors during debugging
    collate_fn=pseudo_collate # CRITICAL: This allows the dataloader to handle MMPose dictionaries
)

# --- 3. HOW TO USE IT IN YOUR LOOP ---
# for batch in rhd_dataloader:
#     # images will be [32, 3, 256, 256]
#     images = torch.stack(batch['inputs']).float() / 255.0 
#     # data_samples contains your keypoints/meta
#     samples = batch['data_samples']