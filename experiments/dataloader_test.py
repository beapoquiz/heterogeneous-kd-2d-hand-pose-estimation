import mmengine
from mmpose.datasets import Rhd2DDataset
from torch.utils.data import DataLoader
from mmengine.dataset import pseudo_collate # The crucial addition

# Set the scope so it finds 'LoadImage', 'TopdownAffine', etc.
try:
    from mmengine.registry import init_default_scope
    init_default_scope('mmpose')
except Exception:
    pass

# 1. Your Verified Paths
DATA_ROOT = r'C:\Users\Bea Juliana Poquiz\Desktop\mmpose_thesis\dataset\rhd'
ANN_FILE = 'annotations/rhd_train.json'

# 2. Your Teacher's Pipeline
pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(type='PackPoseInputs')
]

try:
    print("Initializing Rhd2DDataset...")
    dataset = Rhd2DDataset(
        data_root=DATA_ROOT,
        ann_file=ANN_FILE,
        data_mode='topdown',
        pipeline=pipeline
    )

    # 3. Use pseudo_collate to handle the MMPose data structures
    loader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=pseudo_collate
    )

    # 4. Pull one batch to verify
    batch = next(iter(loader))

    print("\n--- SUCCESS ---")
    print(f"Total Training Images: {len(dataset)}")
    
    # In MMPose 1.x, images are in batch['inputs']
    # It is a list of tensors because of pseudo_collate
    image_shape = batch['inputs'][0].shape 
    print(f"Image Tensor Shape: {image_shape}") # Should be [3, 256, 256]
    
    # Verify we can see the joints we just listed
    sample_joints = batch['data_samples'][0].gt_instances.keypoints
    print(f"Joints detected in sample: {sample_joints.shape}") # Should be (21, 2)

except Exception as e:
    print(f"\n--- FAILED ---")
    import traceback
    traceback.print_exc()