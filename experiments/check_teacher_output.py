import torch
from mmpose.apis import init_model
from mmpose.datasets import Rhd2DDataset
from torch.utils.data import DataLoader
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope

# 1. Initialize Registry
init_default_scope('mmpose')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. LOAD TEACHER
CONFIG = 'mmpose/configs/hand_2d_keypoint/topdown_heatmap/rhd2d/td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256.py'
CHECKPOINT = 'checkpoints/hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth'

print(f"Loading Teacher Model from: {CHECKPOINT}...")
teacher = init_model(CONFIG, CHECKPOINT, device=device)

# --- THE FORCE-ALL CONFIGURATION ---
# We force the output_heatmaps flag in all possible locations 
# to ensure MMPose v1.x doesn't discard them after decoding.
teacher.cfg.test_cfg.output_heatmaps = True

if 'test_cfg' not in teacher.cfg.model:
    teacher.cfg.model.test_cfg = dict()
teacher.cfg.model.test_cfg.output_heatmaps = True

if hasattr(teacher, 'head'):
    teacher.head.test_cfg.output_heatmaps = True

teacher.eval()

# 3. Setup Dataset
DATA_ROOT = r'C:\Users\Bea Juliana Poquiz\Desktop\mmpose_thesis\dataset\rhd'
ANN_FILE = 'annotations/rhd_train.json'

pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(type='PackPoseInputs')
]

try:
    print("Initializing Dataset and loading one batch...")
    dataset = Rhd2DDataset(data_root=DATA_ROOT, ann_file=ANN_FILE, pipeline=pipeline)
    loader = DataLoader(dataset, batch_size=2, collate_fn=pseudo_collate)
    batch = next(iter(loader))

    # 4. THE TEST
    print("Passing batch through Teacher...")
    with torch.no_grad():
        # In MMPose v1, test_step is the standard way to run inference 
        # that respects the cfg flags we just set.
        output = teacher.test_step(batch) 

    print("\n" + "="*40)
    print(f"TEACHER OUTPUT DETECTED")
    print(f"Type: {type(output)}")
    
    if isinstance(output, list):
        print(f"List Length: {len(output)}")
        print(f"First Item Type: {type(output[0])}")
        
        # Checking the DataSample for the hidden heatmap
        if hasattr(output[0], 'pred_fields') and 'heatmaps' in output[0].pred_fields:
            hms = output[0].pred_fields.heatmaps
            print(f"HEATMAPS FOUND! Shape: {hms.shape}")
            print("\nRESULT: SUCCESS.")
            print("The teacher is now outputting heatmaps correctly.")
            print("You can now safely use this logic in distillation.py.")
        else:
            print("\nRESULT: Still no heatmaps. Checking for alternative keys...")
            print(f"Available keys in pred_fields: {output[0].pred_fields.keys()}")
    print("="*40 + "\n")

except Exception as e:
    print(f"\n--- FAILED ---")
    import traceback
    traceback.print_exc()