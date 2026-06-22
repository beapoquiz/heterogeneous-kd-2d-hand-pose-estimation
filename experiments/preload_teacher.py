import torch
from mmpose.apis import init_model
from mmpose.utils import register_all_modules

# 1. Register MMPose components
register_all_modules()

# 2. Define the paths to your Teacher files
# These match the files we saw in your 'checkpoints' folder
config_file = 'checkpoints/td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256.py'
checkpoint_file = 'checkpoints/hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth'

# 3. Choose Hardware (GPU if available, otherwise CPU)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(f"--- Thesis Milestone: Week 3 ---")
print(f"Target Model: HRNet-W18_DARK")
print(f"Device detected: {device}")

try:
    # 4. The Preload Action
    model = init_model(config_file, checkpoint_file, device=device)
    print("SUCCESS: HRNet-W18 Teacher is preloaded and ready.")
    
    # 5. Verify the "DARK" specific components are active
    print(f"Model Type: {type(model).__name__}")
    print(f"Input Shape: {model.cfg.model.data_preprocessor.batch_augments[0].size if 'batch_augments' in model.cfg.model.data_preprocessor else 'Standard'}")

except Exception as e:
    print(f"FAILED to preload: {e}")