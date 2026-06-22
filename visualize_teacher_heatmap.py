import os
import glob
from mmpose.apis import MMPoseInferencer
from mmpose.utils import register_all_modules

# 1. Initialize MMPose
register_all_modules()

# 2. Get the Absolute Path to your project
base_path = os.path.abspath(os.getcwd())

# 3. Define Config and Weights (These were already 'True')
cfg = os.path.join(base_path, 'checkpoints', 'td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256.py')
ckpt = os.path.join(base_path, 'checkpoints', 'hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth')

# 4. SEARCH FOR THE IMAGE
# This looks recursively for any .png inside the 'dataset' folder
search_pattern = os.path.join(base_path, 'dataset', '**', '*.png')
found_images = glob.glob(search_pattern, recursive=True)

if found_images:
    # Filter for training images specifically if possible
    training_images = [f for f in found_images if 'training' in f.lower()]
    img = training_images[0] if training_images else found_images[0]
    img_exists = True
else:
    img = "NONE"
    img_exists = False

# --- DIAGNOSTIC PRINT ---
print("\n--- [Final Path Verification] ---")
print(f"Config:  {'✅ FOUND' if os.path.exists(cfg) else '❌ MISSING'}")
print(f"Weights: {'✅ FOUND' if os.path.exists(ckpt) else '❌ MISSING'}")
print(f"Image:   {'✅ FOUND' if img_exists else '❌ MISSING'}")
if img_exists:
    print(f"Found Image at: {img}")
print("---------------------------------\n")

if not (os.path.exists(cfg) and os.path.exists(ckpt) and img_exists):
    print("❌ Critical Error: Still can't find the image.")
    print("Please manually check if there are .png files in dataset/rhd/training/color/")
    exit()

# 5. RUN THE TEACHER
print("🚀 Teacher is initializing (HRNet-W18)...")
try:
    inferencer = MMPoseInferencer(pose2d=cfg, pose2d_weights=ckpt, device='cpu')

    print("Detecting keypoints...")
    # result_generator is a generator; we need to iterate to trigger the action
    result_generator = inferencer(
        img, 
        show=True, 
        out_dir='vis_results', 
        draw_heatmap=True
    )

    for _ in result_generator:
        pass

    print("\n🎉 SUCCESS! The window should be open.")
    print("Check the 'vis_results' folder for the analysis.")

except Exception as e:
    print(f"❌ Inference failed: {e}")