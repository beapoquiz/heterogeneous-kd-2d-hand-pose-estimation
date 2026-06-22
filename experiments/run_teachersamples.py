import os
import random
import cv2
from mmpose.apis import MMPoseInferencer

# --- CONFIGURATION ---
# Note: Since this is in the root 'mmpose_thesis' folder, we adjust the path
RHD_DIR = "dataset/rhd/evaluation/color" 
NUM_SAMPLES = 5
OUTPUT_FOLDER = "vis_results/teacher_samples"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def main():
    print(f"Checking folder: {os.path.abspath(RHD_DIR)}")
    
    # 1. Check for images (Logic matches your Student script)
    all_images = [f for f in os.listdir(RHD_DIR) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(all_images)} images.")
    
    if len(all_images) == 0:
        print("No images found! Check your RHD path.")
        return

    # 2. Select samples
    test_samples = random.sample(all_images, min(NUM_SAMPLES, len(all_images)))

    # 3. Setup Teacher Model (MMPose Inferencer)
    # This replaces the BlazePalm and BlazeHand Landmark setup
    config_file = 'mmpose/configs/hand_2d_keypoint/topdown_heatmap/rhd2d/td-hm_hrnet-w18_8xb32-210e_rhd2d-256x256.py'
    weights_url = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w18_rhd2d_256x256-02a83e05_20200914.pth'

    print("Initializing HRNet Teacher...")
    inferencer = MMPoseInferencer(
        pose2d=config_file,
        pose2d_weights=weights_url,
        device='cpu' 
    )

    # 4. Loop through samples (Logic matches your Student script)
    for img_name in test_samples:
        img_path = os.path.join(RHD_DIR, img_name)
        print(f"Processing: {img_name}")
        
        # MMPoseInferencer performs Step A (Detection) and Step B (Landmarks) internally
        # It also handles the resizing and padding automatically
        result_generator = inferencer(
            img_path, 
            vis_out_dir=OUTPUT_FOLDER,
            draw_heatmap=True # Good for your thesis to show how the teacher "thinks"
        )
        
        # We must 'exhaust' the generator to trigger the save
        for result in result_generator:
            # result['predictions'] contains the [x, y] coordinates if you need them
            pass

        # Since MMPose saves files automatically, we just load it to show it
        # The file is saved as 'vis_results/teacher_samples/visuals/img_name'
        res_img_path = os.path.join(OUTPUT_FOLDER, img_name)
        res_frame = cv2.imread(res_img_path)
        
        if res_frame is not None:
            cv2.imshow("RHD Teacher Test", res_frame)
            print("Press any key for next image...")
            cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

