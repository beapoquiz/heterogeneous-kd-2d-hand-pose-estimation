import os
import random
import numpy as np
import torch
import cv2

# 1. Import tools from the student folder
from blazebase import resize_pad, denormalize_detections
from blazepalm import BlazePalm
from blazehand_landmark import BlazeHandLandmark
from visualization import draw_landmarks, draw_roi, HAND_CONNECTIONS

# --- CONFIGURATION ---
RHD_DIR = "../dataset/rhd/evaluation/color" 
NUM_SAMPLES = 5
OUTPUT_FOLDER = "test_results"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def main():
    print(f"Checking folder: {os.path.abspath(RHD_DIR)}")
    
    # Check for images
    all_images = [f for f in os.listdir(RHD_DIR) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(all_images)} images.")
    
    if len(all_images) == 0:
        print("Still no images found. Verify the path!")
        return

    # Select samples
    test_samples = random.sample(all_images, min(NUM_SAMPLES, len(all_images)))

    # --- SETUP MODELS ---
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    
    palm_detector = BlazePalm().to(gpu)
    palm_detector.load_weights("blazepalm.pth")
    palm_detector.load_anchors("anchors_palm.npy")
    palm_detector.min_score_thresh = 0.3  # Lowered for synthetic RHD images

    hand_regressor = BlazeHandLandmark().to(gpu)
    hand_regressor.load_weights("blazehand_landmark.pth")

    # --- LOOP THROUGH RANDOM SAMPLES ---
    for img_name in test_samples:
        img_path = os.path.join(RHD_DIR, img_name)
        print(f"Processing: {img_name}")
        
        frame = cv2.imread(img_path)
        if frame is None: continue

        img_rgb = np.ascontiguousarray(frame[:, :, ::-1])
        img1, img2, scale, pad = resize_pad(img_rgb)

        # Step A: Find Palms
        normalized_palm_detections = palm_detector.predict_on_image(img1)
        palm_detections = denormalize_detections(normalized_palm_detections, scale, pad)

        if palm_detections.shape[0] > 0:
            # Step B: Find Landmarks
            xc, yc, scale_roi, theta = palm_detector.detection2roi(palm_detections.cpu())
            img, affine2, box2 = hand_regressor.extract_roi(img_rgb, xc, yc, theta, scale_roi)
            flags2, handed2, normalized_landmarks2 = hand_regressor(img.to(gpu))
            landmarks2 = hand_regressor.denormalize_landmarks(normalized_landmarks2.cpu(), affine2)

            for i in range(len(flags2)):
                if flags2[i] > 0.5:
                    draw_landmarks(frame, landmarks2[i][:, :2], HAND_CONNECTIONS, size=2)
            
            draw_roi(frame, box2)
        else:
            print(f"No hand detected in {img_name}")

        # Save and Show
        save_path = os.path.join(OUTPUT_FOLDER, f"res_{img_name}")
        cv2.imwrite(save_path, frame)
        cv2.imshow("RHD Random Test", frame)
        print("Press any key for next image...")
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()