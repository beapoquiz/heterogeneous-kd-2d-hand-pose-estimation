import numpy as np
import torch
import cv2
import sys

# 1. Import tools EXACTLY as in demo.py
from blazebase import resize_pad, denormalize_detections
from blazepalm import BlazePalm
from blazehand_landmark import BlazeHandLandmark
from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS

# --- CONFIGURATION ---
IMAGE_PATH = "../hand3.jpg"  # Your image file
OUTPUT_FILE = "student_final_result.jpg"

def main():
    # 2. Setup Device
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    print(f"Running on: {gpu}")

    # 3. Load Models (Same logic as demo.py, but just for HANDS)
    print("Loading BlazePalm and BlazeHand...")
    
    palm_detector = BlazePalm().to(gpu)
    palm_detector.load_weights("blazepalm.pth")
    palm_detector.load_anchors("anchors_palm.npy")
    palm_detector.min_score_thresh = 0.5  # Adjusted for better detection

    hand_regressor = BlazeHandLandmark().to(gpu)
    hand_regressor.load_weights("blazehand_landmark.pth")

    # 4. Read Image (Replaces 'capture.read()')
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print(f"Error: Could not find {IMAGE_PATH}")
        return

    print(f"Image loaded: {frame.shape}")

    # 5. Preprocessing (EXACTLY as in demo.py)
    # demo.py flips BGR to RGB here for processing
    # We use .copy() to ensure we don't modify the original 'frame' variable yet
    img_rgb = np.ascontiguousarray(frame[:, :, ::-1])
    
    # This function pads the image to be square, preventing the "wrong crop" issue
    img1, img2, scale, pad = resize_pad(img_rgb)

    # 6. Run Inference (EXACTLY as in demo.py)
    # Step A: Find Palms
    normalized_palm_detections = palm_detector.predict_on_image(img1)
    
    # Map detections back to the original image size
    palm_detections = denormalize_detections(normalized_palm_detections, scale, pad)

    if palm_detections.shape[0] == 0:
        print("No palms detected.")
        return

    print(f"Found {palm_detections.shape[0]} palm(s).")

    # Step B: Find Landmarks for each palm
    xc, yc, scale_roi, theta = palm_detector.detection2roi(palm_detections.cpu())
    
    # 'extract_roi' zooms into the hand properly
    img, affine2, box2 = hand_regressor.extract_roi(img_rgb, xc, yc, theta, scale_roi)
    
    # Predict landmarks
    flags2, handed2, normalized_landmarks2 = hand_regressor(img.to(gpu))
    
    # Map landmarks back to original image
    landmarks2 = hand_regressor.denormalize_landmarks(normalized_landmarks2.cpu(), affine2)

    # 7. Visualization
    # We draw on 'frame' which is still the original BGR image (Full Size)
    for i in range(len(flags2)):
        landmark, flag = landmarks2[i], flags2[i]
        
        # Only draw if confidence is decent
        if flag > 0.5:
            # Draw the skeleton using the repo's helper
            draw_landmarks(frame, landmark[:, :2], HAND_CONNECTIONS, size=2)

    # Draw the red bounding box
    draw_roi(frame, box2)
    # draw_detections(frame, palm_detections) # Optional: draws the raw detection box

    # 8. Display and Save
    # Use WINDOW_NORMAL so you can resize the window if the image is huge
    window_name = "Final Student Result"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, frame)
    
    cv2.imwrite(OUTPUT_FILE, frame)
    print(f"Result saved as {OUTPUT_FILE}")
    
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()