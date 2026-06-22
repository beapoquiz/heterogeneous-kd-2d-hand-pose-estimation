import os
import pickle
import numpy as np
import torch
import cv2
from tqdm import tqdm

# Import Student modules (Ensure these are in your student_model folder)
from blazebase import resize_pad, denormalize_detections
from blazepalm import BlazePalm
from blazehand_landmark import BlazeHandLandmark

# --- CONFIGURATION ---
RHD_DIR = "../dataset/rhd/evaluation/color"
ANNO_PATH = "../dataset/rhd/evaluation/anno_evaluation.pickle"
NUM_SAMPLES = 500  # Start with 500 for a quick check

def calculate_metrics(pred, gt_all):
    """Calculates EPE and PCK for the Student Model."""
    if torch.is_tensor(pred): pred = pred.detach().cpu().numpy()
    if torch.is_tensor(gt_all): gt_all = gt_all.detach().cpu().numpy()

    # Split 42 points into Left (0-20) and Right (21-41)
    gt_l, gt_r = gt_all[0:21, :2], gt_all[21:42, :2]

    # Calculate distances for both hands
    dist_l = np.linalg.norm(pred - gt_l, axis=1)
    dist_r = np.linalg.norm(pred - gt_r, axis=1)
    
    # Select the hand that yields the lower mean error
    final_distances = dist_l if np.mean(dist_l) < np.mean(dist_r) else dist_r
    
    epe = np.mean(final_distances)
    
    # PCK @ 20px (Standard for comparison)
    pck_20 = np.sum(final_distances <= 20) / 21
    
    # For AUC: Calculate PCK at 20 intervals from 0 to 30px
    thresholds = np.linspace(0, 30, 20)
    pck_curve = [np.sum(final_distances <= t) / 21 for t in thresholds]
    
    return epe, pck_20, pck_curve

def main():
    with open(ANNO_PATH, 'rb') as f:
        annos = pickle.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    palm_detector = BlazePalm().to(device)
    palm_detector.load_weights("blazepalm.pth") 
    palm_detector.load_anchors("anchors_palm.npy")
    
    hand_regressor = BlazeHandLandmark().to(device)
    hand_regressor.load_weights("blazehand_landmark.pth")
    
    all_epes = []
    all_pcks = []
    all_curves = []
    detected_count = 0
    
    image_ids = sorted(annos.keys())
    print(f"Evaluating Student on {NUM_SAMPLES} RHD samples...")

    with torch.no_grad():
        for img_id in tqdm(image_ids[:NUM_SAMPLES]):
            img_path = os.path.join(RHD_DIR, f"{img_id:05d}.png")
            frame = cv2.imread(img_path)
            if frame is None: continue
            
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img1, _, scale, pad = resize_pad(img_rgb)
            
            # Detect
            norm_palm = palm_detector.predict_on_image(img1)
            palms = denormalize_detections(norm_palm, scale, pad)
            
            if palms.shape[0] > 0:
                # Regress Landmarks
                xc, yc, scale_roi, theta = palm_detector.detection2roi(palms.cpu())
                img_roi, affine, _ = hand_regressor.extract_roi(img_rgb, xc, yc, theta, scale_roi)
                flags, _, norm_landmarks = hand_regressor(img_roi.to(device))
                
                if flags[0] > 0.5:
                    landmarks = hand_regressor.denormalize_landmarks(norm_landmarks.cpu(), affine)
                    
                    # Calculate Metrics
                    epe, pck, curve = calculate_metrics(landmarks[0][:, :2], annos[img_id]['uv_vis'])
                    all_epes.append(epe)
                    all_pcks.append(pck)
                    all_curves.append(curve)
                    detected_count += 1

    # Final Statistics
    if all_epes:
        avg_epe = np.mean(all_epes)
        avg_pck = np.mean(all_pcks)
        # AUC is the average PCK across the 0-30px range
        avg_auc = np.mean(all_curves) 
        
        print(f"\n--- STUDENT ACCURACY REPORT ---")
        print(f"PCK: {avg_pck:.6f}")
        print(f"AUC: {avg_auc:.6f}")
        print(f"EPE: {avg_epe:.6f}")
        print(f"Detection Rate: {(detected_count/NUM_SAMPLES)*100:.2f}%")
        print(f"--------------------------------------")

if __name__ == "__main__":
    main()