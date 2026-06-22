import torch
import cv2
import numpy as np
import sys
import os

# --- 1. SETUP ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STUDENT_FOLDER = os.path.join(os.getcwd(), 'student_model')
sys.path.append(STUDENT_FOLDER)
from blazehand_landmark import BlazeHandLandmark

# --- 2. LOAD MODEL ---
model = BlazeHandLandmark().to(device)
model.load_state_dict(torch.load('checkpoints/distilled_student_epoch_210.pth', map_location=device))
model.eval()

# --- 3. LOAD & PREPROCESS IMAGE ---
img_path = r'C:\Users\Bea Juliana Poquiz\Desktop\mmpose_thesis\dataset\rhd\evaluation\color\00000.png'
img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (256, 256))

# Normalize exactly as training did: / 255.0 only
input_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

# --- 4. INFERENCE ---
with torch.no_grad():
    _, _, pred = model(input_tensor)
    # Raw output is 0-1, scale to pixel space exactly as training did
    coords_px = pred[0, :, :2].cpu().numpy() * 256

print("Coords min:", coords_px.min(), "max:", coords_px.max())
print("First 5 keypoints (pixel space):", coords_px[:5])

# --- 5. VISUALIZE ---
vis_img = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

for i in range(21):
    x = int(np.clip(coords_px[i, 0], 0, 255))
    y = int(np.clip(coords_px[i, 1], 0, 255))
    cv2.circle(vis_img, (x, y), 4, (0, 255, 0), -1)
    cv2.putText(vis_img, str(i), (x+4, y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

cv2.imwrite('FIXED_STUDENT_OUTPUT.png', vis_img)
print("Saved FIXED_STUDENT_OUTPUT.png")