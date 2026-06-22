import torch
import numpy as np
import sys
import os

# --- 1. SETUP ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STUDENT_FOLDER = os.path.join(os.getcwd(), 'student_model')
sys.path.append(STUDENT_FOLDER)
from blazehand_landmark import BlazeHandLandmark # type: ignore

# --- 2. LOAD ---
model = BlazeHandLandmark().to(device)
model.load_state_dict(torch.load('checkpoints/distilled_student_epoch_210.pth', map_location=device))
model.eval()

# --- 3. INPUT (Dummy data to check consistency) ---
# We use a random image-sized tensor to see if the model produces "hand-like" patterns
dummy_input = torch.randn(1, 3, 256, 256).to(device)

with torch.no_grad():
    _, _, pred = model(dummy_input)
    coords = pred[0, :, :2].cpu().numpy()

# --- 4. DIAGNOSTIC MATH ---
def calculate_dist(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

# In a real hand, the distance from Wrist (0) to Middle Finger Tip (12) 
# is usually the longest.
wrist_to_middle = calculate_dist(coords[0], coords[12])
thumb_to_pinky = calculate_dist(coords[4], coords[20])

print("\n" + "="*30)
print("HAND STRUCTURE DIAGNOSTIC")
print("="*30)
print(f"Wrist-to-Middle Distance: {wrist_to_middle:.4f}")
print(f"Thumb-to-Pinky Span:    {thumb_to_pinky:.4f}")

if wrist_to_middle > thumb_to_pinky and wrist_to_middle > 0.05:
    print("\nRESULT: SUCCESSFUL STRUCTURE")
    print("The model is outputting a structured hand skeleton.")
    print("The problem is definitely just the POSITION (Inference math).")
else:
    print("\nRESULT: STRUCTURE FAILURE (BEES)")
    print("The coordinates are likely random noise.")
print("="*30)