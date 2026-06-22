import cv2
import torch
import numpy as np
import sys, os, time
import mediapipe as mp

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE, 'student_model'))
from blazehand_landmark import BlazeHandLandmark

CHECKPOINT = os.path.join(BASE, 'checkpoints', 'distilled_v2_epoch_100.pth')

BONES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]

# Finger tip indices for colour coding
TIPS = {4: (0,200,255), 8: (0,255,100), 12: (255,100,0),
        16: (200,0,255), 20: (0,100,255)}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

print('Loading distilled v2 student...')
model = BlazeHandLandmark().to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()
print('Model ready.')

# MediaPipe used only for bounding-box detection
mp_hands = mp.solutions.hands
detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def hand_bbox(landmarks, fw, fh, pad=0.25):
    xs = [lm.x * fw for lm in landmarks]
    ys = [lm.y * fh for lm in landmarks]
    bw = max(xs) - min(xs)
    bh = max(ys) - min(ys)
    px, py = bw * pad, bh * pad
    x1 = max(0,  int(min(xs) - px))
    y1 = max(0,  int(min(ys) - py))
    x2 = min(fw, int(max(xs) + px))
    y2 = min(fh, int(max(ys) + py))
    return x1, y1, x2, y2


def square_crop(x1, y1, x2, y2, fw, fh):
    size = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(0,  cx - size // 2)
    y1 = max(0,  cy - size // 2)
    x2 = min(fw, x1 + size)
    y2 = min(fh, y1 + size)
    return x1, y1, x2, y2


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('ERROR: Cannot open webcam.')
    sys.exit(1)

print("Running — press 'q' to quit, 's' to save a screenshot.")

prev = time.time()
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)          # mirror so it feels natural
    fh, fw = frame.shape[:2]

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result  = detector.process(rgb)

    if result.multi_hand_landmarks:
        for hand_lm in result.multi_hand_landmarks:

            # --- bounding box from MediaPipe landmarks ---
            x1, y1, x2, y2 = hand_bbox(hand_lm.landmark, fw, fh)
            x1, y1, x2, y2 = square_crop(x1, y1, x2, y2, fw, fh)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            cw, ch = x2 - x1, y2 - y1

            # --- preprocess for student ---
            inp_rgb = cv2.cvtColor(cv2.resize(crop, (256, 256)),
                                   cv2.COLOR_BGR2RGB)
            inp = (torch.from_numpy(inp_rgb.transpose(2, 0, 1))
                   .float().unsqueeze(0) / 255.0).to(device)

            # --- student inference ---
            with torch.no_grad():
                hand_flag, _, landmarks = model(inp)
                kps = landmarks[0, :, :2].cpu().numpy() * 256  # (21,2) in 256x256

            # --- map back to full frame ---
            kps[:, 0] = kps[:, 0] * (cw / 256.0) + x1
            kps[:, 1] = kps[:, 1] * (ch / 256.0) + y1
            kps = kps.astype(int)

            # --- draw skeleton ---
            for i, j in BONES:
                cv2.line(frame,
                         tuple(kps[i]), tuple(kps[j]),
                         (220, 220, 220), 2, cv2.LINE_AA)

            # --- draw joints ---
            for idx in range(21):
                colour = TIPS.get(idx, (0, 255, 0))
                cv2.circle(frame, tuple(kps[idx]), 5, colour, -1, cv2.LINE_AA)
                cv2.circle(frame, tuple(kps[idx]), 5, (0, 0, 0),  1, cv2.LINE_AA)

            # --- bounding box + confidence ---
            score = hand_flag[0].item() if hand_flag.numel() else 0.0
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 1)
            cv2.putText(frame, f'conf {score:.2f}',
                        (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

    # --- FPS overlay ---
    now  = time.time()
    fps  = 1.0 / max(now - prev, 1e-6)
    prev = now
    cv2.putText(frame, f'FPS {fps:.1f}',
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, 'Distilled v2 Student  |  21 hand keypoints',
                (10, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.imshow('Hand Keypoint Demo — Distilled v2', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        saved += 1
        os.makedirs(os.path.join(BASE, 'thesis_figures'), exist_ok=True)
        path = os.path.join(BASE, 'thesis_figures', f'webcam_demo_{saved}.png')
        cv2.imwrite(path, frame)
        print(f'Screenshot saved: {path}')

cap.release()
cv2.destroyAllWindows()
detector.close()
