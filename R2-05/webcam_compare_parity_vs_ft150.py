#!/usr/bin/env python3
"""
R2-05/webcam_compare_parity_vs_ft150.py

Headless, auto-capturing side-by-side comparison: R2-05 Parity Direct
Supervision (v2match, epoch 100) vs. FT V2-150 (distilled), both run on
the SAME captured frame at the same instant (one MediaPipe bbox detection
shared by both models per frame, so neither gets an easier crop) -- this
is a stricter fairness guarantee than running the two existing interactive
demos back-to-back, since hand position/pose necessarily drifts between
two separate live sessions.

No cv2.imshow / cv2.waitKey -- this cannot be driven interactively from
this environment (no live display, no way to time a keypress against a
live feed). Instead it runs for a fixed wall-clock duration and
auto-saves a side-by-side frame at a fixed interval. A real hand must be
in front of the webcam during that window for the output to mean anything
-- point the camera at your hand and run this when ready.

Usage:
    python R2-05/webcam_compare_parity_vs_ft150.py [--seconds 20] [--interval 0.5]

Writes:
    R2-05/webcam_frames/compare_NNN.png  (left: parity direct sup, right: FT V2-150)
"""

import argparse
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE, 'student_model'))
from blazehand_landmark import BlazeHandLandmark  # noqa: E402

CKPT_A = os.path.join(BASE, 'R2-05', 'runs', 'parity_baseline', 'checkpoints', 'parity_final.pth')
LABEL_A = 'Direct Sup. (R2-05 parity, ep.100)'
CKPT_B = os.path.join(BASE, 'checkpoints', 'distilled_v2_ft_epoch_150.pth')
LABEL_B = 'FT V2-150 (distilled)'

BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
TIPS = {4: (0, 200, 255), 8: (0, 255, 100), 12: (255, 100, 0),
        16: (200, 0, 255), 20: (0, 100, 255)}


def hand_bbox(landmarks, fw, fh, pad=0.25):
    xs = [lm.x * fw for lm in landmarks]
    ys = [lm.y * fh for lm in landmarks]
    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
    px, py = bw * pad, bh * pad
    x1 = max(0, int(min(xs) - px))
    y1 = max(0, int(min(ys) - py))
    x2 = min(fw, int(max(xs) + px))
    y2 = min(fh, int(max(ys) + py))
    return x1, y1, x2, y2


def square_crop(x1, y1, x2, y2, fw, fh):
    size = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(fw, x1 + size)
    y2 = min(fh, y1 + size)
    return x1, y1, x2, y2


def draw_overlay(frame, kps, score, x1, y1, x2, y2, label):
    for i, j in BONES:
        cv2.line(frame, tuple(kps[i]), tuple(kps[j]), (220, 220, 220), 2, cv2.LINE_AA)
    for idx in range(21):
        colour = TIPS.get(idx, (0, 255, 0))
        cv2.circle(frame, tuple(kps[idx]), 5, colour, -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(kps[idx]), 5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 1)
    cv2.putText(frame, f'conf {score:.2f}', (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
    fh = frame.shape[0]
    cv2.putText(frame, label, (10, fh - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=20.0)
    ap.add_argument('--interval', type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print(f'Loading model A: {LABEL_A} <- {os.path.relpath(CKPT_A, BASE)}')
    model_a = BlazeHandLandmark().to(device)
    model_a.load_state_dict(torch.load(CKPT_A, map_location=device))
    model_a.eval()

    print(f'Loading model B: {LABEL_B} <- {os.path.relpath(CKPT_B, BASE)}')
    model_b = BlazeHandLandmark().to(device)
    model_b.load_state_dict(torch.load(CKPT_B, map_location=device))
    model_b.eval()

    detector = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('ERROR: Cannot open webcam.')
        sys.exit(1)

    out_dir = os.path.join(BASE, 'R2-05', 'webcam_frames')
    os.makedirs(out_dir, exist_ok=True)

    print(f'Capturing for {args.seconds:.0f}s, saving a side-by-side frame every '
          f'{args.interval:.1f}s. Put your hand in front of the camera now.')

    t_start = time.time()
    next_save = t_start
    saved = 0
    detected_frames = 0
    total_frames = 0

    while time.time() - t_start < args.seconds:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)

        frame_a = frame.copy()
        frame_b = frame.copy()

        if result.multi_hand_landmarks:
            detected_frames += 1
            hand_lm = result.multi_hand_landmarks[0]
            x1, y1, x2, y2 = hand_bbox(hand_lm.landmark, fw, fh)
            x1, y1, x2, y2 = square_crop(x1, y1, x2, y2, fw, fh)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                cw, ch = x2 - x1, y2 - y1
                inp_rgb = cv2.cvtColor(cv2.resize(crop, (256, 256)), cv2.COLOR_BGR2RGB)
                inp = (torch.from_numpy(inp_rgb.transpose(2, 0, 1))
                       .float().unsqueeze(0) / 255.0).to(device)

                with torch.no_grad():
                    flag_a, _, lm_a = model_a(inp)
                    flag_b, _, lm_b = model_b(inp)

                kps_a = lm_a[0, :, :2].cpu().numpy() * 256
                kps_a[:, 0] = kps_a[:, 0] * (cw / 256.0) + x1
                kps_a[:, 1] = kps_a[:, 1] * (ch / 256.0) + y1
                kps_a = kps_a.astype(int)

                kps_b = lm_b[0, :, :2].cpu().numpy() * 256
                kps_b[:, 0] = kps_b[:, 0] * (cw / 256.0) + x1
                kps_b[:, 1] = kps_b[:, 1] * (ch / 256.0) + y1
                kps_b = kps_b.astype(int)

                draw_overlay(frame_a, kps_a, flag_a[0].item(), x1, y1, x2, y2, LABEL_A)
                draw_overlay(frame_b, kps_b, flag_b[0].item(), x1, y1, x2, y2, LABEL_B)

        combined = np.hstack([frame_a, frame_b])
        cv2.line(combined, (fw, 0), (fw, fh), (80, 80, 80), 2)

        if time.time() >= next_save:
            saved += 1
            path = os.path.join(out_dir, f'compare_{saved:03d}.png')
            cv2.imwrite(path, combined)
            print(f'  [{time.time()-t_start:5.1f}s] saved {path} '
                  f'(hand detected this frame: {bool(result.multi_hand_landmarks)})')
            next_save += args.interval

    cap.release()
    detector.close()
    det_rate = 100.0 * detected_frames / max(total_frames, 1)
    print(f'\nDone. {saved} comparison frames saved to {out_dir}')
    print(f'Hand-detection rate during capture: {det_rate:.1f}% of {total_frames} raw frames read')


if __name__ == '__main__':
    main()
