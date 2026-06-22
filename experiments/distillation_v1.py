import torch
import torch.nn as nn
import numpy as np
import sys
import os
from mmpose.apis import init_model
from mmpose.codecs import MSRAHeatmap
from mmpose.datasets import Rhd2DDataset
from torch.utils.data import DataLoader
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope

def main():
    # --- 1. INITIALIZE --- dito yung teacher and student
    init_default_scope('mmpose')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Add student folder to path
    STUDENT_FOLDER = os.path.join(os.getcwd(), 'student_model')
    if STUDENT_FOLDER not in sys.path:
        sys.path.append(STUDENT_FOLDER)

    from blazehand_landmark import BlazeHandLandmark # type: ignore

    # --- 2. SETUP MODELS --- yung config ng teacher
    print("Loading Teacher (HRNet)...")
    CONFIG = 'mmpose/configs/hand_2d_keypoint/topdown_heatmap/rhd2d/td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256.py'
    CHECKPOINT = 'checkpoints/hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth'
    
    teacher = init_model(CONFIG, CHECKPOINT, device=str(device))
    
    # --- DATA SURGERY: Force Heatmap Output --- pagforce ng heatmap palabas (kahit mga 1 month lang HAHAHA)
    teacher.cfg.model.test_cfg.output_heatmaps = True
    if hasattr(teacher, 'head'):
        teacher.head.output_heatmaps = True
        if hasattr(teacher.head, 'test_cfg'):
            if isinstance(teacher.head.test_cfg, dict):
                teacher.head.test_cfg['output_heatmaps'] = True
            else:
                teacher.head.test_cfg.output_heatmaps = True
    
    teacher.eval()

    print("Loading Student (BlazeHand)...")
    student = BlazeHandLandmark().to(device)
    student.train()

    # --- 3. DATA & CODEC ---
    # MSRAHeatmap allows for refined sub-pixel coordinate extraction (Refined Hard-Max)
    codec = MSRAHeatmap(input_size=(256, 256), heatmap_size=(64, 64), sigma=2, unbiased=True)
    DATA_ROOT = r'C:\Users\Bea Juliana Poquiz\Desktop\mmpose_thesis\dataset\rhd'
    
    pipeline = [
        dict(type='LoadImage'),
        dict(type='GetBBoxCenterScale'),
        dict(type='TopdownAffine', input_size=(256, 256)),
        dict(type='PackPoseInputs')
    ]

    print("Preparing Dataset...")
    train_dataset = Rhd2DDataset(data_root=DATA_ROOT, ann_file='annotations/rhd_train.json', pipeline=pipeline)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        collate_fn=pseudo_collate, 
        num_workers=0
    )

    # --- 4. OPTIMIZER & LOSS ---
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # --- 5. TRAINING LOOP ---
    print("\n" + "="*40)
    print("STARTING OPTIMIZED DISTILLATION (210 Epochs)")
    print("Strategy: MSRA Refined Hard-Max Targets")
    print("="*40)

    for epoch in range(210):
        epoch_loss = 0.0
        for i, batch in enumerate(train_loader):
            # A. Prepare Inputs [Batch, 3, 256, 256]
            images = torch.stack(batch['inputs']).to(device).float() / 255.0
            
            # B. TEACHER PHASE (Sub-pixel Ground Truth Extraction)
            with torch.no_grad():
                output_samples = teacher.predict(images, batch['data_samples'])
                batch_coords = []
                
                for sample in output_samples:
                    hms = None
                    if hasattr(sample, 'pred_fields'):
                        hms = sample.pred_fields.get('heatmaps', sample.pred_fields.get('pred_heatmaps', None))
                    if hms is None:
                        hms = sample.get('heatmaps', sample.get('pred_heatmaps', None))

                    if hms is None:
                        raise RuntimeError("Teacher output NO heatmaps. Ensure you are using the Heatmap model.")

                    if torch.is_tensor(hms):
                        hms = hms.detach().cpu().numpy()
                    
                    # MSRA Decoding for sub-pixel accuracy
                    decoded = codec.decode(hms)
                    if isinstance(decoded, dict):
                        # We use .get() and type-hinting to satisfy Pylance
                        raw_keypoints = decoded.get('keypoints')
                        if isinstance(raw_keypoints, (list, np.ndarray)):
                            coords = raw_keypoints[0]
                        else:
                            coords = raw_keypoints
                    else:
                        coords = decoded[0]
                
                # Create targets and fix dimensionality mismatch [Batch, 21, 2]
                teacher_targets = torch.from_numpy(np.stack(batch_coords)).to(device).float()
                if teacher_targets.dim() == 4: # Squeeze extra dimension if present [B, 1, 21, 2] -> [B, 21, 2]
                    teacher_targets = teacher_targets.squeeze(1)

            # C. STUDENT PHASE
            optimizer.zero_grad()
            
            # BlazeHand output: _, _, [Batch, 21, 3]
            _, _, pred_landmarks = student(images)
            
            # Slice to 2D and scale to pixel space (256x256)
            student_coords = pred_landmarks[:, :, :2] * 256
            
            # Ensure student and teacher shapes match exactly to avoid broadcasting errors
            teacher_targets = teacher_targets.view(student_coords.shape)
            
            # Calculate Knowledge Distillation Loss
            loss = criterion(student_coords, teacher_targets)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/210] | Step [{i}/{len(train_loader)}] | Loss: {loss.item():.4f}")
                # Debug check: Print first joint of the first sample to verify float precision
                sample_coord = teacher_targets[0, 0].cpu().numpy()
                print(f"   > Refined Target Check (Joint 0): {sample_coord}")

        # Summary and Checkpoint
        avg_loss = epoch_loss / len(train_loader)
        print(f"===> Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.6f}")

        if (epoch + 1) % 10 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint_name = f'checkpoints/distilled_student_epoch_{epoch+1}.pth'
            torch.save(student.state_dict(), checkpoint_name)
            print(f"Successfully saved: {checkpoint_name}")

if __name__ == '__main__':
    main()