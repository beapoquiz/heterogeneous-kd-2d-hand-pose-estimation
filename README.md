# Hand Pose Estimation via Knowledge Distillation

Undergraduate thesis project: transferring knowledge from a high-accuracy teacher model (HRNetV2-W18 + DARK) into a lightweight student model (BlazeHandLandmark) for real-time 2D hand keypoint estimation on the RHD dataset.

---

## Results

### Speed and Efficiency (RTX 4050 Laptop GPU, CUDA 12.1)

| Model | Params | Weight Size | GPU FPS | CPU FPS |
|---|---|---|---|---|
| Teacher — HRNetV2-W18 + DARK | 9.64 M | 37.2 MB | 6.9 | — |
| Student — BlazeHandLandmark (distilled) | 2.01 M | 7.7 MB | **139.0** | 25.6 |
| **Compression ratio** | **4.8×** | **4.8×** | **20.2× faster** | — |

- Student detection rate on RHD test set (2,727 images): **100%**
- Student GPU real-time: **YES** (139.0 FPS >> 30 FPS threshold)

### Accuracy on RHD Test Set

| Model | PCK@0.2 (vs GT) | MPJPE | AUC |
|---|---|---|---|
| Teacher — HRNetV2-W18 + DARK | **99.2%** | 2.21 px | 0.902 |
| Student — BlazeHandLandmark (distilled + fine-tuned) | **74.7%** | 37.0 px | 0.722 |
| Direct Supervision Baseline | 33.8% | 72.7 px | 0.458 |
| MediaPipe SDK (zmurez weights) | 30.1% | 104.3 px | 0.395 |

Full per-joint breakdown and fine-tuned model results are in the `results_*.txt` files.

---

## Architecture

**Teacher:** `HRNetV2-W18` backbone + DARK (unbiased heatmap decoding), trained for 210 epochs on RHD2D via MMPose. Outputs 21 heatmaps (64×64).

**Student:** `BlazeHandLandmark` — a PyTorch port of the MediaPipe hand landmark model. Outputs 21 normalised (x,y,z) coordinates directly.

**Loss function (both training stages):**
```
L = L_wing(p̂, p_T) + 0.002 × L_bone(p̂, p_T)
```
- Targets `p_T` are teacher heatmap-decoded coordinates (not ground-truth labels)
- Wing Loss: w=10, ε=2
- Bone Loss: MSE over 20 finger-segment lengths

---

## Setup

### 1. Clone and enter the repo
```bash
git clone <repo-url>
cd mmpose_thesis
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install openmim
mim install mmengine mmcv mmpose
pip install torch torchvision numpy opencv-python mediapipe matplotlib tqdm
```

### 4. Download the student starting weights

The student model starts from MediaPipe's pretrained BlazeHand weights. Download `blazehand_landmark.pth` from the [zmurez/blazepalm](https://github.com/zmurez/MediaPipePyTorch) releases and place it at `student_model/blazehand_landmark.pth`.

### 5. Download the teacher checkpoint
Download `hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth` from the [MMPose Model Zoo](https://mmpose.readthedocs.io/en/latest/model_zoo/hand_2d_keypoint.html) and place it in `checkpoints/`.

Also download the MMPose config for the teacher:
```
mmpose/configs/hand_2d_keypoint/topdown_heatmap/rhd2d/td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256.py
```
Place it at that relative path (clone the MMPose repo or copy from the official source).

### 6. Download the RHD dataset
Download RHD2D from the [RHD project page](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html) and extract to `dataset/rhd/`.

### 7. (Optional) Download FreiHAND
Download from the [FreiHAND project page](https://lmb.informatik.uni-freiburg.de/projects/freihand/) and place the zip files in `freihand/`. The dataset loader (`freihand_dataset.py`) will extract them automatically.

---

## Training

**Stage 1 — Knowledge Distillation (100 epochs)**
```bash
python distillation_v2.py
```
Saves checkpoints to `checkpoints/distilled_v2_epoch_*.pth`.

**Stage 2 — Fine-tuning with early stopping (≤50 epochs)**
```bash
python distillation_finetune.py
```
Starts from `checkpoints/distilled_v2_epoch_100.pth`. Saves best model as `checkpoints/distilled_v2_ft_best.pth`.

**Baseline — Direct supervision (no distillation)**
```bash
python directsup_baseline.py
```

**FreiHAND fine-tuning**
```bash
python finetune_freihand.py
```

---

## Evaluation

**Full benchmark (speed, size, detection rate)**
```bash
python thesis_benchmark.py
```

**Teacher model evaluation**
```bash
python teacher_comprehensive_eval.py
```

**Direct supervision baseline (ground truth)**
```bash
python directsup_gt_comprehensive_eval.py
```

**FreiHAND best model**
```bash
python freihand_ft_best_comprehensive_eval.py
python freihand_ft_best_tc_comprehensive_eval.py
```

**MediaPipe baseline comparison**
```bash
python mediapipe_gt_eval.py
python mediapipe_vs_blazehand_eval.py
```

**Three-way comparison (untrained / MediaPipe / distilled)**
```bash
python three_way_student_comparison.py
```

---

## Demo

**Live webcam (distilled model, epoch 100)**
```bash
python webcam_demo.py
```

**Live webcam (fine-tuned model, epoch 150)**
```bash
python webcam_demo_finetuned.py
```

---

## Project Structure

```
mmpose_thesis/
├── student_model/                      # BlazeHandLandmark architecture (MediaPipe PyTorch port)
│   ├── blazehand_landmark.py
│   ├── blazebase.py
│   └── ...
│
├── distillation_v2.py                  # Stage 1: KD training (100 ep)
├── distillation_finetune.py            # Stage 2: fine-tuning with early stopping
├── directsup_baseline.py               # Direct supervision baseline
├── finetune_freihand.py                # FreiHAND transfer learning
├── freihand_dataset.py                 # FreiHAND PyTorch Dataset loader
│
├── thesis_benchmark.py                 # Main benchmark (speed + size)
├── teacher_comprehensive_eval.py       # Teacher accuracy on RHD
├── directsup_gt_comprehensive_eval.py  # Direct supervision accuracy vs GT
├── freihand_ft_best_comprehensive_eval.py
├── freihand_ft_best_tc_comprehensive_eval.py
├── mediapipe_gt_eval.py                # MediaPipe SDK baseline
├── mediapipe_vs_blazehand_eval.py      # MediaPipe vs BlazeHand starting weights
├── three_way_student_comparison.py     # Untrained / MediaPipe / Distilled comparison
│
├── distillation_learning_curve.py      # Generate learning curve CSV
├── plot_learning_curve.py              # Plot learning curve from CSV
├── visualize_teacher_heatmap.py        # Teacher heatmap visualisation
├── webcam_demo.py                      # Live demo (distilled model)
├── webcam_demo_finetuned.py            # Live demo (fine-tuned model)
│
├── experiments/                        # Archived trial-and-error scripts
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Notes

- Model checkpoints (`checkpoints/`), datasets (`dataset/`, `freihand/`), generated figures (`thesis_figures/`), and the MMPose framework directory are excluded from version control via `.gitignore`.
- All archived experimental scripts are preserved in `experiments/` for reference.
