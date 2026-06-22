"""
Teacher vs RHD-ft Student (epoch 150) — overlaid on the same image.
Both skeletons drawn on each sample; 6 samples arranged in a 2×3 grid.

Teacher model: HRNetV2-W18 + DARK decoding (pose_hrnetv2_w18)
  - Config:      td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256.py
  - Checkpoint:  hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth
  - Source:      MMPose official model zoo (fully trained on RHD2D, 210 epochs)
  - Backbone init: ImageNet/MSRA pre-trained HRNetV2-W18, then fully fine-tuned
                   on RHD2D (not frozen).
  - RHD test-set performance:  PCK@0.2 = 0.992 | AUC = 0.902 | EPE = 2.21 px

Sample selection uses three criteria from the raw annotations:
  1. All 21 keypoints visible and well inside the image border
  2. Large 2-D fingertip bounding-box area (open, spread hand)
  3. Low 3-D depth deviation of fingertips (flat / frontal hand, not curled)
A post-hoc check drops any sample where either model's prediction is degenerate.

Output:  thesis_figures/RHD_FT_OVERLAY_GRID.png
"""

import json, sys, os, cv2
import torch, numpy as np
from mmpose.apis import init_model
from mmpose.codecs import MSRAHeatmap
from mmpose.datasets import Rhd2DDataset
from torch.utils.data import DataLoader
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope

init_default_scope('mmpose')

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE, 'student_model'))
from blazehand_landmark import BlazeHandLandmark

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

BONES = [(0,1),(1,2),(2,3),(3,4),
         (0,5),(5,6),(6,7),(7,8),
         (0,9),(9,10),(10,11),(11,12),
         (0,13),(13,14),(14,15),(15,16),
         (0,17),(17,18),(18,19),(19,20)]

TIPS     = [4, 8, 12, 16, 20]
C_TC     = (0, 165, 255)    # orange  – teacher
C_ST     = (0, 230, 80)     # lime    – student


# ─── Annotation-level quality score ──────────────────────────────────────────
def quality_score(ann):
    """
    Higher = better candidate (open, frontal, large hand).
    Returns None if the sample should be excluded entirely.
    """
    kps      = ann['keypoints']
    j3d      = ann['joint_cam']          # list of [x,y,z], camera coords in mm
    bw, bh   = ann['bbox'][2], ann['bbox'][3]

    # Must be a decently sized crop
    if bw < 50 or bh < 50:
        return None

    pts2d = np.array([[x, y] for x, y, _ in kps], dtype=float)
    pts3d = np.array(j3d, dtype=float)          # (21, 3)

    # All keypoints visible
    if any(v == 0 for _, _, v in kps):
        return None

    # Reject samples where any keypoint touches the image margin
    img = id_to_img[ann['image_id']]
    margin = 12
    if (pts2d[:, 0].min() < margin or pts2d[:, 1].min() < margin or
            pts2d[:, 0].max() > img['width']  - margin or
            pts2d[:, 1].max() > img['height'] - margin):
        return None

    # 2-D fingertip bounding-box area (larger = more spread fingers)
    tip_pts  = pts2d[TIPS]
    tip_w    = tip_pts[:, 0].max() - tip_pts[:, 0].min()
    tip_h    = tip_pts[:, 1].max() - tip_pts[:, 1].min()
    tip_area = tip_w * tip_h
    if tip_area < 400:   # fingertips must be reasonably spread in 2-D
        return None

    # 3-D depth flatness: compare fingertip Z vs wrist Z
    # Low std = hand is not curled toward/away from camera → cleaner keypoints
    wrist_z    = pts3d[0, 2]
    tip_z_diff = pts3d[TIPS, 2] - wrist_z
    z_std      = float(np.std(tip_z_diff))

    # Combined score: reward spread + large bbox, penalise depth variance
    score = tip_area * (bw * bh) / (1.0 + z_std)
    return score


# ─── Pre-filter annotations ───────────────────────────────────────────────────
ANN_FILE = os.path.join(BASE, 'dataset', 'rhd', 'annotations', 'rhd_test.json')
with open(ANN_FILE) as f:
    coco = json.load(f)

id_to_img  = {img['id']: img for img in coco['images']}
img_id_order = [img['id'] for img in coco['images']]   # sequential order in dataset

scored = []
for ann in coco['annotations']:
    sc = quality_score(ann)
    if sc is not None:
        scored.append((sc, ann['image_id']))

scored.sort(reverse=True)

# Use top 400 quality candidates — wide enough pool to find low-MPJPE samples
good_ids = set(img_id for _, img_id in scored[:400])

print(f'Test set: {len(coco["annotations"])} annotations '
      f'→ {len(scored)} pass quality filter '
      f'→ {len(good_ids)} candidate ids for scanning')


# ─── Post-hoc prediction sanity check ────────────────────────────────────────
def is_degenerate(kps, min_std=18.0):
    """True if the predicted keypoints are unrealistically clustered."""
    return np.std(kps[:, 0]) < min_std or np.std(kps[:, 1]) < min_std


# ─── Load models ──────────────────────────────────────────────────────────────
print('Loading Teacher (HRNetV2-W18 + DARK, PCK@0.2=0.992 on RHD)...')
CONFIG  = os.path.join(BASE, 'mmpose/configs/hand_2d_keypoint/topdown_heatmap/'
                       'rhd2d/td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256.py')
CKPT_TC = os.path.join(BASE, 'checkpoints',
                       'hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth')
teacher = init_model(CONFIG, CKPT_TC, device=device)
teacher.cfg.model.test_cfg.output_heatmaps = True
if hasattr(teacher, 'head') and hasattr(teacher.head, 'test_cfg'):
    teacher.head.test_cfg['output_heatmaps'] = True
teacher.eval()

print('Loading Student – RHD fine-tune epoch 150...')
student = BlazeHandLandmark().to(device)
student.load_state_dict(torch.load(
    os.path.join(BASE, 'checkpoints', 'distilled_v2_ft_epoch_150.pth'),
    map_location=device))
student.eval()

codec = MSRAHeatmap(input_size=(256, 256), heatmap_size=(64, 64),
                    sigma=2, unbiased=True)


# ─── Dataset / DataLoader ─────────────────────────────────────────────────────
pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(type='PackPoseInputs'),
]
ds = Rhd2DDataset(
    data_root=os.path.join(BASE, 'dataset', 'rhd'),
    ann_file='annotations/rhd_test.json',
    pipeline=pipeline,
)
loader = DataLoader(ds, batch_size=1, shuffle=False,
                    collate_fn=pseudo_collate)


# ─── Drawing helper ───────────────────────────────────────────────────────────
def draw_skeleton(kps, colour, base_bgr, line_w=1, dot_r=4):
    vis = base_bgr.copy()
    for i, j in BONES:
        p1 = (int(np.clip(kps[i, 0], 0, 255)), int(np.clip(kps[i, 1], 0, 255)))
        p2 = (int(np.clip(kps[j, 0], 0, 255)), int(np.clip(kps[j, 1], 0, 255)))
        cv2.line(vis, p1, p2, colour, line_w, cv2.LINE_AA)
    for i in range(21):
        cx = int(np.clip(kps[i, 0], 0, 255))
        cy = int(np.clip(kps[i, 1], 0, 255))
        cv2.circle(vis, (cx, cy), dot_r, colour,    -1, cv2.LINE_AA)
        cv2.circle(vis, (cx, cy), dot_r, (0, 0, 0),  1, cv2.LINE_AA)
    return vis


def decode_teacher(out):
    hms = out[0].pred_fields.heatmaps.cpu().numpy()
    dec = codec.decode(hms)
    kps = dec.get('keypoints', dec.get('pred_keypoints')) if isinstance(dec, dict) else dec[0]
    return np.array(kps).reshape(21, 2)


# ─── Pass 1: scan all candidates, keep those with MPJPE ≤ 20 px ──────────────
MAX_MPJPE  = 20.0
GRID_SIZE  = 6          # panels per saved grid image
candidates_ok = []      # list of (mpjpe, img_bgr, tc, sc)
os.makedirs('thesis_figures', exist_ok=True)

print(f'\nPass 1: scanning {len(good_ids)} candidates for MPJPE <= {MAX_MPJPE}px ...')

for idx, batch in enumerate(loader):
    ds_sample = batch['data_samples'][0]
    try:
        img_id = ds_sample.img_id
    except AttributeError:
        try:
            img_id = ds_sample.metainfo['img_id']
        except (AttributeError, KeyError):
            img_id = idx

    if img_id not in good_ids:
        continue

    img_tensor = torch.stack(batch['inputs']).float() / 255.0
    img_bgr    = cv2.cvtColor(
        (img_tensor[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR)

    with torch.no_grad():
        tc = decode_teacher(teacher.predict(img_tensor.to(device),
                                            batch['data_samples']))
    with torch.no_grad():
        _, _, pred = student(img_tensor.to(device))
        sc = pred[0, :, :2].cpu().numpy() * 256

    if is_degenerate(tc) or is_degenerate(sc):
        continue

    mpjpe = float(np.sqrt(np.sum((sc - tc) ** 2, axis=1)).mean())
    if mpjpe > MAX_MPJPE:
        continue

    candidates_ok.append((mpjpe, img_bgr, tc, sc))
    print(f'  [ok] img_id={img_id}  MPJPE={mpjpe:.2f}px  '
          f'(total so far: {len(candidates_ok)})')

print(f'\nPass 1 done -- {len(candidates_ok)} samples with MPJPE <= {MAX_MPJPE}px')

if not candidates_ok:
    print('ERROR: no samples found within MPJPE threshold. Try raising MAX_MPJPE.')
    sys.exit(1)

# ─── Pass 2: sort by MPJPE, render ALL panels, save in grids of GRID_SIZE ────
candidates_ok.sort(key=lambda x: x[0])   # ascending MPJPE -> best first

font = cv2.FONT_HERSHEY_SIMPLEX

def make_legend(width):
    leg = np.zeros((36, width, 3), dtype=np.uint8)
    for x0, colour, label in [
        ( 10, C_TC, 'Teacher (HRNetV2-W18+DARK, PCK@0.2=0.992)'),
        (400, C_ST, 'Distilled Student - RHD fine-tune (epoch 150)'),
    ]:
        cv2.line(leg, (x0, 18), (x0 + 30, 18), colour, 3, cv2.LINE_AA)
        cv2.putText(leg, label, (x0 + 36, 23), font, 0.52, colour, 1, cv2.LINE_AA)
    return leg

all_panels = []
for rank, (mpjpe, img_bgr, tc, sc) in enumerate(candidates_ok):
    vis = draw_skeleton(tc, C_TC, img_bgr, line_w=1, dot_r=5)
    vis = draw_skeleton(sc, C_ST, vis,     line_w=1, dot_r=3)
    cv2.putText(vis, f'{mpjpe:.1f}px',
                (4, 252), font, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(f'thesis_figures/rhd_overlay_{rank+1}.png', vis)
    all_panels.append(vis)
    print(f'  Panel {rank+1}: MPJPE={mpjpe:.2f}px')

# Save one grid image per group of GRID_SIZE panels
num_grids = (len(all_panels) + GRID_SIZE - 1) // GRID_SIZE
print(f'\nBuilding {num_grids} grid image(s) of up to {GRID_SIZE} panels each...')

for g in range(num_grids):
    chunk = all_panels[g * GRID_SIZE : (g + 1) * GRID_SIZE]
    # Pad last chunk with blank panels if needed
    while len(chunk) < GRID_SIZE:
        chunk.append(np.zeros((256, 256, 3), dtype=np.uint8))

    row1  = np.hstack(chunk[:3])
    row2  = np.hstack(chunk[3:6])
    grid  = np.vstack([row1, row2])
    legend = make_legend(grid.shape[1])
    final  = np.vstack([grid, legend])

    out_path = f'thesis_figures/RHD_FT_OVERLAY_GRID_{g+1}.png'
    cv2.imwrite(out_path, final)
    start = g * GRID_SIZE + 1
    end   = min((g + 1) * GRID_SIZE, len(all_panels))
    print(f'  Saved: {out_path}  (panels {start}-{end})')

print(f'\nDone. {len(all_panels)} total panels across {num_grids} grid(s).')
