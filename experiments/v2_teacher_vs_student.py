import torch, sys, os, cv2, numpy as np
from mmpose.apis import init_model
from mmpose.codecs import MSRAHeatmap
from mmpose.datasets import Rhd2DDataset
from torch.utils.data import DataLoader
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
init_default_scope('mmpose')

sys.path.append(os.path.join(os.getcwd(), 'student_model'))
from blazehand_landmark import BlazeHandLandmark

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use epoch 100 — confirmed best on full test set
student = BlazeHandLandmark().to(device)
student.load_state_dict(torch.load(
    'checkpoints/distilled_v2_epoch_100.pth', map_location=device))
student.eval()

CONFIG = 'mmpose/configs/hand_2d_keypoint/topdown_heatmap/rhd2d/td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256.py'
CHECKPOINT = 'checkpoints/hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth'
teacher = init_model(CONFIG, CHECKPOINT, device=device)
teacher.cfg.model.test_cfg.output_heatmaps = True
if hasattr(teacher, 'head') and hasattr(teacher.head, 'test_cfg'):
    teacher.head.test_cfg['output_heatmaps'] = True
teacher.eval()

codec = MSRAHeatmap(input_size=(256,256), heatmap_size=(64,64), sigma=2, unbiased=True)
pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256,256)),
    dict(type='PackPoseInputs')
]
ds = Rhd2DDataset(
    data_root=r'C:\Users\Bea Juliana Poquiz\Desktop\mmpose_thesis\dataset\rhd',
    ann_file='annotations/rhd_test.json',
    pipeline=pipeline
)
loader = DataLoader(ds, batch_size=1, shuffle=False,
                    collate_fn=pseudo_collate)

BONES = [(0,1),(1,2),(2,3),(3,4),
         (0,5),(5,6),(6,7),(7,8),
         (0,9),(9,10),(10,11),(11,12),
         (0,13),(13,14),(14,15),(15,16),
         (0,17),(17,18),(18,19),(19,20)]

ORANGE = (0, 165, 255)
GREEN  = (0, 255, 0)

os.makedirs('thesis_figures', exist_ok=True)
images = []
TARGET = 6  # generate 6 samples

print(f'Generating {TARGET} Teacher vs Student visualizations...')

for idx, batch in enumerate(loader):
    if len(images) >= TARGET:
        break

    img_tensor = torch.stack(batch['inputs']).float() / 255.0

    with torch.no_grad():
        out = teacher.predict(img_tensor.to(device), batch['data_samples'])
        hms = out[0].pred_fields.heatmaps.cpu().numpy()
        decoded = codec.decode(hms)
        if isinstance(decoded, dict):
            tc = np.array(decoded['keypoints']).reshape(21, 2)
        else:
            tc = np.array(decoded[0]).reshape(21, 2)

    with torch.no_grad():
        _, _, pred = student(img_tensor.to(device))
        sc = pred[0, :, :2].cpu().numpy() * 256

    img_np = (img_tensor[0].permute(1,2,0).numpy()*255).astype(np.uint8)
    vis = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Draw teacher skeleton (orange)
    for i, j in BONES:
        p1 = (int(np.clip(tc[i,0],0,255)), int(np.clip(tc[i,1],0,255)))
        p2 = (int(np.clip(tc[j,0],0,255)), int(np.clip(tc[j,1],0,255)))
        cv2.line(vis, p1, p2, ORANGE, 1)
    for i in range(21):
        cv2.circle(vis, (int(np.clip(tc[i,0],0,255)),
                         int(np.clip(tc[i,1],0,255))), 4, ORANGE, -1)

    # Draw student skeleton (green)
    for i, j in BONES:
        p1 = (int(np.clip(sc[i,0],0,255)), int(np.clip(sc[i,1],0,255)))
        p2 = (int(np.clip(sc[j,0],0,255)), int(np.clip(sc[j,1],0,255)))
        cv2.line(vis, p1, p2, GREEN, 1)
    for i in range(21):
        cv2.circle(vis, (int(np.clip(sc[i,0],0,255)),
                         int(np.clip(sc[i,1],0,255))), 3, GREEN, -1)

    mpjpe = np.sqrt(np.sum((sc - tc)**2, axis=1)).mean()
    cv2.putText(vis, f'MPJPE:{mpjpe:.1f}px',
                (5, 248), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    images.append(vis)
    fname = f'thesis_figures/v2_sample_{len(images)}.png'
    cv2.imwrite(fname, vis)
    print(f'  Saved: {fname} | MPJPE: {mpjpe:.2f}px')

# Build 2x3 grid
if len(images) == 6:
    row1 = np.hstack(images[:3])
    row2 = np.hstack(images[3:])
    grid = np.vstack([row1, row2])

    # Add legend
    legend = np.zeros((35, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(legend, 'Orange = Teacher (HRNetV2-W18)',
                (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ORANGE, 1)
    cv2.putText(legend, 'Green = Student (Distilled v2, Epoch 100)',
                (370, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 1)
    final = np.vstack([grid, legend])
    cv2.imwrite('thesis_figures/TEACHER_VS_STUDENT_V2_GRID.png', final)
    print()
    print('Saved: thesis_figures/TEACHER_VS_STUDENT_V2_GRID.png')