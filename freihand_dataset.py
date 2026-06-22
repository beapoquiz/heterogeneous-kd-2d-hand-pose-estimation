import json
import os
import zipfile
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


def _ensure_extracted(zip_path, expected_folder):
    """Extract ZIP if the target folder doesn't exist yet."""
    if os.path.isdir(expected_folder):
        return
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Neither folder nor zip found:\n  {expected_folder}\n  {zip_path}")
    print(f"Extracting {os.path.basename(zip_path)} — this may take a few minutes...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(os.path.dirname(zip_path))
    print(f"Done → {expected_folder}")


def _project_3d_to_2d(xyz, K):
    """
    xyz : (21, 3) in camera coordinates (metres)
    K   : (3, 3) intrinsic matrix
    returns: (21, 2) pixel coordinates
    """
    xyz = np.array(xyz, dtype=np.float32)   # (21, 3)
    K   = np.array(K,   dtype=np.float32)   # (3, 3)
    uvd = (K @ xyz.T).T                      # (21, 3)
    uv  = uvd[:, :2] / uvd[:, 2:3]          # (21, 2)  pixel coords in 224×224
    return uv


class FreiHandDataset(Dataset):
    """
    PyTorch Dataset for the FreiHAND dataset.

    Loads images (224×224), projects 3D joints to 2D using camera intrinsics,
    rescales everything to `input_size × input_size`, and optionally applies
    random augmentations that match the RHD training pipeline.

    Args:
        freihand_root : path to the extracted FreiHAND_pub_v2 folder
                        (or the zip lives one level up, alongside this folder)
        input_size    : target resolution (default 256, matching BlazeHand)
        augment       : enable random flip / scale / rotation / colour jitter
        max_samples   : cap dataset size (useful for quick experiments)
    """

    SRC_SIZE = 224  # FreiHAND images are 224×224

    def __init__(self, freihand_root, input_size=256, augment=False, max_samples=None):
        # Only attempt ZIP extraction if the annotation files aren't already present
        K_path   = os.path.join(freihand_root, 'training_K.json')
        xyz_path = os.path.join(freihand_root, 'training_xyz.json')

        if not os.path.isfile(K_path):
            raise FileNotFoundError(f"training_K.json not found in {freihand_root}")
        if not os.path.isfile(xyz_path):
            raise FileNotFoundError(f"training_xyz.json not found in {freihand_root}")

        with open(K_path)   as f: self.K_list   = json.load(f)
        with open(xyz_path) as f: self.xyz_list = json.load(f)

        assert len(self.K_list) == len(self.xyz_list), "K / xyz length mismatch"

        self.img_dir    = os.path.join(freihand_root, 'training', 'rgb')
        self.input_size = input_size
        self.augment    = augment
        self.scale      = input_size / self.SRC_SIZE   # 256/224 ≈ 1.143

        n = len(self.K_list)
        self.indices = list(range(min(n, max_samples) if max_samples else n))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]

        # ── load image ──────────────────────────────────────────────────────
        img_path = os.path.join(self.img_dir, f'{idx:08d}.jpg')
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # (224, 224, 3)

        # ── 2D keypoints in 224×224 space ───────────────────────────────────
        kps = _project_3d_to_2d(self.xyz_list[idx], self.K_list[idx])  # (21,2)

        # ── resize to input_size×input_size ─────────────────────────────────
        img = cv2.resize(img, (self.input_size, self.input_size))
        kps = kps * self.scale                               # (21, 2) in 256×256

        # ── optional augmentation ────────────────────────────────────────────
        if self.augment:
            img, kps = _augment(img, kps, self.input_size)

        # ── clamp & convert ──────────────────────────────────────────────────
        kps = np.clip(kps, 0, self.input_size - 1).astype(np.float32)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3,H,W)
        kps_t = torch.from_numpy(kps)                                     # (21,2)

        return img_t, kps_t


# ─── augmentation helpers ─────────────────────────────────────────────────────

def _augment(img, kps, size):
    h, w = img.shape[:2]

    # random horizontal flip
    if np.random.random() < 0.5:
        img = img[:, ::-1, :].copy()
        kps = kps.copy()
        kps[:, 0] = (w - 1) - kps[:, 0]

    # random scale (0.75 – 1.25) with centre-crop / pad
    if np.random.random() < 0.8:
        s = np.random.uniform(0.75, 1.25)
        nh, nw = int(h * s), int(w * s)
        img_s  = cv2.resize(img, (nw, nh))
        kps_s  = kps * s
        if s > 1.0:
            y0 = (nh - h) // 2
            x0 = (nw - w) // 2
            img = img_s[y0:y0+h, x0:x0+w]
            kps = kps_s - np.array([[x0, y0]], dtype=np.float32)
        else:
            py = (h - nh) // 2
            px = (w - nw) // 2
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            canvas[py:py+nh, px:px+nw] = img_s
            img = canvas
            kps = kps_s + np.array([[px, py]], dtype=np.float32)

    # random rotation (±45°)
    if np.random.random() < 0.6:
        angle = np.random.uniform(-45, 45)
        cx, cy = w / 2.0, h / 2.0
        M   = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h))
        kps_h = np.hstack([kps, np.ones((len(kps), 1), dtype=np.float32)])
        kps   = (M @ kps_h.T).T

    # colour jitter
    if np.random.random() < 0.5:
        img = img.astype(np.float32)
        img *= np.random.uniform(0.7, 1.3)
        img  = np.clip(img, 0, 255).astype(np.uint8)

    return img, kps


def make_train_val_loaders(freihand_root, input_size=256,
                           val_frac=0.05, batch_size=32, seed=42):
    """
    Split the 130 240-sample training set into train / val by unique scene.

    FreiHAND has 32 560 unique scenes × 4 background variants = 130 240 images.
    We split by unique scene index to avoid data leakage.
    """
    from torch.utils.data import DataLoader, Subset

    full = FreiHandDataset(freihand_root, input_size=input_size, augment=False)
    n    = len(full)

    UNIQUE  = 32_560                          # scenes before 4-bg augmentation
    n_val_u = max(1, int(UNIQUE * val_frac))  # unique scenes held out
    rng     = np.random.default_rng(seed)
    all_u   = np.arange(UNIQUE)
    rng.shuffle(all_u)
    val_u   = set(all_u[:n_val_u].tolist())

    # Each scene i appears at indices i, i+UNIQUE, i+2*UNIQUE, i+3*UNIQUE
    train_idx, val_idx = [], []
    for idx in range(n):
        scene = idx % UNIQUE
        (val_idx if scene in val_u else train_idx).append(idx)

    train_ds = Subset(FreiHandDataset(freihand_root, input_size, augment=True),  train_idx)
    val_ds   = Subset(FreiHandDataset(freihand_root, input_size, augment=False), val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,          shuffle=False,
                              num_workers=0)

    print(f"FreiHAND split — train: {len(train_ds):,}  val: {len(val_ds):,}")
    return train_loader, val_loader
