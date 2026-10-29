#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

# Allow PIL to load mildly truncated images when possible
ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================================================
# ARGUMENTS
# =========================================================
def get_args():
    p = argparse.ArgumentParser("Semi supervised strawberry segmentation")

    # Annotated COCO dataset
    p.add_argument(
        "--coco_root",
        type=Path,
        default=Path("/home/ehr/Desktop/anupama/CPP_Strawberry_RGB_MS_2025-26/Unprocessed_data/all.v1i.coco"),
        help="COCO export root containing train valid test with _annotations.coco.json"
    )

    # Built labeled dataset target
    p.add_argument(
        "--labeled_root",
        type=Path,
        default=Path("/home/ehr/Desktop/anupama/plant/data_labeled_from_coco"),
        help="Where converted train/val images+masks will be written"
    )

    # Multiple unlabeled folders
    p.add_argument(
        "--unlabeled_dirs",
        type=str,
        default=",".join([
            "/home/ehr/Desktop/anupama/CPP_Strawberry_RGB_MS_2025-26/Unprocessed_data/2025_04",
            "/home/ehr/Desktop/anupama/CPP_Strawberry_RGB_MS_2025-26/Unprocessed_data/2025_05",
            "/home/ehr/Desktop/anupama/CPP_Strawberry_RGB_MS_2025-26/Unprocessed_data/2025_06",
            "/home/ehr/Desktop/anupama/CPP_Strawberry_RGB_MS_2025-26/Unprocessed_data/20250814_Strawberry_RGB_5m",
        ]),
        help="Comma separated unlabeled image directories"
    )

    p.add_argument(
        "--save_dir",
        type=Path,
        default=Path("/home/ehr/Desktop/anupama/plant/runs_semi_supervised"),
    )

    p.add_argument("--img_size", type=int, default=768)
    p.add_argument("--batch_size_labeled", type=int, default=2)
    p.add_argument("--batch_size_unlabeled", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--pseudo_threshold", type=float, default=0.90)
    p.add_argument("--lambda_u", type=float, default=0.5)
    p.add_argument("--ema_decay", type=float, default=0.999)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    p.add_argument("--gpu", type=int, default=0)

    return p.parse_args()

A = get_args()

# =========================================================
# GLOBALS
# =========================================================
CLASS_NAMES = [
    "healthy",
    "moderately_healthy",
    "moderately_unhealthy",
    "unhealthy",
    "severely_unhealthy",
]

NUM_CLASSES = 5
IGNORE_INDEX = 255

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# COCO class mapping on disk:
# 0 background, 1..5 plant classes
COCO_NAME_TO_ID = {
    "healthy": 1,
    "moderately_healthy": 2,
    "moderately_unhealthy": 3,
    "unhealthy": 4,
    "severely_unhealthy": 5,
    "background": 0,
}

# =========================================================
# DEVICE
# =========================================================
def resolve_device():
    if A.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device(f"cuda:{A.gpu}")

    if A.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")

    if A.device == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device(f"cuda:{A.gpu}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = resolve_device()
IS_CUDA = DEVICE.type == "cuda"
AMP_ENABLED = bool(A.amp) and IS_CUDA

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

random.seed(A.seed)
np.random.seed(A.seed)
torch.manual_seed(A.seed)
if IS_CUDA:
    torch.cuda.set_device(DEVICE)

print(f"Device: {DEVICE} | AMP: {AMP_ENABLED}")

# =========================================================
# HELPERS
# =========================================================
def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def load_rgb_image_safe(path: Path) -> np.ndarray:
    """
    Safely load an RGB image as a numpy array.
    Raises OSError if the file cannot be decoded.
    """
    try:
        with Image.open(path) as img:
            return np.array(img.convert("RGB"))
    except (OSError, UnidentifiedImageError, ValueError) as e:
        raise OSError(f"Failed to load image: {path} | {e}") from e


def random_crop_pair(img: np.ndarray, msk: np.ndarray, size: int):
    h, w = msk.shape
    if h < size or w < size:
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        msk = cv2.copyMakeBorder(msk, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=IGNORE_INDEX)
        h, w = msk.shape

    y = np.random.randint(0, h - size + 1)
    x = np.random.randint(0, w - size + 1)

    return img[y:y+size, x:x+size], msk[y:y+size, x:x+size]


def random_crop_image(img: np.ndarray, size: int):
    h, w = img.shape[:2]
    if h < size or w < size:
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        h, w = img.shape[:2]

    y = np.random.randint(0, h - size + 1)
    x = np.random.randint(0, w - size + 1)
    return img[y:y+size, x:x+size]

# =========================================================
# COCO TO MASKS
# =========================================================
def canon(s: str) -> str:
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")


def poly_to_mask(seg, h, w):
    m = np.zeros((h, w), np.uint8)
    for s in seg:
        pts = np.array(s, np.float32).reshape(-1, 2).round().astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(m, [pts], 1)
    return m


def bbox_to_mask(bbox, h, w):
    x, y, bw, bh = bbox
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(w, int(x + bw)), min(h, int(y + bh))
    m = np.zeros((h, w), np.uint8)
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = 1
    return m


def build_labeled_from_coco(coco_root: Path, out_root: Path):
    print(f"Building labeled dataset from COCO: {coco_root}")

    try:
        from pycocotools import mask as R
    except Exception:
        R = None
        print("pycocotools not installed, RLE masks will be skipped if present")

    split_map = {
        "train": "train",
        "valid": "val",
        "val": "val",
    }

    built_any = False

    for src_split, dst_split in split_map.items():
        ann_path = coco_root / src_split / "_annotations.coco.json"
        if not ann_path.exists():
            continue

        coco = json.loads(ann_path.read_text())
        images = {im["id"]: im for im in coco.get("images", [])}
        cats = {c["id"]: canon(c["name"]) for c in coco.get("categories", [])}

        anns_by_image = {}
        for a in coco.get("annotations", []):
            anns_by_image.setdefault(a["image_id"], []).append(a)

        out_img_dir = out_root / dst_split / "images"
        out_mask_dir = out_root / dst_split / "masks"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_mask_dir.mkdir(parents=True, exist_ok=True)

        for image_id, im in tqdm(images.items(), desc=f"build {src_split}->{dst_split}"):
            fname = im["file_name"]
            src_img = None

            candidates = [
                coco_root / src_split / fname,
                coco_root / src_split / "images" / fname,
            ]
            for c in candidates:
                if c.exists():
                    src_img = c
                    break

            if src_img is None:
                continue

            arr = cv2.imread(str(src_img))
            if arr is None:
                continue

            h, w = arr.shape[:2]
            mask = np.zeros((h, w), np.uint8)

            for ann in anns_by_image.get(image_id, []):
                cname = cats.get(ann["category_id"], "background")
                cid = COCO_NAME_TO_ID.get(cname, 0)

                seg = ann.get("segmentation")
                bbox = ann.get("bbox")

                inst = None
                if isinstance(seg, list) and seg:
                    inst = poly_to_mask(seg, h, w)
                elif isinstance(seg, dict) and "counts" in seg and R is not None:
                    rle = R.frPyObjects(seg, h, w) if isinstance(seg.get("counts"), list) else seg
                    dec = R.decode(rle)
                    inst = dec if dec.ndim == 2 else dec.any(2).astype(np.uint8)
                elif bbox is not None:
                    inst = bbox_to_mask(bbox, h, w)

                if inst is not None:
                    mask[inst > 0] = cid

            dst_img = out_img_dir / Path(fname).name
            dst_mask = out_mask_dir / f"{Path(fname).stem}.png"

            if not dst_img.exists():
                dst_img.write_bytes(src_img.read_bytes())
            Image.fromarray(mask).save(dst_mask)
            built_any = True

    if not built_any:
        raise RuntimeError("Could not build labeled dataset from COCO root")

    print(f"Finished building labeled dataset at: {out_root}")

# =========================================================
# DATASETS
# =========================================================
class LabeledSegDataset(Dataset):
    """
    Disk mask values:
    0 = background
    1..5 = classes

    Training target:
    background -> IGNORE_INDEX
    1..5 -> 0..4
    """
    def __init__(self, root: Path, split: str, img_size: int = 768, aug: bool = False):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.aug = aug

        self.img_dir = self.root / split / "images"
        self.mask_dir = self.root / split / "masks"

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.2, 0.1) if aug else None

        imgs = sorted([p for p in self.img_dir.glob("*") if p.is_file() and is_image_file(p)])
        self.pairs = []

        for ip in imgs:
            matches = list(self.mask_dir.glob(ip.stem + ".*"))
            if matches:
                self.pairs.append((ip, matches[0]))

        if not self.pairs:
            raise RuntimeError(f"No labeled image mask pairs found in {self.img_dir}")

        print(f"Labeled {split}: {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]

        try:
            img = load_rgb_image_safe(ip)
        except OSError as e:
            raise RuntimeError(f"Failed to load labeled image: {ip} | {e}") from e

        raw = np.array(Image.open(mp).convert("L")).astype(np.int64)
        mask = np.where(raw == 0, IGNORE_INDEX, raw - 1).astype(np.int64)

        if self.aug:
            scale = np.random.uniform(0.8, 1.2)
            nh = max(16, int(img.shape[0] * scale))
            nw = max(16, int(img.shape[1] * scale))

            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

            img, mask = random_crop_pair(img, mask, self.img_size)

            if random.random() < 0.5:
                img = img[:, ::-1].copy()
                mask = mask[:, ::-1].copy()

            if self.color_jitter is not None:
                img = np.array(self.color_jitter(Image.fromarray(img)))
        else:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        img_t = self.to_tensor(Image.fromarray(img))
        img_t = self.normalize(img_t)

        return img_t, torch.from_numpy(mask).long()


class UnlabeledImageDataset(Dataset):
    def __init__(self, dirs, img_size: int = 768, max_retries: int = 20):
        self.img_size = img_size
        self.max_retries = max_retries
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.2, 0.1)

        self.images = []
        for d in dirs:
            d = Path(d)
            if not d.exists():
                print(f"Warning: unlabeled dir not found: {d}")
                continue
            for p in d.rglob("*"):
                if p.is_file() and is_image_file(p):
                    self.images.append(p)

        self.images = sorted(self.images)

        if not self.images:
            raise RuntimeError("No unlabeled images found in provided unlabeled directories")

        print(f"Unlabeled images: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def _strong_aug(self, img: np.ndarray):
        img = random_crop_image(img, self.img_size)
        if random.random() < 0.5:
            img = img[:, ::-1].copy()
        img = np.array(self.color_jitter(Image.fromarray(img)))
        return img

    def _weak_aug(self, img: np.ndarray):
        img = random_crop_image(img, self.img_size)
        if random.random() < 0.5:
            img = img[:, ::-1].copy()
        return img

    def __getitem__(self, idx):
        for attempt in range(self.max_retries):
            ip = self.images[idx]

            try:
                img = load_rgb_image_safe(ip)

                weak = self._weak_aug(img.copy())
                strong = self._strong_aug(img.copy())

                weak_t = self.normalize(self.to_tensor(Image.fromarray(weak)))
                strong_t = self.normalize(self.to_tensor(Image.fromarray(strong)))

                return weak_t, strong_t, str(ip)

            except OSError as e:
                print(f"[Warning] Skipping corrupted unlabeled image: {ip} | {e}")
                idx = random.randint(0, len(self.images) - 1)

        raise RuntimeError(
            f"Could not load a valid unlabeled image after {self.max_retries} attempts. "
            f"Please inspect the unlabeled dataset for corrupted files."
        )

# =========================================================
# MODEL
# =========================================================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.l1 = m.layer1
        self.l2 = m.layer2
        self.l3 = m.layer3
        self.l4 = m.layer4

    def forward(self, x):
        x = self.stem(x)
        c1 = self.l1(x)
        c2 = self.l2(c1)
        c3 = self.l3(c2)
        c4 = self.l4(c3)
        return c1, c2, c3, c4


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 256, rates=(1, 6, 12, 18)):
        super().__init__()

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
            )
        ] + [
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
            )
            for r in rates[1:]
        ])

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(True),
        )

        self.proj = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 1), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        xs = [b(x) for b in self.branches]
        g = self.pool(x)
        g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        xs.append(g)
        return self.proj(torch.cat(xs, dim=1))


class SegHead(nn.Module):
    def __init__(self, nc=NUM_CLASSES):
        super().__init__()
        self.aspp = ASPP(2048, 256)
        self.l3 = nn.Conv2d(1024, 256, 1)
        self.l2 = nn.Conv2d(512, 256, 1)
        self.l1 = nn.Conv2d(256, 256, 1)
        self.out = nn.Conv2d(256, nc, 1)

    def forward(self, fs):
        c1, c2, c3, c4 = fs

        p4 = self.aspp(c4)
        p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.l2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.l1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)

        return self.out(F.interpolate(p1, scale_factor=4, mode="bilinear", align_corners=False))


class SegModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.seg = SegHead(NUM_CLASSES)

    def forward(self, x):
        return self.seg(self.enc(x))

# =========================================================
# LOSSES
# =========================================================
def zero_loss_from_logits(logits):
    return logits.sum() * 0.0


def dice_loss_mc_ignore(logits, targets, ignore_index=IGNORE_INDEX, eps=1e-6):
    valid = (targets != ignore_index)
    if valid.sum() == 0:
        return zero_loss_from_logits(logits)

    probs = F.softmax(logits, dim=1)
    probs = probs.permute(0, 2, 3, 1)[valid]
    t = targets[valid]
    one_hot = F.one_hot(t, num_classes=NUM_CLASSES).float()

    num = 2 * (probs * one_hot).sum(0) + eps
    den = probs.sum(0) + one_hot.sum(0) + eps

    return 1 - (num / den).mean()


def supervised_loss(logits, targets):
    ce = F.cross_entropy(logits, targets, ignore_index=IGNORE_INDEX)
    dice = dice_loss_mc_ignore(logits, targets, ignore_index=IGNORE_INDEX)
    return 0.7 * ce + 0.3 * dice


def pseudo_label_loss(student_logits, teacher_logits, threshold=0.90):
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits, dim=1)
        confidence, pseudo = teacher_probs.max(dim=1)
        mask = confidence >= threshold

    if mask.sum() == 0:
        return zero_loss_from_logits(student_logits), 0.0

    ce = F.cross_entropy(student_logits, pseudo, reduction="none")
    loss = ce[mask].mean()
    keep_ratio = float(mask.float().mean().item())
    return loss, keep_ratio

# =========================================================
# METRICS
# =========================================================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)

    for x, y in tqdm(loader, desc="[Val]"):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True).long()

        logits = model(x)
        logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(1)

        valid = (y != IGNORE_INDEX)
        if valid.any():
            yv = y[valid].view(-1)
            pv = pred[valid].view(-1)
            idx = (yv * NUM_CLASSES + pv).to(torch.int64)
            conf += torch.bincount(idx.cpu(), minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)

    eps = 1e-12
    tp = torch.diag(conf).float()
    fp = conf.sum(0).float() - tp
    fn = conf.sum(1).float() - tp

    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    iou = tp / (tp + fp + fn + eps)
    supp = conf.sum(1).float()

    present = supp > 0

    def m(v):
        return v[present].mean().item() if present.any() else float("nan")

    macro = {
        "mPrecision": m(prec),
        "mRecall": m(rec),
        "mF1": m(f1),
        "mIoU": m(iou),
        "acc": (tp.sum() / conf.sum().clamp_min(1)).item(),
    }

    per_class = []
    for i in range(NUM_CLASSES):
        per_class.append({
            "class_id": i,
            "class_name": CLASS_NAMES[i],
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "iou": float(iou[i]),
            "support": int(supp[i]),
        })

    return macro, per_class, conf.numpy()

# =========================================================
# EMA
# =========================================================
@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, decay: float):
    sdict = student.state_dict()
    tdict = teacher.state_dict()

    for k in tdict.keys():
        if tdict[k].dtype.is_floating_point:
            tdict[k].mul_(decay).add_(sdict[k], alpha=1.0 - decay)
        else:
            tdict[k].copy_(sdict[k])

# =========================================================
# SAVE
# =========================================================
def save_checkpoint(model, path: Path, epoch: int, metric: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "metric_mIoU": metric,
        "model_state": model.state_dict(),
        "class_names": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "img_size": A.img_size,
    }, str(path))

# =========================================================
# MAIN
# =========================================================
def main():
    # Step 1: build labeled train/val from COCO if needed
    build_labeled_from_coco(A.coco_root, A.labeled_root)

    # Step 2: parse unlabeled dirs
    unlabeled_dirs = [s.strip() for s in A.unlabeled_dirs.split(",") if s.strip()]

    # Step 3: datasets
    labeled_train = LabeledSegDataset(A.labeled_root, "train", img_size=A.img_size, aug=True)
    labeled_val = LabeledSegDataset(A.labeled_root, "val", img_size=A.img_size, aug=False)
    unlabeled_train = UnlabeledImageDataset(unlabeled_dirs, img_size=A.img_size)

    pin = True if IS_CUDA else False

    dl_l = DataLoader(
        labeled_train,
        batch_size=A.batch_size_labeled,
        shuffle=True,
        num_workers=A.workers,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=(A.workers > 0),
    )

    dl_u = DataLoader(
        unlabeled_train,
        batch_size=A.batch_size_unlabeled,
        shuffle=True,
        num_workers=A.workers,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=(A.workers > 0),
    )

    dl_val = DataLoader(
        labeled_val,
        batch_size=A.batch_size_labeled,
        shuffle=False,
        num_workers=A.workers,
        pin_memory=pin,
        persistent_workers=(A.workers > 0),
    )

    student = SegModel().to(DEVICE)
    teacher = deepcopy(student).to(DEVICE)
    teacher.eval()

    optimizer = torch.optim.AdamW(student.parameters(), lr=A.lr, weight_decay=A.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)

    best_miou = -1.0
    A.save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, A.epochs + 1):
        student.train()
        teacher.eval()

        labeled_iter = iter(dl_l)
        unlabeled_iter = iter(dl_u)
        steps = min(len(dl_l), len(dl_u))

        pbar = tqdm(range(steps), desc=f"[Train {epoch}/{A.epochs}]")

        run_sup = 0.0
        run_unsup = 0.0
        run_keep = 0.0

        for step in pbar:
            try:
                xl, yl = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(dl_l)
                xl, yl = next(labeled_iter)

            try:
                xu_w, xu_s, _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(dl_u)
                xu_w, xu_s, _ = next(unlabeled_iter)

            xl = xl.to(DEVICE, non_blocking=True)
            yl = yl.to(DEVICE, non_blocking=True).long()
            xu_w = xu_w.to(DEVICE, non_blocking=True)
            xu_s = xu_s.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                logits_l = student(xl)
                logits_l = F.interpolate(logits_l, size=yl.shape[-2:], mode="bilinear", align_corners=False)
                loss_sup = supervised_loss(logits_l, yl)

                with torch.no_grad():
                    teacher_logits_u = teacher(xu_w)

                student_logits_u = student(xu_s)
                loss_unsup, keep_ratio = pseudo_label_loss(
                    student_logits_u,
                    teacher_logits_u,
                    threshold=A.pseudo_threshold,
                )

                loss = loss_sup + A.lambda_u * loss_unsup

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            update_ema(student, teacher, A.ema_decay)

            run_sup += float(loss_sup.item())
            run_unsup += float(loss_unsup.item())
            run_keep += keep_ratio

            pbar.set_postfix(
                sup=f"{run_sup / (step + 1):.4f}",
                unsup=f"{run_unsup / (step + 1):.4f}",
                keep=f"{run_keep / (step + 1):.3f}",
            )

        macro, per_class, _ = evaluate(student, dl_val)

        print(f"\nEpoch {epoch}")
        print(f"Validation acc   : {macro['acc']:.4f}")
        print(f"Validation mIoU  : {macro['mIoU']:.4f}")
        print(f"Validation mF1   : {macro['mF1']:.4f}")
        print(f"Validation mPrec : {macro['mPrecision']:.4f}")
        print(f"Validation mRec  : {macro['mRecall']:.4f}")

        for r in per_class:
            print(
                f"{r['class_name']:24s} "
                f"prec={r['precision']:.3f} "
                f"rec={r['recall']:.3f} "
                f"f1={r['f1']:.3f} "
                f"iou={r['iou']:.3f} "
                f"support={r['support']}"
            )

        save_checkpoint(student, A.save_dir / "last_model.pt", epoch, macro["mIoU"])

        if macro["mIoU"] > best_miou:
            best_miou = macro["mIoU"]
            save_checkpoint(student, A.save_dir / "best_model.pt", epoch, macro["mIoU"])
            print(f"Saved new best model at epoch {epoch} with mIoU={best_miou:.4f}")

    print("\nDone.")
    print(f"Best model saved at: {A.save_dir / 'best_model.pt'}")
    print(f"Last model saved at: {A.save_dir / 'last_model.pt'}")

if __name__ == "__main__":
    main()