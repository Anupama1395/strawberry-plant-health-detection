#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import (
    resnet18, resnet34, resnet50,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================================================
# ARGUMENTS
# =========================================================
def get_args():
    p = argparse.ArgumentParser("Semi supervised strawberry hierarchical classification")

    p.add_argument(
        "--coco_root",
        type=Path,
        default=Path("/home/ehr/Desktop/anupama/CPP_Strawberry_RGB_MS_2025-26/Unprocessed_data/all.v1i.coco"),
        help="COCO export root containing train/valid with _annotations.coco.json"
    )

    p.add_argument(
        "--labeled_root",
        type=Path,
        default=Path("/home/ehr/Desktop/anupama/plant/data_labeled_cls_from_coco"),
        help="Where converted train/val image classification dataset will be written"
    )

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
        "--ckpt_healthy",
        type=Path,
        default=Path("/home/ehr/Desktop/anupama/plant/training_runs/stage1_healthy_vs_rest_resnet18/best_model.pt")
    )
    p.add_argument(
        "--ckpt_modhealthy",
        type=Path,
        default=Path("/home/ehr/Desktop/anupama/plant/training_runs/stage1_modhealthy_vs_rest_efficientnet_b0/best_model.pt")
    )
    p.add_argument(
        "--ckpt_modunhealthy",
        type=Path,
        default=Path("/home/ehr/Desktop/anupama/plant/training_runs/stage1_modunhealthy_vs_rest_resnet50/best_model.pt")
    )
    p.add_argument(
        "--ckpt_unhealthy",
        type=Path,
        default=Path("/home/ehr/Desktop/anupama/plant/training_runs/stage1_unhealthy_vs_rest_resnet50/best_model.pt")
    )
    p.add_argument(
        "--ckpt_severelyunhealthy",
        type=Path,
        default=Path("/home/ehr/Desktop/anupama/plant/training_runs/stage1_severelyunhealthy_vs_rest_resnet50/best_model.pt")
    )

    p.add_argument(
        "--save_dir",
        type=Path,
        default=Path("/home/ehr/Desktop/anupama/plant/runs_semi_supervised_hierarchical"),
    )

    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size_labeled", type=int, default=16)
    p.add_argument("--batch_size_unlabeled", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--pseudo_threshold", type=float, default=0.80)
    p.add_argument("--lambda_u", type=float, default=0.5)

    p.add_argument("--student_model", choices=["resnet18", "resnet34", "resnet50", "efficientnet_b0"], default="resnet50")
    p.add_argument("--teacher_vote_threshold", type=float, default=0.50)

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
CLASS_TO_ID = {k: i for i, k in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

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


def canon(s: str) -> str:
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")


def load_rgb_image_safe(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as img:
            return np.array(img.convert("RGB"))
    except (OSError, UnidentifiedImageError, ValueError) as e:
        raise OSError(f"Failed to load image: {path} | {e}") from e


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


def build_model(model_name: str, num_classes: int, pretrained: bool = True):
    model_name = model_name.lower().strip()

    if model_name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "resnet34":
        model = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    raise ValueError(f"Unknown model_name: {model_name}")


def infer_model_name_from_ckpt(ckpt: dict) -> str:
    name = ckpt.get("model", None)
    if name is None:
        raise ValueError("Checkpoint does not contain model name")
    return str(name).lower().strip()


def get_teacher_eval_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

# =========================================================
# BUILD LABELED IMAGE LEVEL DATASET FROM COCO
# =========================================================
def build_labeled_classification_from_coco(coco_root: Path, out_root: Path):
    print(f"Building labeled classification dataset from COCO: {coco_root}")

    split_map = {
        "train": "train",
        "valid": "val",
        "val": "val",
    }

    built_count = 0

    for src_split, dst_split in split_map.items():
        ann_path = coco_root / src_split / "_annotations.coco.json"
        if not ann_path.exists():
            continue

        coco = json.loads(ann_path.read_text())
        images = {im["id"]: im for im in coco.get("images", [])}
        cats = {c["id"]: canon(c["name"]) for c in coco.get("categories", [])}

        print(f"\n[{src_split}] Categories found in COCO:")
        for cid, cname in cats.items():
            print(f"  id={cid} -> {cname}")

        anns_by_image = {}
        for a in coco.get("annotations", []):
            anns_by_image.setdefault(a["image_id"], []).append(a)

        out_img_dir = out_root / dst_split / "images"
        out_lbl_dir = out_root / dst_split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        split_written = 0

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

            anns = anns_by_image.get(image_id, [])
            if not anns:
                continue

            class_counts = {k: 0 for k in CLASS_NAMES}

            for ann in anns:
                raw_name = cats.get(ann["category_id"], "background")
                cname = canon(raw_name)

                if cname in CLASS_TO_ID:
                    class_counts[cname] += 1

            valid_total = sum(class_counts.values())
            if valid_total == 0:
                continue

            label_name = max(class_counts.items(), key=lambda kv: kv[1])[0]
            label_id = CLASS_TO_ID[label_name]

            dst_img = out_img_dir / Path(fname).name
            dst_lbl = out_lbl_dir / f"{Path(fname).stem}.txt"

            if not dst_img.exists():
                dst_img.write_bytes(src_img.read_bytes())
            dst_lbl.write_text(str(label_id))

            split_written += 1
            built_count += 1

        print(f"[{src_split}] Wrote {split_written} labeled classification samples")

    if built_count == 0:
        raise RuntimeError(
            "Could not build labeled classification dataset from COCO root. "
            "No usable class labels matched CLASS_NAMES."
        )

    print(f"\nFinished building labeled classification dataset at: {out_root}")
    print(f"Total classification samples written: {built_count}")

# =========================================================
# DATASETS
# =========================================================
class LabeledClsDataset(Dataset):
    def __init__(self, root: Path, split: str, img_size: int = 224, aug: bool = False):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.aug = aug

        self.img_dir = self.root / split / "images"
        self.lbl_dir = self.root / split / "labels"

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.color_jitter = transforms.ColorJitter(0.25, 0.25, 0.15, 0.05) if aug else None

        self.items = []
        for ip in sorted(self.img_dir.glob("*")):
            if ip.is_file() and is_image_file(ip):
                lp = self.lbl_dir / f"{ip.stem}.txt"
                if lp.exists():
                    self.items.append((ip, lp))

        if not self.items:
            raise RuntimeError(f"No labeled classification items found in {self.img_dir}")

        print(f"Labeled classification {split}: {len(self.items)} items")

    def __len__(self):
        return len(self.items)

    def _augment(self, img: np.ndarray):
        if self.aug:
            scale = np.random.uniform(0.8, 1.2)
            nh = max(16, int(img.shape[0] * scale))
            nw = max(16, int(img.shape[1] * scale))
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            img = random_crop_image(img, self.img_size)
            if random.random() < 0.5:
                img = img[:, ::-1].copy()
            if self.color_jitter is not None:
                img = np.array(self.color_jitter(Image.fromarray(img)))
        else:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        return img

    def __getitem__(self, idx):
        ip, lp = self.items[idx]
        img = load_rgb_image_safe(ip)
        img = self._augment(img)
        y = int(lp.read_text().strip())

        img_t = self.normalize(self.to_tensor(Image.fromarray(img)))
        return img_t, torch.tensor(y, dtype=torch.long)


class UnlabeledClsDataset(Dataset):
    def __init__(self, dirs, img_size: int = 224, max_retries: int = 20):
        self.img_size = img_size
        self.max_retries = max_retries
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.color_jitter = transforms.ColorJitter(0.25, 0.25, 0.15, 0.05)

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

    def _weak_aug(self, img: np.ndarray):
        img = random_crop_image(img, self.img_size)
        if random.random() < 0.5:
            img = img[:, ::-1].copy()
        return img

    def _strong_aug(self, img: np.ndarray):
        img = random_crop_image(img, self.img_size)
        if random.random() < 0.5:
            img = img[:, ::-1].copy()
        img = np.array(self.color_jitter(Image.fromarray(img)))
        return img

    def __getitem__(self, idx):
        for _ in range(self.max_retries):
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

        raise RuntimeError("Could not load a valid unlabeled image after retries")

# =========================================================
# HIERARCHICAL TEACHER
# =========================================================
class StageBinaryTeacher:
    def __init__(self, ckpt_path: Path, target_class_name: str, device: torch.device):
        self.ckpt_path = Path(ckpt_path)
        self.target_class_name = target_class_name
        self.device = device

        ckpt = torch.load(str(self.ckpt_path), map_location="cpu")
        model_name = infer_model_name_from_ckpt(ckpt)
        img_size = int(ckpt.get("img_size", 224))

        self.model_name = model_name
        self.img_size = img_size
        self.transform = get_teacher_eval_transform(self.img_size)

        self.model = build_model(model_name, num_classes=2, pretrained=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded teacher: {target_class_name} | {model_name} | img_size={img_size}")

    @torch.no_grad()
    def predict_target_probability(self, pil_img: Image.Image) -> float:
        x = self.transform(pil_img.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0]
        return float(probs[0].item())


class HierarchicalTeacherEnsemble:
    def __init__(self, device: torch.device):
        self.device = device
        self.teachers = {
            "healthy": StageBinaryTeacher(A.ckpt_healthy, "healthy", device),
            "moderately_healthy": StageBinaryTeacher(A.ckpt_modhealthy, "moderately_healthy", device),
            "moderately_unhealthy": StageBinaryTeacher(A.ckpt_modunhealthy, "moderately_unhealthy", device),
            "unhealthy": StageBinaryTeacher(A.ckpt_unhealthy, "unhealthy", device),
            "severely_unhealthy": StageBinaryTeacher(A.ckpt_severelyunhealthy, "severely_unhealthy", device),
        }

        self.priority = [
            "healthy",
            "moderately_healthy",
            "moderately_unhealthy",
            "unhealthy",
            "severely_unhealthy",
        ]

    @torch.no_grad()
    def predict_class(self, pil_img: Image.Image) -> Tuple[int, float, Dict[str, float]]:
        probs = {}
        for cname in self.priority:
            probs[cname] = self.teachers[cname].predict_target_probability(pil_img)

        for cname in self.priority:
            if probs[cname] >= A.teacher_vote_threshold:
                return CLASS_TO_ID[cname], float(probs[cname]), probs

        best_name = max(probs.items(), key=lambda kv: kv[1])[0]
        return CLASS_TO_ID[best_name], float(probs[best_name]), probs

# =========================================================
# STUDENT MODEL
# =========================================================
class StudentClassifier(nn.Module):
    def __init__(self, model_name: str = "resnet50", num_classes: int = NUM_CLASSES):
        super().__init__()
        self.model = build_model(model_name, num_classes=num_classes, pretrained=True)

    def forward(self, x):
        return self.model(x)

# =========================================================
# LOSSES AND METRICS
# =========================================================
def zero_loss_from_logits(logits):
    return logits.sum() * 0.0


def supervised_loss(logits, targets):
    return F.cross_entropy(logits, targets)


def pseudo_label_loss(student_logits, pseudo_targets, confidences, threshold=0.80):
    mask = confidences >= threshold
    if mask.sum() == 0:
        return zero_loss_from_logits(student_logits), 0.0

    ce = F.cross_entropy(student_logits, pseudo_targets, reduction="none")
    loss = ce[mask].mean()
    keep_ratio = float(mask.float().mean().item())
    return loss, keep_ratio


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)

    for x, y in tqdm(loader, desc="[Val]"):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True).long()

        logits = model(x)
        pred = logits.argmax(1)

        idx = (y * NUM_CLASSES + pred).to(torch.int64)
        conf += torch.bincount(idx.cpu(), minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)

    eps = 1e-12
    tp = torch.diag(conf).float()
    fp = conf.sum(0).float() - tp
    fn = conf.sum(1).float() - tp

    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    supp = conf.sum(1).float()

    present = supp > 0

    def m(v):
        return v[present].mean().item() if present.any() else float("nan")

    macro = {
        "mPrecision": m(prec),
        "mRecall": m(rec),
        "mF1": m(f1),
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
            "support": int(supp[i]),
        })

    return macro, per_class, conf.numpy()

# =========================================================
# SAVE
# =========================================================
def save_checkpoint(model, path: Path, epoch: int, metric: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "metric_mF1": metric,
        "model_state": model.state_dict(),
        "class_names": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "img_size": A.img_size,
        "student_model": A.student_model,
    }, str(path))

# =========================================================
# PSEUDO LABELS
# =========================================================
@torch.no_grad()
def get_hierarchical_pseudo_labels(teacher_ensemble: HierarchicalTeacherEnsemble, image_paths: List[str]):
    labels = []
    confs = []

    for p in image_paths:
        with Image.open(p) as pil_img:
            pil_img = pil_img.convert("RGB")
            cls_id, conf, _ = teacher_ensemble.predict_class(pil_img)
        labels.append(cls_id)
        confs.append(conf)

    return (
        torch.tensor(labels, dtype=torch.long, device=DEVICE),
        torch.tensor(confs, dtype=torch.float32, device=DEVICE),
    )

# =========================================================
# MAIN
# =========================================================
def main():
    if A.labeled_root.exists():
        print(f"Removing old labeled classification dataset: {A.labeled_root}")
        shutil.rmtree(A.labeled_root)

    build_labeled_classification_from_coco(A.coco_root, A.labeled_root)

    unlabeled_dirs = [s.strip() for s in A.unlabeled_dirs.split(",") if s.strip()]

    labeled_train = LabeledClsDataset(A.labeled_root, "train", img_size=A.img_size, aug=True)
    labeled_val = LabeledClsDataset(A.labeled_root, "val", img_size=A.img_size, aug=False)
    unlabeled_train = UnlabeledClsDataset(unlabeled_dirs, img_size=A.img_size)

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

    teacher_ensemble = HierarchicalTeacherEnsemble(DEVICE)
    student = StudentClassifier(model_name=A.student_model, num_classes=NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.AdamW(student.parameters(), lr=A.lr, weight_decay=A.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)

    best_mf1 = -1.0
    A.save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, A.epochs + 1):
        student.train()

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
                xu_w, xu_s, xu_paths = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(dl_u)
                xu_w, xu_s, xu_paths = next(unlabeled_iter)

            xl = xl.to(DEVICE, non_blocking=True)
            yl = yl.to(DEVICE, non_blocking=True).long()
            xu_s = xu_s.to(DEVICE, non_blocking=True)

            pseudo_targets, pseudo_confidences = get_hierarchical_pseudo_labels(teacher_ensemble, list(xu_paths))

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                logits_l = student(xl)
                loss_sup = supervised_loss(logits_l, yl)

                student_logits_u = student(xu_s)
                loss_unsup, keep_ratio = pseudo_label_loss(
                    student_logits_u,
                    pseudo_targets,
                    pseudo_confidences,
                    threshold=A.pseudo_threshold,
                )

                loss = loss_sup + A.lambda_u * loss_unsup

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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
        print(f"Validation mF1   : {macro['mF1']:.4f}")
        print(f"Validation mPrec : {macro['mPrecision']:.4f}")
        print(f"Validation mRec  : {macro['mRecall']:.4f}")

        for r in per_class:
            print(
                f"{r['class_name']:24s} "
                f"prec={r['precision']:.3f} "
                f"rec={r['recall']:.3f} "
                f"f1={r['f1']:.3f} "
                f"support={r['support']}"
            )

        save_checkpoint(student, A.save_dir / "last_model.pt", epoch, macro["mF1"])

        if macro["mF1"] > best_mf1:
            best_mf1 = macro["mF1"]
            save_checkpoint(student, A.save_dir / "best_model.pt", epoch, macro["mF1"])
            print(f"Saved new best model at epoch {epoch} with mF1={best_mf1:.4f}")

    print("\nDone.")
    print(f"Best model saved at: {A.save_dir / 'best_model.pt'}")
    print(f"Last model saved at: {A.save_dir / 'last_model.pt'}")

if __name__ == "__main__":
    main()