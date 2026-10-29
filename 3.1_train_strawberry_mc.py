#!/usr/bin/env python3
"""
Train + validate a strawberry plant health classifier from tile folders.

Expected folder layout (both are supported):

Option A (your tile generator layout):
  /.../class_tiles/
    train/<class_name>/images/*.jpg
    val/<class_name>/images/*.jpg

Option B (flat class folders):
  /.../class_tiles/
    train/<class_name>/*.jpg
    val/<class_name>/*.jpg

This script:
1) Loads images
2) Trains a CNN (choose model: resnet18, resnet34, resnet50, efficientnet_b0)
3) Evaluates every epoch (train and val)
4) Prints per-epoch metrics and per-class accuracy
5) Supports imbalance handling: softened class weights and weighted sampler
6) Uses stronger augmentation and optional mixup
7) Uses scheduler and early stopping
8) Saves best model

Run example:
  source /home/ehr/Desktop/anupama/.venv/bin/activate
  python3 3.1_train_strawberry_mc.py --model resnet34 --use_weighted_sampler --epochs 30
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import (
    resnet18, resnet34, resnet50,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)

try:
    from sklearn.metrics import confusion_matrix, classification_report, f1_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


DEFAULT_TILES_ROOT = Path("/home/ehr/Desktop/anupama/plant/class_tiles")
DEFAULT_OUT_DIR = Path("/home/ehr/Desktop/anupama/plant/training_runs")

CLASS_NAMES = [
    "healthy",
    "moderately_healthy",
    "moderately_unhealthy",
    "unhealthy",
    "severely_unhealthy",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class TileFolderDataset(Dataset):
    """
    Supports:
      train/<class>/images/*.jpg   (preferred if present)
      train/<class>/*.jpg
    """
    def __init__(self, tiles_root: Path, split: str, transform=None):
        self.tiles_root = Path(tiles_root)
        self.split = split
        self.transform = transform

        split_dir = self.tiles_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split folder not found: {split_dir}")

        items: List[Tuple[Path, int]] = []

        for cls_name in CLASS_NAMES:
            base_dir = split_dir / cls_name
            if not base_dir.exists():
                print(f"WARNING: class folder missing: {base_dir}")
                continue

            # Your generator: .../<class>/images/*.jpg
            cls_dir = base_dir / "images" if (base_dir / "images").exists() else base_dir

            for img_path in sorted(cls_dir.rglob("*")):
                if img_path.is_file() and is_image_file(img_path):
                    items.append((img_path, CLASS_TO_ID[cls_name]))

        if len(items) == 0:
            raise RuntimeError(f"No images found for split '{split}' under: {split_dir}")

        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, label = self.items[idx]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            if self.transform is not None:
                im = self.transform(im)
        return im, label


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    macro_f1: float


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_split_exists(tiles_root: Path, split: str):
    split_dir = tiles_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


def get_class_counts(dataset: TileFolderDataset) -> np.ndarray:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for _, y in dataset.items:
        counts[y] += 1
    return counts


def compute_class_weights_softened(
    dataset: TileFolderDataset,
    weight_power: float = 0.5,
    max_weight: float = 20.0
) -> torch.Tensor:
    counts = get_class_counts(dataset)
    counts = np.maximum(counts, 1)
    total = float(counts.sum())
    base = total / counts.astype(np.float32)
    softened = np.power(base, weight_power)
    softened = np.minimum(softened, max_weight)
    return torch.tensor(softened, dtype=torch.float32)


def make_weighted_sampler_from_counts(dataset: TileFolderDataset) -> WeightedRandomSampler:
    counts = get_class_counts(dataset)
    counts = np.maximum(counts, 1)
    class_weights = 1.0 / counts.astype(np.float32)
    sample_weights = [class_weights[y] for _, y in dataset.items]
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
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
    raise ValueError("Unknown model. Use resnet18, resnet34, resnet50, efficientnet_b0")


def maybe_freeze_backbone(model: nn.Module, model_name: str, freeze: bool):
    if not freeze:
        return
    model_name = model_name.lower().strip()
    if model_name.startswith("resnet"):
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False
    elif model_name.startswith("efficientnet"):
        for name, p in model.named_parameters():
            if not name.startswith("classifier."):
                p.requires_grad = False


def unfreeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, float(lam)


def mixup_loss(criterion, logits, y_a, y_b, lam: float):
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    accs = []
    all_y = []
    all_p = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        losses.append(loss.item())
        accs.append(accuracy_from_logits(logits, y))

        preds = torch.argmax(logits, dim=1)
        all_y.append(y.detach().cpu().numpy())
        all_p.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(all_y, axis=0)
    y_pred = np.concatenate(all_p, axis=0)

    per_class_acc = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    for k in range(len(CLASS_NAMES)):
        mask = (y_true == k)
        per_class_acc[k] = (y_pred[mask] == k).mean() if mask.sum() > 0 else np.nan

    return float(np.mean(losses)), float(np.mean(accs)), y_true, y_pred, per_class_acc


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    use_mixup: bool,
    mixup_alpha: float,
    grad_clip: float,
    scheduler=None,
    scheduler_name: str = "none"
):
    model.train()
    losses = []
    accs = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_mixup:
            x_m, y_a, y_b, lam = mixup_batch(x, y, alpha=mixup_alpha)
            logits = model(x_m)
            loss = mixup_loss(criterion, logits, y_a, y_b, lam)
            preds = torch.argmax(logits, dim=1)
            acc = ((preds == y_a).float().mean().item() + (preds == y_b).float().mean().item()) * 0.5
        else:
            logits = model(x)
            loss = criterion(logits, y)
            acc = accuracy_from_logits(logits, y)

        loss.backward()

        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # OneCycleLR must step per batch
        if scheduler is not None and scheduler_name.lower() == "onecycle":
            scheduler.step()

        losses.append(loss.item())
        accs.append(acc)

    return float(np.mean(losses)), float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--tiles_root", type=str, default=str(DEFAULT_TILES_ROOT))
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))

    ap.add_argument("--model", type=str, default="resnet18",
                    help="resnet18, resnet34, resnet50, efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--use_weighted_sampler", action="store_true")

    # Make these toggles actually controllable
    ap.add_argument("--use_class_weights", action="store_true", default=False)
    ap.add_argument("--no_class_weights", action="store_true", help="Disable class weighted loss")

    ap.add_argument("--weight_power", type=float, default=0.5,
                    help="0.0 no reweight, 0.5 recommended, 1.0 strong")
    ap.add_argument("--max_weight", type=float, default=20.0)

    ap.add_argument("--label_smoothing", type=float, default=0.05)

    ap.add_argument("--pretrained", action="store_true", help="Use pretrained weights")
    ap.add_argument("--no_pretrained", action="store_true", help="Do not use pretrained weights")

    ap.add_argument("--freeze_epochs", type=int, default=2,
                    help="Freeze backbone for first N epochs, then unfreeze")
    
    ap.add_argument("--use_mixup", action="store_true")
    ap.add_argument("--mixup_alpha", type=float, default=0.2)

    ap.add_argument("--scheduler", type=str, default="onecycle",
                    help="onecycle or cosine or none")
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--save_best_by", type=str, default="macro_f1",
                    help="macro_f1 or val_acc")
    ap.add_argument("--early_stop_patience", type=int, default=6)

    args = ap.parse_args()

    # Resolve toggles
    use_class_weights = True
    if args.no_class_weights:
        use_class_weights = False
    elif args.use_class_weights:
        use_class_weights = True

    pretrained = True
    if args.no_pretrained:
        pretrained = False
    elif args.pretrained:
        pretrained = True

    tiles_root = Path(args.tiles_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_split_exists(tiles_root, "train")
    ensure_split_exists(tiles_root, "val")

    print("\n=== INPUT PATH CHECK ===")
    print(f"Tiles root: {tiles_root}")
    for split in ["train", "val"]:
        split_dir = tiles_root / split
        total = 0
        print(f"\nSplit: {split} -> {split_dir}")
        for c in CLASS_NAMES:
            cdir = split_dir / c
            imgdir = cdir / "images" if (cdir / "images").exists() else cdir
            n = len([p for p in imgdir.rglob("*") if p.is_file() and is_image_file(p)]) if imgdir.exists() else 0
            total += n
            print(f"  {c:20s}: {n}")
        print(f"  TOTAL: {total}")
        if total == 0:
            raise RuntimeError(f"No images found in split '{split}'.")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = TileFolderDataset(tiles_root=tiles_root, split="train", transform=train_tf)
    val_ds = TileFolderDataset(tiles_root=tiles_root, split="val", transform=val_tf)

    counts = get_class_counts(train_ds)
    print("\n=== CLASS COUNTS (TRAIN) ===")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:20s}: {int(counts[i])}")

    if use_class_weights:
        class_weights = compute_class_weights_softened(
            train_ds,
            weight_power=float(args.weight_power),
            max_weight=float(args.max_weight)
        )
        print("\n=== CLASS WEIGHTS (SOFTENED) ===")
        for i, name in enumerate(CLASS_NAMES):
            print(f"{name:20s}: {float(class_weights[i]):.6f}")
    else:
        class_weights = torch.ones(len(CLASS_NAMES), dtype=torch.float32)
        print("\n=== CLASS WEIGHTS ===")
        for i, name in enumerate(CLASS_NAMES):
            print(f"{name:20s}: 1.000000")

    if args.use_weighted_sampler:
        sampler = make_weighted_sampler_from_counts(train_ds)
        shuffle = False
        print("\nUsing WeightedRandomSampler for balanced batches.")
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args.model, num_classes=len(CLASS_NAMES), pretrained=pretrained)
    model = model.to(device)

    if args.freeze_epochs > 0:
        maybe_freeze_backbone(model, args.model, freeze=True)
        print(f"\nBackbone frozen for first {args.freeze_epochs} epoch(s).")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=float(args.label_smoothing)
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = None
    if args.scheduler.lower() == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.2
        )
    elif args.scheduler.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

    best_score = -1.0
    best_val_acc = -1.0
    best_macro_f1 = -1.0
    history: List[Dict] = []
    no_improve = 0

    print("\n=== TRAINING START ===")
    for epoch in range(1, args.epochs + 1):

        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            unfreeze_all(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            if args.scheduler.lower() == "onecycle":
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=args.lr,
                    epochs=args.epochs,
                    steps_per_epoch=len(train_loader),
                    pct_start=0.2
                )
            elif args.scheduler.lower() == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

            print("\nBackbone unfrozen. Training full model now.")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            use_mixup=bool(args.use_mixup),
            mixup_alpha=float(args.mixup_alpha),
            grad_clip=float(args.grad_clip),
            scheduler=scheduler,
            scheduler_name=args.scheduler
        )

        val_loss, val_acc, y_true, y_pred, per_class_acc = evaluate(model, val_loader, criterion, device)

        if SKLEARN_OK:
            macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
        else:
            macro_f1 = float("nan")

        # Step per-epoch schedulers here (NOT onecycle)
        if scheduler is not None and args.scheduler.lower() != "onecycle":
            scheduler.step()

        lr_now = float(optimizer.param_groups[0]["lr"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "macro_f1": macro_f1,
            "lr": lr_now,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} | "
            f"macro_f1 {macro_f1:.4f} | "
            f"lr {lr_now:.2e}"
        )

        print("\nPer class accuracy on val:")
        for i, name in enumerate(CLASS_NAMES):
            v = per_class_acc[i]
            if np.isnan(v):
                print(f"  {name:20s}: n/a")
            else:
                print(f"  {name:20s}: {v*100:.2f}%")

        if SKLEARN_OK:
            rep = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0)
            print(f"\nEpoch {epoch:03d} Classification Report:\n{rep}")

        score = val_acc if args.save_best_by.lower() == "val_acc" else (macro_f1 if SKLEARN_OK else val_acc)

        if score > best_score:
            best_score = score
            best_val_acc = max(best_val_acc, val_acc)
            best_macro_f1 = max(best_macro_f1, macro_f1 if SKLEARN_OK else -1.0)
            no_improve = 0

            best_path = out_dir / "best_model.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "img_size": args.img_size,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "macro_f1": macro_f1,
                    "model": args.model,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"  Saved best model -> {best_path}")

            if SKLEARN_OK:
                cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
                np.save(out_dir / "confusion_matrix.npy", cm.astype(np.int32))
                (out_dir / "classification_report.txt").write_text(rep)
        else:
            no_improve += 1

        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f"\nEarly stopping: no improvement for {args.early_stop_patience} epochs.")
            break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    print("\n=== TRAINING DONE ===")
    print(f"Best score: {best_score:.4f} (saved by {args.save_best_by})")
    print(f"Best val_acc observed: {best_val_acc:.4f}")
    if SKLEARN_OK:
        print(f"Best macro_f1 observed: {best_macro_f1:.4f}")
    print(f"Outputs saved to: {out_dir}")
    print("Files:")
    print(f"  - {out_dir/'best_model.pt'}")
    print(f"  - {out_dir/'history.json'}")
    if SKLEARN_OK:
        print(f"  - {out_dir/'classification_report.txt'}")
        print(f"  - {out_dir/'confusion_matrix.npy'}")


if __name__ == "__main__":
    main()