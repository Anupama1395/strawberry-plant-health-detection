#!/usr/bin/env python
# coding: utf-8
# plant_contrastive_seg.py
# Contrastive pretraining + binary segmentation fine-tuning.
# Robust mask AUTO-DETECTION (alpha/gray/palette/RGB), adaptive grids, hard audits.

import os, math, random
from collections import Counter
from pathlib import Path
from typing import Tuple, Dict, Optional, Callable, List

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import resnet50

try:
    from sklearn.metrics import average_precision_score
    _HAS_SK = True
except Exception:
    _HAS_SK = False


# ---------------- Config ----------------

DATA_ROOT = Path(os.environ.get("PLANT_DATA_ROOT", "/home/ehr/Desktop/anupama/plant/data/july_processed"))
# expects {train,val}/{images,masks}
IMG_SIZE     = 512
BATCH        = 4
EPOCHS_PRE   = 10
EPOCHS_FT    = 10
LR_PRE       = 1e-3
LR_FT        = 7e-4
TEMP         = 0.15
DOWNSIZE     = 64
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
THRESH_EVAL  = 0.5
PRINT_DIGITS = 6

# Optional hard override of mask rule. One of:
# 'alpha', 'value>0', 'value==0', 'value==255', 'rgb_nonblack', 'rgb_nonwhite'
FORCE_RULE = os.environ.get("PLANT_MASK_RULE", "").strip().lower()

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True


# ---------------- Mask reading with auto-detect ----------------

MaskRuleFn = Callable[[Image.Image], np.ndarray]
_DETECTED_RULE_NAME: Optional[str] = None
_READ_MASK_FN: Optional[MaskRuleFn] = None

def _alpha_rule(im: Image.Image) -> np.ndarray:
    # Foreground = alpha > 0 (RGBA or LA). If no alpha, returns all zeros.
    if im.mode in ("RGBA", "LA"):
        arr = np.array(im, dtype=np.uint8)
        return (arr[..., -1] > 0).astype(np.uint8)
    return np.zeros((im.height, im.width), np.uint8)

def _value_gt0_rule(im: Image.Image) -> np.ndarray:
    # Foreground = value > 0 (palette/grayscale), otherwise RGB any non-black
    if im.mode == "P":
        arr = np.array(im, dtype=np.uint8)
        return (arr > 0).astype(np.uint8)
    if im.mode == "L":
        arr = np.array(im, dtype=np.uint8)
        return (arr > 0).astype(np.uint8)
    rgb = np.array(im.convert("RGB"), dtype=np.uint8)
    return (rgb.sum(axis=2) > 0).astype(np.uint8)

def _value_eq0_rule(im: Image.Image) -> np.ndarray:
    # Foreground = value == 0 (fg drawn as 0, bg = 255)
    if im.mode == "P":
        arr = np.array(im, dtype=np.uint8)
        return (arr == 0).astype(np.uint8)
    if im.mode == "L":
        arr = np.array(im, dtype=np.uint8)
        return (arr == 0).astype(np.uint8)
    rgb = np.array(im.convert("RGB"), dtype=np.uint8)
    return (rgb.sum(axis=2) == 0).astype(np.uint8)

def _value_eq255_rule(im: Image.Image) -> np.ndarray:
    # Foreground = value == 255 (binary white foreground)
    if im.mode in ("P","L"):
        arr = np.array(im, dtype=np.uint8)
        return (arr == 255).astype(np.uint8)
    rgb = np.array(im.convert("RGB"), dtype=np.uint8)
    return ((rgb[...,0]==255)&(rgb[...,1]==255)&(rgb[...,2]==255)).astype(np.uint8)

def _rgb_nonblack_rule(im: Image.Image) -> np.ndarray:
    rgb = np.array(im.convert("RGB"), dtype=np.uint8)
    return (rgb.sum(axis=2) > 0).astype(np.uint8)

def _rgb_nonwhite_rule(im: Image.Image) -> np.ndarray:
    rgb = np.array(im.convert("RGB"), dtype=np.uint8)
    return (~((rgb[...,0]==255)&(rgb[...,1]==255)&(rgb[...,2]==255))).astype(np.uint8)

_RULE_CANDIDATES: List[Tuple[str, MaskRuleFn]] = [
    ("alpha",         _alpha_rule),
    ("value>0",       _value_gt0_rule),
    ("value==0",      _value_eq0_rule),
    ("value==255",    _value_eq255_rule),
    ("rgb_nonblack",  _rgb_nonblack_rule),
    ("rgb_nonwhite",  _rgb_nonwhite_rule),
]

def _summarize_mask_values(im: Image.Image) -> str:
    try:
        arr = np.array(im)
        if arr.ndim == 3 and arr.shape[2] in (3,4):
            g = arr.mean(axis=2).astype(np.uint8).reshape(-1)
            vals, cnts = np.unique(g, return_counts=True)
            top = sorted(zip(cnts, vals), reverse=True)[:5]
            return f"mode={im.mode} min={int(g.min())} max={int(g.max())} uniq(top5)={[ (int(v),int(c)) for c,v in top ]}"
        else:
            flat = arr.reshape(-1)
            vals, cnts = np.unique(flat, return_counts=True)
            top = sorted(zip(cnts, vals), reverse=True)[:5]
            return f"mode={im.mode} min={int(flat.min())} max={int(flat.max())} uniq(top5)={[ (int(v),int(c)) for c,v in top ]}"
    except Exception as e:
        return f"summary_error={e}"

def _auto_detect_mask_rule(split="train", sample_limit=24) -> Tuple[str, MaskRuleFn, int]:
    # If forced, honor it.
    global FORCE_RULE
    if FORCE_RULE:
        for name, fn in _RULE_CANDIDATES:
            if FORCE_RULE == name:
                print(f"[*] Forcing mask rule via PLANT_MASK_RULE={FORCE_RULE}")
                score = 0
                for p in sorted((DATA_ROOT / split / "masks").glob("*.png"))[:sample_limit]:
                    score += int(fn(Image.open(p)).sum())
                return name, fn, score
        print(f"[warn] Unknown PLANT_MASK_RULE={FORCE_RULE}; ignoring.")

    mdir = DATA_ROOT / split / "masks"
    files = sorted(mdir.glob("*.png"))
    if not files:
        raise RuntimeError(f"[auto] No masks found under {mdir}")

    files = files[:sample_limit]
    best_name = None; best_fn = None; best_fg = -1
    scores = []
    for name, fn in _RULE_CANDIDATES:
        total_fg = 0
        for p in files:
            im = Image.open(p)
            total_fg += int(fn(im).sum())
        scores.append((name, total_fg))
        if total_fg > best_fg:
            best_fg = total_fg
            best_name, best_fn = name, fn

    print("[auto] mask rule scores (split=%s): %s" %
          (split, ", ".join([f"{n}:{fg}" for n,fg in scores])))

    if best_fg <= 0:
        print("[auto] All candidate rules produced 0 foreground on samples. Diagnostics:")
        for p in files[:6]:
            im = Image.open(p)
            print("   ", p.name, "->", _summarize_mask_values(im))
        print("[auto] If files look blank, fix/export masks. Otherwise set PLANT_MASK_RULE explicitly.")
    return best_name, best_fn, best_fg

def init_mask_reader():
    global _DETECTED_RULE_NAME, _READ_MASK_FN
    n_tr, fn_tr, sc_tr = _auto_detect_mask_rule("train", sample_limit=24)
    n_va, fn_va, sc_va = _auto_detect_mask_rule("val",   sample_limit=24)
    if (sc_va or 0) > (sc_tr or 0):
        chosen = (n_va, fn_va, sc_va, "val")
    else:
        chosen = (n_tr, fn_tr, sc_tr, "train")
    _DETECTED_RULE_NAME, _READ_MASK_FN = chosen[0], chosen[1]
    print(f"[*] Using mask rule: {_DETECTED_RULE_NAME} (from {chosen[3]} with score {chosen[2]})")

def read_mask_binary_path(mp: Path) -> np.ndarray:
    im = Image.open(mp)
    binm = _READ_MASK_FN(im) if _READ_MASK_FN is not None else np.zeros((im.height, im.width), np.uint8)
    return binm.astype(np.uint8)


# ---------------- Dataset ----------------

class SegBinaryDataset(Dataset):
    def __init__(self, root: Path, split="train", img_size=512, aug_pretrain=False):
        assert _READ_MASK_FN is not None, "Call init_mask_reader() before building datasets."
        self.img_dir = root / split / "images"
        self.mask_dir = root / split / "masks"
        self.items = []
        for p in sorted(self.img_dir.glob("*")):
            m = self.mask_dir / f"{p.stem}.png"
            if m.exists():
                self.items.append((p, m))

        self.img_size = img_size
        self.aug_pretrain = aug_pretrain
        self.to_tensor = transforms.ToTensor()
        self.cj = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        ip, mp = self.items[idx]
        img = np.array(Image.open(ip).convert("RGB"))
        mask = read_mask_binary_path(mp)

        # light aug
        if self.aug_pretrain and random.random() < 0.5:
            img = img[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
        if self.aug_pretrain:
            img = np.array(self.cj(Image.fromarray(img)))

        # --- PIL-based resize (no OpenCV) ---
        img = np.array(Image.fromarray(img).resize((self.img_size, self.img_size), Image.BILINEAR))
        mask = np.array(Image.fromarray(mask).resize((self.img_size, self.img_size), Image.NEAREST))

        img = self.to_tensor(Image.fromarray(img))    # [3,H,W], float32 in [0,1]
        mask = torch.from_numpy(mask).float().unsqueeze(0)  # [1,H,W], {0,1}

        return img, mask


# ---------------- Encoder + Heads ----------------

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        m = resnet50(weights=None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.l1 = m.layer1
        self.l2 = m.layer2
        self.l3 = m.layer3
        self.l4 = m.layer4
        self.out_ch = 2048

    def forward(self, x):
        x = self.stem(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, in_ch, hid=256, out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hid, 1, bias=False),
            nn.BatchNorm2d(hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, out, 1, bias=False),
        )
    def forward(self, x):
        z = self.net(x)
        z = F.normalize(z, dim=1, p=2, eps=1e-8)
        return z

class SegmentationHead(nn.Module):
    def __init__(self, in_ch, out_ch=1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_ch, 1)
        )
    def forward(self, x):
        return self.head(x)


# ---------------- Utilities ----------------

def downsample_mask(mask: torch.Tensor, target_hw: Tuple[int,int]):
    _, _, H, W = mask.shape
    th, tw = target_hw
    kh = max(1, math.ceil(H / th))
    kw = max(1, math.ceil(W / tw))
    pooled = F.max_pool2d(mask, kernel_size=(kh, kw), stride=(kh, kw), ceil_mode=True)
    pooled = F.interpolate(pooled, size=(th, tw), mode="nearest")
    return (pooled[:, 0] > 0.5).long()

def supervised_contrastive_loss_within(z, y, T=0.07):
    B,C,h,w = z.shape
    z = z.permute(0,2,3,1).reshape(B, h*w, C)
    y = y.reshape(B, h*w)
    total_loss = 0.0; count = 0
    eye = torch.eye(h*w, device=z.device, dtype=torch.bool)
    for b in range(B):
        zb = z[b]; yb = y[b]
        fg = (yb == 1)
        if fg.sum() <= 1:
            continue
        sim = (zb @ zb.T) / T
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp_sim = torch.exp(sim)
        same = (fg.unsqueeze(1) & fg.unsqueeze(0)) & (~eye)
        denom = (exp_sim * (~eye)).sum(dim=1) + 1e-8
        pos_sum = (exp_sim * same).sum(dim=1) + 1e-8
        loss_i = -torch.log(pos_sum / denom)
        valid = (same.sum(dim=1) > 0) & fg
        if valid.any():
            total_loss += loss_i[valid].mean()
            count += 1
    return total_loss / max(count, 1)

def supervised_contrastive_loss_cross(zA, yA, zB, yB, T=0.07):
    B,C,ha,wa = zA.shape
    zA = zA.permute(0,2,3,1).reshape(B, ha*wa, C)
    zB = zB.permute(0,2,3,1).reshape(B, ha*wa, C)
    yA = yA.reshape(B, ha*wa)
    yB = yB.reshape(B, ha*wa)
    total = 0.0; count = 0
    eye = torch.eye(ha*wa, device=zA.device, dtype=torch.bool)
    for b in range(B):
        za, zb = zA[b], zB[b]; ya, yb = yA[b], yB[b]
        fgA = (ya == 1); fgB = (yb == 1)
        if fgA.sum() == 0 or (fgA.sum()+fgB.sum()) == 0:
            continue
        sim_aa = (za @ za.T) / T
        sim_ab = (za @ zb.T) / T
        m = torch.cat([sim_aa, sim_ab], dim=1).max(dim=1, keepdim=True).values.detach()
        sim_aa -= m; sim_ab -= m
        exp_aa = torch.exp(sim_aa); exp_ab = torch.exp(sim_ab)
        denom = (exp_aa * (~eye)).sum(dim=1) + exp_ab.sum(dim=1) + 1e-8
        pos_aa = (fgA.unsqueeze(1) & fgA.unsqueeze(0)) & (~eye)
        pos_ab = (fgA.unsqueeze(1) & fgB.unsqueeze(0))
        pos = (exp_aa * pos_aa).sum(dim=1) + (exp_ab * pos_ab).sum(dim=1) + 1e-8
        valid = (pos_aa.sum(dim=1) + pos_ab.sum(dim=1)) > 0
        if valid.any():
            total += (-torch.log(pos / denom))[valid].mean()
            count += 1
    return total / max(count, 1)

def dice_loss(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    num = 2 * (prob*target).sum(dim=(2,3)) + eps
    den = prob.sum(dim=(2,3)) + target.sum(dim=(2,3)) + eps
    return 1 - (num/den).mean()

def bin_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = np.logical_and(y_true==1, y_pred==1).sum()
    tn = np.logical_and(y_true==0, y_pred==0).sum()
    fp = np.logical_and(y_true==0, y_pred==1).sum()
    fn = np.logical_and(y_true==1, y_pred==0).sum()
    prec = float(tp / (tp+fp)) if (tp+fp)>0 else 0.0
    rec  = float(tp / (tp+fn)) if (tp+fn)>0 else 0.0
    f1   = float(2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0
    iou  = float(tp / (tp+fp+fn)) if (tp+fp+fn)>0 else 0.0
    acc  = float((tp+tn) / max(1, tp+tn+fp+fn))
    return {"precision":prec, "recall":rec, "dice":f1, "iou":iou, "accuracy":acc}


# ---------------- Data helpers ----------------

def sanity_check_data(root: Path):
    tr_img = list((root / "train" / "images").glob("*"))
    tr_msk = list((root / "train" / "masks").glob("*.png"))
    va_img = list((root / "val"   / "images").glob("*"))
    va_msk = list((root / "val"   / "masks").glob("*.png"))

    def stems(ps): return {p.stem for p in ps}
    tr_missing = sorted(stems(tr_img) - stems(tr_msk))[:10]
    va_missing = sorted(stems(va_img) - stems(va_msk))[:10]

    print("[sanity] DATA_ROOT =", root.resolve())
    print(f"[sanity] train images: {len(tr_img)} | train masks: {len(tr_msk)}")
    print(f"[sanity] val   images: {len(va_img)} | val   masks: {len(va_msk)}")
    if tr_missing:
        print("[sanity] (example) train images missing masks:", tr_missing)
    if va_missing:
        print("[sanity] (example) val images missing masks:", va_missing)

def _sanity_foreground(split="val", sample_limit: int = 40) -> float:
    mask_dir = DATA_ROOT / split / "masks"
    files = sorted(mask_dir.glob("*.png"))
    if not files:
        print(f"[sanity] no masks found under {mask_dir}")
        return 0.0
    if sample_limit > 0:
        files = files[:sample_limit]
    tot = 0; pos = 0; modes=[]
    for mp in files:
        im = Image.open(mp); modes.append(im.mode)
        b = read_mask_binary_path(mp)
        tot += b.size
        pos += int(b.sum())
    frac = pos / max(1, tot)
    print(f"[sanity] {split} foreground fraction (sampled {len(files)}): {frac:.6f} | modes={Counter(modes)}")
    return float(frac)

def audit_dataset_masks(split="val", max_show=8):
    mdir = DATA_ROOT / split / "masks"
    files = sorted(mdir.glob("*.png"))
    if not files:
        raise RuntimeError(f"[audit] No masks found under {mdir}")

    nz = []
    for p in files:
        fg = int(read_mask_binary_path(p).sum())
        nz.append((p.name, fg))
    total = len(nz)
    nonzero = sum(int(fg > 0) for _, fg in nz)
    print(f"[audit] split={split} files={total} with_FG>0={nonzero}")

    show_pos = [x for x in nz if x[1] > 0][:max_show]
    show_zero = [x for x in nz if x[1] == 0][:max_show]
    if show_pos:
        print("[audit] examples WITH FG:", show_pos)
    if show_zero:
        print("[audit] examples ZERO FG:", show_zero)

    if nonzero == 0:
        raise AssertionError(
            f"[audit] All masks in split '{split}' read as **zero foreground**. "
            f"Open a few of the listed files and confirm they’re not blank or mis-exported."
        )

def build_loaders(aug_pretrain=False):
    tr = SegBinaryDataset(DATA_ROOT, "train", IMG_SIZE, aug_pretrain=aug_pretrain)
    va = SegBinaryDataset(DATA_ROOT, "val",   IMG_SIZE, aug_pretrain=False)

    if len(tr) == 0:
        raise RuntimeError(
            f"No train samples found under {DATA_ROOT}/train/images & masks. "
            f"Check DATA_ROOT and that image/mask stems match."
        )
    if len(va) == 0:
        print("[warn] No val samples found—metrics will be empty.")

    pin = (DEVICE == "cuda")
    dl_tr = DataLoader(tr, batch_size=BATCH, shuffle=True,  num_workers=4, pin_memory=pin)
    dl_va = DataLoader(va, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=pin)
    return dl_tr, dl_va


# ---------------- Training + Evaluation ----------------

def pretrain_contrastive():
    dl_tr, _ = build_loaders(aug_pretrain=True)

    enc = Encoder().to(DEVICE)
    proj = ProjectionHead(enc.out_ch, hid=256, out=128).to(DEVICE)
    opt = torch.optim.AdamW(list(enc.parameters())+list(proj.parameters()), lr=LR_PRE, weight_decay=1e-4)

    skipped_batches = 0
    for epoch in range(1, EPOCHS_PRE+1):
        enc.train(); proj.train()
        pbar = tqdm(dl_tr, desc=f"[Pretrain {epoch}/{EPOCHS_PRE}]")
        loss_meter = 0.0; n=0
        for img, mask in pbar:
            img, mask = img.to(DEVICE), mask.to(DEVICE)

            with torch.no_grad():
                sizes_to_try = [(DOWNSIZE, DOWNSIZE), (96, 96), (128, 128)]
                grid_hw = None; y_small = None; valid_idx = None
                for (hh, ww) in sizes_to_try:
                    y_small_try = downsample_mask(mask, (hh, ww))
                    fg_per_img = (y_small_try == 1).sum(dim=(1,2))
                    valid_idx = (fg_per_img > 0).nonzero(as_tuple=True)[0]
                    if valid_idx.numel() > 0:
                        grid_hw = (hh, ww)
                        y_small = y_small_try[valid_idx]
                        break
                if (grid_hw is None) or (valid_idx is None) or (valid_idx.numel() == 0):
                    skipped_batches += 1
                    pbar.set_postfix(skipped=skipped_batches, loss=f"{(loss_meter/max(n,1)):.{PRINT_DIGITS}f}")
                    continue

            img = img[valid_idx]
            feat = enc(img)
            feat_small = F.interpolate(feat, size=grid_hw, mode="bilinear", align_corners=False)
            z = proj(feat_small)
            y = y_small

            if z.size(0) >= 2:
                if z.size(0) % 2 == 1:
                    zA, zB = z[:-1:2], z[1:-1:2]
                    yA, yB = y[:-1:2], y[1:-1:2]
                else:
                    zA, zB = z[::2], z[1::2]
                    yA, yB = y[::2], y[1::2]
                loss = supervised_contrastive_loss_cross(zA, yA, zB, yB, T=TEMP)
            else:
                loss = supervised_contrastive_loss_within(z, y, T=TEMP)

            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = img.size(0)
            loss_meter += loss.item()*bs; n += bs
            pbar.set_postfix(skipped=skipped_batches, loss=f"{loss_meter/max(n,1):.{PRINT_DIGITS}f}")

    torch.save(enc.state_dict(), "encoder_contrastive.pt")
    print("✓ Saved: encoder_contrastive.pt")

def evaluate_on_loader(enc: Encoder, seg: SegmentationHead, dl: DataLoader, thresh: float = THRESH_EVAL) -> Dict[str,float]:
    enc.eval(); seg.eval()
    mets = {"precision":[], "recall":[], "dice":[], "iou":[], "accuracy":[]}
    all_probs = []; all_gts = []
    with torch.no_grad():
        for img, mask in dl:
            img = img.to(DEVICE)
            logits_small = seg(enc(img))
            logits = F.interpolate(logits_small, size=mask.shape[-2:], mode="bilinear", align_corners=False)
            prob = torch.sigmoid(logits).cpu().numpy()
            pred = (prob > thresh).astype(np.uint8)

            gt_np = mask.cpu().numpy().astype(np.uint8)
            all_probs.append(prob.reshape(-1))
            all_gts.append(gt_np.reshape(-1))

            for i in range(mask.size(0)):
                yt = gt_np[i,0].astype(np.uint8).flatten()
                yp = pred[i,0].astype(np.uint8).flatten()
                m = bin_metrics(yt, yp)
                for k in mets: mets[k].append(m[k])

    out = {k: float(np.mean(v)) if v else 0.0 for k,v in mets.items()}
    if all_probs and all_gts and _HAS_SK:
        probs = np.concatenate(all_probs, axis=0)
        gts   = np.concatenate(all_gts,   axis=0)
        if (gts == 1).any():
            try: out["ap"] = float(average_precision_score(gts, probs))
            except Exception: out["ap"] = float("nan")
        else:
            out["ap"] = float("nan")
    else:
        out["ap"] = float("nan" if not _HAS_SK else 0.0)
    return out

def finetune_segmentation():
    _ = _sanity_foreground("train", sample_limit=40)
    val_fg_frac = _sanity_foreground("val", sample_limit=40)
    assert val_fg_frac > 0.0, (
        "Val masks contain no foreground pixels with the detected rule. "
        "Fix DATA_ROOT/val/masks or set PLANT_MASK_RULE explicitly."
    )

    dl_tr, dl_va = build_loaders(aug_pretrain=False)

    enc = Encoder().to(DEVICE)
    if Path("encoder_contrastive.pt").exists():
        enc.load_state_dict(torch.load("encoder_contrastive.pt", map_location=DEVICE))
        print("Loaded contrastively pretrained encoder.")
    seg = SegmentationHead(enc.out_ch, out_ch=1).to(DEVICE)

    params = list(enc.parameters()) + list(seg.parameters())
    opt = torch.optim.AdamW(params, lr=LR_FT, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    best = 1e9
    for epoch in range(1, EPOCHS_FT+1):
        enc.train(); seg.train()
        tr_loss=0; n=0
        for img, mask in tqdm(dl_tr, desc=f"[Finetune {epoch}/{EPOCHS_FT} train]"):
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            logits_small = seg(enc(img))
            logits = F.interpolate(logits_small, size=mask.shape[-2:], mode="bilinear", align_corners=False)
            loss = 0.5*bce(logits, mask) + 0.5*dice_loss(logits, mask)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item()*img.size(0); n += img.size(0)
        tr_loss /= max(n,1)

        enc.eval(); seg.eval()
        vl_loss=0; m=0
        with torch.no_grad():
            for img, mask in tqdm(dl_va, desc=f"[Finetune {epoch}/{EPOCHS_FT} val]"):
                img, mask = img.to(DEVICE), mask.to(DEVICE)
                logits = F.interpolate(seg(enc(img)), size=mask.shape[-2:], mode="bilinear", align_corners=False)
                loss = 0.5*bce(logits, mask) + 0.5*dice_loss(logits, mask)
                vl_loss += loss.item()*img.size(0); m += img.size(0)
        vl_loss /= max(m,1)

        if len(dl_va) > 0:
            val_metrics = evaluate_on_loader(enc, seg, dl_va, thresh=THRESH_EVAL)
            print(
                f"Epoch {epoch}: train={tr_loss:.4f} val_loss={vl_loss:.4f} | "
                f"val@{THRESH_EVAL:.2f} "
                f"P={val_metrics['precision']:.3f} "
                f"R={val_metrics['recall']:.3f} "
                f"D={val_metrics['dice']:.3f} "
                f"IoU={val_metrics['iou']:.3f} "
                f"Acc={val_metrics['accuracy']:.3f} | "
                f"AP={val_metrics['ap']:.3f}"
            )
        else:
            print(f"Epoch {epoch}: train={tr_loss:.4f} val_loss={vl_loss:.4f} | (no val set)")

        if vl_loss < best:
            best = vl_loss
            torch.save({"enc": enc.state_dict(), "seg": seg.state_dict()}, "plant_unet_contrastive.pt")
            print("✓ Saved: plant_unet_contrastive.pt")


# ---------------- Main ----------------

if __name__ == "__main__":
    sanity_check_data(DATA_ROOT)

    # 1) Auto-detect and lock mask rule (prints scores + diagnostics if needed)
    init_mask_reader()

    # 2) Quick peek with the chosen rule
    try:
        val_masks = sorted((DATA_ROOT / "val" / "masks").glob("*.png"))[:8]
        modes = [Image.open(p).mode for p in val_masks]
        fg_counts = [int(read_mask_binary_path(p).sum()) for p in val_masks]
        print("[peek] val mask modes:", Counter(modes))
        print("[peek] val sample FG pixel counts:", fg_counts)
    except Exception as _e:
        print("[peek] skip peek due to:", _e)

    # 3) Hard audit — fail fast if still empty with the detected rule
    audit_dataset_masks("train")
    audit_dataset_masks("val")

    # 4) Train
    pretrain_contrastive()
    finetune_segmentation()
