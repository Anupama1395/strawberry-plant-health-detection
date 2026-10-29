#!/usr/bin/env python3
# plot_loss.py
# Parses the training log (pretrain + finetune) and plots loss curves automatically.

import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys


def parse_log(log_path: Path):
    """Reads the log file and extracts loss values for each training phase."""
    pre_loss, ft_train, ft_val = {}, {}, {}

    # Regex for pretraining and finetuning logs
    rx_pre = re.compile(r"\[Pretrain\s+(\d+)\s*/\s*(\d+)\].*?loss=([0-9]*\.?[0-9]+)")
    rx_ft = re.compile(r"Epoch\s+(\d+):\s*train=([0-9]*\.?[0-9]+)\s*val=([0-9]*\.?[0-9]+)")

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if m := rx_pre.search(line):
                ep, loss = int(m.group(1)), float(m.group(3))
                pre_loss[ep] = loss
            elif m := rx_ft.search(line):
                ep, tr, vl = int(m.group(1)), float(m.group(2)), float(m.group(3))
                ft_train[ep], ft_val[ep] = tr, vl

    pre_ep = sorted(pre_loss.keys())
    ft_ep = sorted(ft_train.keys())
    return (
        (pre_ep, [pre_loss[e] for e in pre_ep]),
        (ft_ep, [ft_train[e] for e in ft_ep], [ft_val[e] for e in ft_ep])
    )


def plot_pretrain(xs, ys, out_path, show=False):
    if not xs:
        print("⚠️ No pretraining data found.")
        return
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, "o-", color="royalblue", label="Contrastive Loss")
    plt.title("Pretraining — Contrastive Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.grid(True, ls="--", lw=0.5); plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"✅ Saved pretraining plot to {out_path}")
    if show: plt.show()


def plot_finetune(xs, y_tr, y_vl, out_path, show=False):
    if not xs:
        print("⚠️ No finetuning data found.")
        return
    plt.figure(figsize=(8, 5))
    plt.plot(xs, y_tr, "o-", label="Train Loss", color="orange")
    plt.plot(xs, y_vl, "s--", label="Val Loss", color="seagreen")

    best_idx = int(np.argmin(y_vl))
    plt.scatter(xs[best_idx], y_vl[best_idx], s=120, c="red", marker="*", label=f"Best Val {y_vl[best_idx]:.4f}")
    plt.title("Finetuning — BCE + Dice Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.grid(True, ls="--", lw=0.5); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=150)
    print(f"✅ Saved finetuning plot to {out_path}")
    if show: plt.show()


def save_csv(out_dir, pre_data, ft_data):
    """Save numeric results to CSV."""
    csv_path = out_dir / "loss_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Phase", "Epoch", "Train_Loss", "Val_Loss"])
        pre_ep, pre_vals = pre_data
        for e, v in zip(pre_ep, pre_vals):
            w.writerow(["Pretrain", e, v, ""])
        ft_ep, ft_tr, ft_vl = ft_data
        for e, t, v in zip(ft_ep, ft_tr, ft_vl):
            w.writerow(["Finetune", e, t, v])
    print(f"📄 Saved summary CSV to {csv_path}")


def autodetect_log():
    """Finds the most recent .log or .txt file recursively."""
    logs = sorted(list(Path(".").rglob("*.log")) + list(Path(".").rglob("*.txt")),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        print("⚠️ No log files found in current directory.")
        dummy = Path("dummy_log.txt")
        dummy.write_text("[Pretrain 1/1] loss=0.5\nEpoch 1: train=0.4 val=0.35")
        print("🪄 Created dummy_log.txt for demo plotting.")
        return dummy
    if len(logs) == 1:
        print(f"📘 Using log: {logs[0]}")
        return logs[0]
    print("🗂 Found multiple logs:")
    for i, f in enumerate(logs, 1):
        print(f"  {i}) {f}")
    try:
        idx = int(input("Select log number to use: ").strip())
        return logs[idx - 1]
    except Exception:
        print("Invalid choice, defaulting to newest.")
        return logs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, default=None, help="Path to log file (auto-detects newest if omitted).")
    ap.add_argument("--out", type=str, default="plots", help="Output directory for plots.")
    ap.add_argument("--show", action="store_true", help="Display plots interactively.")
    args = ap.parse_args()

    log_path = Path(args.log) if args.log else autodetect_log()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔍 Reading log from: {log_path}")
    pre_data, ft_data = parse_log(log_path)

    pre_ep, pre_vals = pre_data
    ft_ep, ft_tr, ft_vl = ft_data

    if pre_ep: plot_pretrain(pre_ep, pre_vals, out_dir / "pretrain_loss.png", args.show)
    if ft_ep: plot_finetune(ft_ep, ft_tr, ft_vl, out_dir / "finetune_losses.png", args.show)
    save_csv(out_dir, pre_data, ft_data)


if __name__ == "__main__":
    main()
