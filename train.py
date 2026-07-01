"""train.py — train a WBC classifier in one of two modes.

  * ``--mode holdout``  train on (phase1_train + phase2_train), validate on
    phase2_eval.  A single best checkpoint ``holdout_best.pt`` is saved.

  * ``--mode kfold``    stratified K-fold cross-validation over all labeled
    data (phase1_train + phase2_train + phase2_eval).  One checkpoint per fold
    ``fold{i}_best.pt`` is saved.

Everything a downstream script needs (model name, image size, per-channel
mean/std, class list, fold layout) is written to ``checkpoints/<name>/meta.json``.

Example
-------
    python train.py --mode holdout --model resnet50 --name res50_holdout --epochs 20
    python train.py --mode kfold --k 5 --model convnext_tiny --name cvx_k5 --epochs 15
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.amp import autocast, GradScaler

try:
    from tqdm import tqdm
except Exception:                       # tqdm optional
    def tqdm(x, **k):
        return x

from dataset import (
    CLASSES, NUM_CLASSES, IMAGENET_MEAN, IMAGENET_STD,
    set_seed, resolve_device, build_transforms, compute_channel_stats,
    make_loader, compute_class_weights, make_weighted_sampler,
    get_holdout_dfs, get_kfold_dfs, get_kfold_pool,
)
from model import build_model, save_checkpoint


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Train WBC classifier")
    p.add_argument("--data-root", default="data")
    p.add_argument("--mode", choices=["holdout", "kfold"], default="holdout")
    p.add_argument("--k", type=int, default=5, help="number of folds (kfold mode)")
    p.add_argument("--name", required=True, help="run name -> checkpoints/<name>/")
    p.add_argument("--ckpt-dir", default="checkpoints")

    p.add_argument("--model", default="resnet50")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--drop-rate", type=float, default=0.0)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=6,
                   help="early-stop patience (epochs w/o val-F1 improvement)")

    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--gpuid", type=int, default=0, help="GPU index cuda:0..7")
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA)")

    # imbalance handling
    p.add_argument("--class-weights", action="store_true",
                   help="inverse-frequency weighted CrossEntropy")
    p.add_argument("--sampler", choices=["none", "weighted"], default="none")

    # normalization stats
    p.add_argument("--stats", choices=["train", "imagenet"], default="train")
    p.add_argument("--stats-max-images", type=int, default=3000)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Train / eval loops
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp):
    model.train()
    running = 0.0
    for imgs, labels, _ in tqdm(loader, desc="train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item() * imgs.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running = 0.0
    preds, targets = [], []
    for imgs, labels, _ in tqdm(loader, desc="val", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        running += criterion(logits, labels).item() * imgs.size(0)
        preds.append(logits.argmax(1).cpu().numpy())
        targets.append(labels.cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    macro_f1 = f1_score(targets, preds, labels=list(range(NUM_CLASSES)),
                        average="macro", zero_division=0)
    return running / len(loader.dataset), macro_f1


def run_training(train_df, val_df, args, mean, std, device, ckpt_path, tag=""):
    """Train one model; save the best (val macro-F1) checkpoint. Returns best F1."""
    tfm_train = build_transforms(mean, std, args.img_size, train=True)
    tfm_eval = build_transforms(mean, std, args.img_size, train=False)

    sampler = make_weighted_sampler(train_df) if args.sampler == "weighted" else None
    train_loader = make_loader(train_df, tfm_train, args.batch_size,
                               shuffle=True, num_workers=args.num_workers,
                               has_labels=True, sampler=sampler)
    val_loader = make_loader(val_df, tfm_eval, args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             has_labels=True)

    model = build_model(args.model, num_classes=NUM_CLASSES,
                        pretrained=not args.no_pretrained,
                        drop_rate=args.drop_rate).to(device)

    weight = compute_class_weights(train_df).to(device) if args.class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weight,
                                    label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = args.amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    best_f1, best_epoch, patience = -1.0, -1, 0
    meta = {
        "model_name": args.model, "num_classes": NUM_CLASSES, "classes": CLASSES,
        "img_size": args.img_size, "mean": mean, "std": std,
        "mode": args.mode, "k": args.k, "seed": args.seed, "tag": tag,
    }

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                  device, scaler, use_amp)
        val_loss, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        dt = time.time() - t0
        marker = ""
        if val_f1 > best_f1:
            best_f1, best_epoch, patience = val_f1, epoch, 0
            save_checkpoint(ckpt_path, model, {**meta, "best_f1": best_f1,
                                               "best_epoch": epoch})
            marker = "  *best*"
        else:
            patience += 1
        print(f"[{tag}] epoch {epoch:02d}/{args.epochs} "
              f"| train_loss {tr_loss:.4f} | val_loss {val_loss:.4f} "
              f"| val_macroF1 {val_f1:.4f} | {dt:.0f}s{marker}")
        if args.patience > 0 and patience >= args.patience:
            print(f"[{tag}] early stop at epoch {epoch} "
                  f"(best macroF1 {best_f1:.4f} @ epoch {best_epoch})")
            break

    print(f"[{tag}] BEST val macro-F1 = {best_f1:.4f} (epoch {best_epoch}) "
          f"-> {ckpt_path}")
    return best_f1


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def resolve_stats(args, df):
    if args.stats == "imagenet":
        print("Using ImageNet normalization stats.")
        return list(IMAGENET_MEAN), list(IMAGENET_STD)
    print(f"Computing per-channel train stats on up to "
          f"{args.stats_max_images} images...")
    mean, std = compute_channel_stats(df, img_size=args.img_size,
                                      max_images=args.stats_max_images,
                                      seed=args.seed)
    print(f"  mean={np.round(mean, 4).tolist()}  std={np.round(std, 4).tolist()}")
    return mean, std


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device, args.gpuid)
    print(f"Device: {device} | mode: {args.mode} | model: {args.model}")

    run_dir = Path(args.ckpt_dir) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {"mode": args.mode, "model_name": args.model,
               "img_size": args.img_size, "num_classes": NUM_CLASSES,
               "classes": CLASSES, "seed": args.seed, "args": vars(args)}

    if args.mode == "holdout":
        train_df, val_df = get_holdout_dfs(args.data_root)
        print(f"train: {len(train_df)} | val (phase2_eval): {len(val_df)}")
        mean, std = resolve_stats(args, train_df)
        summary["mean"], summary["std"] = mean, std
        best_f1 = run_training(train_df, val_df, args, mean, std, device,
                               run_dir / "holdout_best.pt", tag="holdout")
        summary["val_f1"] = best_f1

    else:  # kfold
        # stats computed once over the whole labeled pool (stable across folds)
        pool = get_kfold_pool(args.data_root)
        mean, std = resolve_stats(args, pool)
        summary["mean"], summary["std"], summary["k"] = mean, std, args.k

        folds = get_kfold_dfs(args.data_root, k=args.k, seed=args.seed)
        fold_f1 = []
        for i, (tr_df, va_df) in enumerate(folds):
            print(f"\n===== Fold {i + 1}/{args.k} "
                  f"| train {len(tr_df)} | val {len(va_df)} =====")
            f1 = run_training(tr_df, va_df, args, mean, std, device,
                              run_dir / f"fold{i}_best.pt", tag=f"fold{i}")
            fold_f1.append(f1)
        summary["fold_f1"] = fold_f1
        summary["val_f1_mean"] = float(np.mean(fold_f1))
        summary["val_f1_std"] = float(np.std(fold_f1))
        print(f"\nCV macro-F1: {summary['val_f1_mean']:.4f} "
              f"± {summary['val_f1_std']:.4f}  (folds: "
              f"{[round(f, 4) for f in fold_f1]})")

    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved run metadata -> {run_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
