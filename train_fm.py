"""train_fm.py — train a foundation-model WBC classifier (encoder + small head).

Same two modes as train.py:

  * ``--mode holdout``  train on (phase1_train + phase2_train), validate on
    phase2_eval.  A single best checkpoint ``holdout_best.pt`` is saved.

  * ``--mode kfold``    stratified K-fold CV over all labeled data.  One
    checkpoint per fold ``fold{i}_best.pt`` is saved.

Differences vs the baseline train.py
-------------------------------------
  * The backbone is a medical / white-blood-cell **foundation model** (see
    ``model_fm.FM_ZOO``) used as an encoder, with a small trainable adaptive
    head on top (``--head {linear,mlp}``).
  * ``--freeze-encoder`` trains only the head (linear probe); otherwise the
    encoder is fine-tuned with a smaller LR (``--encoder-lr``, default lr/10).
  * ``--stats fm`` (default) uses the encoder's *native* normalization
    (mean/std from its pretrained config) instead of dataset statistics.

Everything a downstream script needs (model name, head config, image size,
mean/std, class list, fold layout) is written to
``checkpoints/<name>/meta.json``.

Example
-------
    python train_fm.py --mode holdout --model dinobloom_small \
        --name dinobloom_small_holdout --epochs 15 --amp
    python train_fm.py --mode kfold --k 5 --model phikon \
        --name phikon_kfold --epochs 12 --amp --freeze-encoder
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from model_fm import build_fm_model, save_fm_checkpoint, fm_norm_stats


# --------------------------------------------------------------------------- #
# Focal loss (parallels train.py:FocalLoss)
# --------------------------------------------------------------------------- #
class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017) with optional per-class weight.

        loss = weight[y] * (1 - p_y)**gamma * CE(logits, y)

    The per-class ``weight`` (e.g. log-inverse-frequency) plays the role of the
    focal *alpha* term, so class imbalance handling is preserved; ``gamma``
    down-weights easy, well-classified examples. ``label_smoothing`` is applied
    to the underlying cross-entropy exactly as in ``nn.CrossEntropyLoss``.
    """

    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.weight = weight            # already on the target device (or None)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight,
                             label_smoothing=self.label_smoothing,
                             reduction="none")
        logp = F.log_softmax(logits, dim=1)
        pt = logp.gather(1, target.unsqueeze(1)).squeeze(1).exp()  # p_true
        loss = (1.0 - pt) ** self.gamma * ce
        return loss.mean()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Train WBC foundation-model classifier")
    p.add_argument("--data-root", default="data")
    p.add_argument("--mode", choices=["holdout", "kfold"], default="holdout")
    p.add_argument("--k", type=int, default=5, help="number of folds (kfold mode)")
    p.add_argument("--name", required=True, help="run name -> checkpoints/<name>/")
    p.add_argument("--ckpt-dir", default="checkpoints")

    # --- foundation model / head ---
    p.add_argument("--model", default="dinobloom_small",
                   help="alias in model_fm.FM_ZOO (or an 'hf-hub:...' id)")
    p.add_argument("--no-pretrained", action="store_true",
                   help="do NOT load the foundation-model weights (debug only)")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--head", choices=["linear", "mlp"], default="linear")
    p.add_argument("--hidden-dim", type=int, default=512,
                   help="hidden width of the MLP head (ignored for linear)")
    p.add_argument("--drop-rate", type=float, default=0.0,
                   help="dropout in the adaptive head")
    p.add_argument("--freeze-encoder", action="store_true",
                   help="freeze encoder, train only the head (linear probe)")

    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4, help="head learning rate")
    p.add_argument("--encoder-lr", type=float, default=None,
                   help="encoder LR when fine-tuning (default: lr/10)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--loss", choices=["ce", "focal"], default="ce",
                   help="classification loss (ce = cross-entropy, focal = focal loss)")
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="focusing parameter gamma for --loss focal")
    p.add_argument("--patience", type=int, default=6,
                   help="early-stop patience (epochs w/o val-F1 improvement); 0 disables")

    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--gpuid", type=int, default=0, help="GPU index cuda:0..7")
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA)")

    # imbalance handling
    p.add_argument("--class-weights", action="store_true",
                   help="weighted loss (weighting set by --weight-scheme)")
    p.add_argument("--weight-scheme", choices=["inv", "sqrt", "log", "effnum"],
                   default="inv",
                   help="class-weight formula when --class-weights is set")
    p.add_argument("--sampler", choices=["none", "weighted"], default="none")

    # normalization stats: fm = encoder's native mean/std (recommended)
    p.add_argument("--stats", choices=["fm", "train", "imagenet"], default="fm")
    p.add_argument("--stats-max-images", type=int, default=3000)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Optimizer with separate encoder / head learning rates
# --------------------------------------------------------------------------- #
def build_optimizer(model, args):
    if args.freeze_encoder:
        return torch.optim.AdamW(
            [p for p in model.head.parameters()],
            lr=args.lr, weight_decay=args.weight_decay)
    enc_lr = args.encoder_lr if args.encoder_lr is not None else args.lr * 0.1
    groups = [
        {"params": list(model.encoder.parameters()), "lr": enc_lr},
        {"params": list(model.head.parameters()), "lr": args.lr},
    ]
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)


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
    """Train one FM model; save the best (val macro-F1) checkpoint. Returns best F1."""
    tfm_train = build_transforms(mean, std, args.img_size, train=True)
    tfm_eval = build_transforms(mean, std, args.img_size, train=False)

    sampler = make_weighted_sampler(train_df) if args.sampler == "weighted" else None
    train_loader = make_loader(train_df, tfm_train, args.batch_size,
                               shuffle=True, num_workers=args.num_workers,
                               has_labels=True, sampler=sampler)
    val_loader = make_loader(val_df, tfm_eval, args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             has_labels=True)

    model = build_fm_model(args.model, num_classes=NUM_CLASSES,
                           pretrained=not args.no_pretrained,
                           img_size=args.img_size, head=args.head,
                           hidden_dim=args.hidden_dim, drop_rate=args.drop_rate,
                           freeze_encoder=args.freeze_encoder).to(device)

    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] {args.model} | head={args.head} "
          f"| freeze_encoder={args.freeze_encoder} "
          f"| trainable {n_tr/1e6:.2f}M / {n_all/1e6:.2f}M params")

    weight = (compute_class_weights(train_df, scheme=args.weight_scheme).to(device)
              if args.class_weights else None)
    if args.loss == "focal":
        criterion = FocalLoss(weight=weight, gamma=args.focal_gamma,
                              label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=weight,
                                        label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = args.amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    best_f1, best_epoch, patience = -1.0, -1, 0
    meta = {
        "model_name": args.model, "num_classes": NUM_CLASSES, "classes": CLASSES,
        "img_size": args.img_size, "mean": mean, "std": std,
        "head": args.head, "hidden_dim": args.hidden_dim,
        "freeze_encoder": args.freeze_encoder,
        "mode": args.mode, "k": args.k, "seed": args.seed, "tag": tag,
        "is_fm": True,
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
            save_fm_checkpoint(ckpt_path, model, {**meta, "best_f1": best_f1,
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
    if args.stats == "fm":
        mean, std = fm_norm_stats(args.model)
        print(f"Using foundation-model native normalization: "
              f"mean={np.round(mean, 4).tolist()} std={np.round(std, 4).tolist()}")
        return mean, std
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
    print(f"Device: {device} | mode: {args.mode} | FM: {args.model}")

    run_dir = Path(args.ckpt_dir) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {"mode": args.mode, "model_name": args.model, "is_fm": True,
               "head": args.head, "hidden_dim": args.hidden_dim,
               "freeze_encoder": args.freeze_encoder,
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
