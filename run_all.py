"""run_all.py — run the full pipeline: train -> test -> inference.

Forwards a shared set of options to train.py / test.py / inference.py so a whole
experiment (both dataloader options) can be launched with a single command.

Examples
--------
    # Holdout on GPU 0
    python run_all.py --mode holdout --model resnet50 --name res50_holdout \
        --epochs 20 --amp --device gpu --gpuid 0

    # 5-fold cross-validation on GPU 3
    python run_all.py --mode kfold --k 5 --model convnext_tiny --name cvx_k5 \
        --epochs 15 --amp --device gpu --gpuid 3

    # Only (re)generate the submission from an already-trained run
    python run_all.py --name res50_holdout --skip-train --skip-test
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Train -> Test -> Inference pipeline")
    # shared
    p.add_argument("--data-root", default="data")
    p.add_argument("--name", required=True)
    p.add_argument("--mode", choices=["holdout", "kfold"], default="holdout")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--model", default="resnet50")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--gpuid", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--results-dir", default=".results")

    # train-only
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--class-weights", action="store_true")
    p.add_argument("--sampler", choices=["none", "weighted"], default="none")
    p.add_argument("--stats", choices=["train", "imagenet"], default="train")

    # test / inference
    p.add_argument("--tta", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--out", default=None, help="submission csv path")

    # stage control
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--skip-infer", action="store_true")
    return p.parse_args()


def run(stage: str, cmd: list[str]) -> None:
    print("\n" + "=" * 70)
    print(f"[run_all] {stage}: {' '.join(cmd)}")
    print("=" * 70, flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)          # raises if the stage fails
    print(f"[run_all] {stage} done in {time.time() - t0:.0f}s")


def main():
    a = parse_args()
    py = sys.executable
    here = str(Path(__file__).parent)
    common = ["--data-root", a.data_root, "--name", a.name,
              "--device", a.device, "--gpuid", str(a.gpuid),
              "--batch-size", str(a.batch_size), "--num-workers", str(a.num_workers)]

    # ---------------- train ----------------
    if not a.skip_train:
        cmd = [py, f"{here}/train.py", "--mode", a.mode, "--k", str(a.k),
               "--model", a.model, "--img-size", str(a.img_size),
               "--epochs", str(a.epochs), "--lr", str(a.lr),
               "--weight-decay", str(a.weight_decay),
               "--label-smoothing", str(a.label_smoothing),
               "--patience", str(a.patience), "--seed", str(a.seed),
               "--sampler", a.sampler, "--stats", a.stats,
               "--ckpt-dir", a.ckpt_dir] + common
        if a.amp:
            cmd.append("--amp")
        if a.no_pretrained:
            cmd.append("--no-pretrained")
        if a.class_weights:
            cmd.append("--class-weights")
        run("TRAIN", cmd)

    # ---------------- test -----------------
    if not a.skip_test:
        cmd = [py, f"{here}/test.py", "--tta", a.tta,
               "--ckpt-dir", a.ckpt_dir, "--results-dir", a.results_dir] + common
        run("TEST", cmd)

    # -------------- inference --------------
    if not a.skip_infer:
        cmd = [py, f"{here}/inference.py", "--tta", a.tta,
               "--ckpt-dir", a.ckpt_dir] + common
        if a.out:
            cmd += ["--out", a.out]
        run("INFERENCE", cmd)

    print("\n[run_all] pipeline complete.")


if __name__ == "__main__":
    main()
