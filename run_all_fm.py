"""run_all_fm.py — full pipeline for a foundation-model run: train -> test ->
inference.

Mirrors run_all.py but drives the *_fm.py scripts and forwards the extra
foundation-model options (``--head``, ``--hidden-dim``, ``--freeze-encoder``,
``--encoder-lr``, ``--stats fm``).

Examples
--------
    # Holdout fine-tune of DinoBloom-S on GPU 2
    python run_all_fm.py --mode holdout --model dinobloom_small \
        --name dinobloom_small_holdout --epochs 15 --amp --device gpu --gpuid 2

    # 5-fold linear-probe of Phikon on GPU 3
    python run_all_fm.py --mode kfold --k 5 --model phikon \
        --name phikon_kfold --epochs 12 --amp --freeze-encoder --gpuid 3

    # Only (re)generate the submission from an already-trained run
    python run_all_fm.py --name dinobloom_small_holdout --skip-train --skip-test
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="FM Train -> Test -> Inference pipeline")
    # shared
    p.add_argument("--data-root", default="data")
    p.add_argument("--name", required=True)
    p.add_argument("--mode", choices=["holdout", "kfold"], default="holdout")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--model", default="dinobloom_small")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--gpuid", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--results-dir", default=".results")

    # foundation-model / head
    p.add_argument("--head", choices=["linear", "mlp"], default="linear")
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--drop-rate", type=float, default=0.0)
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--encoder-lr", type=float, default=None)

    # train-only
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--class-weights", action="store_true")
    p.add_argument("--sampler", choices=["none", "weighted"], default="none")
    p.add_argument("--stats", choices=["fm", "train", "imagenet"], default="fm")

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
    print(f"[run_all_fm] {stage}: {' '.join(cmd)}")
    print("=" * 70, flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)          # raises if the stage fails
    print(f"[run_all_fm] {stage} done in {time.time() - t0:.0f}s")


def main():
    a = parse_args()
    py = sys.executable
    here = str(Path(__file__).parent)
    common = ["--data-root", a.data_root, "--name", a.name,
              "--device", a.device, "--gpuid", str(a.gpuid),
              "--batch-size", str(a.batch_size), "--num-workers", str(a.num_workers)]

    # ---------------- train ----------------
    if not a.skip_train:
        cmd = [py, f"{here}/train_fm.py", "--mode", a.mode, "--k", str(a.k),
               "--model", a.model, "--img-size", str(a.img_size),
               "--head", a.head, "--hidden-dim", str(a.hidden_dim),
               "--drop-rate", str(a.drop_rate),
               "--epochs", str(a.epochs), "--lr", str(a.lr),
               "--weight-decay", str(a.weight_decay),
               "--label-smoothing", str(a.label_smoothing),
               "--patience", str(a.patience), "--seed", str(a.seed),
               "--sampler", a.sampler, "--stats", a.stats,
               "--ckpt-dir", a.ckpt_dir] + common
        if a.encoder_lr is not None:
            cmd += ["--encoder-lr", str(a.encoder_lr)]
        if a.amp:
            cmd.append("--amp")
        if a.no_pretrained:
            cmd.append("--no-pretrained")
        if a.class_weights:
            cmd.append("--class-weights")
        if a.freeze_encoder:
            cmd.append("--freeze-encoder")
        run("TRAIN", cmd)

    # ---------------- test -----------------
    if not a.skip_test:
        cmd = [py, f"{here}/test_fm.py", "--tta", a.tta,
               "--ckpt-dir", a.ckpt_dir, "--results-dir", a.results_dir] + common
        run("TEST", cmd)

    # -------------- inference --------------
    if not a.skip_infer:
        cmd = [py, f"{here}/inference_fm.py", "--tta", a.tta,
               "--ckpt-dir", a.ckpt_dir] + common
        if a.out:
            cmd += ["--out", a.out]
        run("INFERENCE", cmd)

    print("\n[run_all_fm] pipeline complete.")


if __name__ == "__main__":
    main()
