"""test_fm.py — evaluate a trained foundation-model run and log metrics.

Identical evaluation logic to test.py (macro-F1 emphasized, per-fold + OOF for
kfold, optional TTA); the only difference is that checkpoints are rebuilt with
``model_fm.load_fm_checkpoint`` (encoder + adaptive head).

Reads ``checkpoints/<name>/meta.json`` to recover mode, model, image size and
normalization, then writes a fresh report folder ``<results-dir>/<name>/``.

Example
-------
    python test_fm.py --name dinobloom_small_holdout
    python test_fm.py --name phikon_kfold --tta on
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score

from dataset import (
    CLASSES, CLASS_TO_IDX, NUM_CLASSES,
    set_seed, resolve_device, build_transforms, make_loader,
    get_holdout_dfs, get_kfold_dfs,
)
from eval_viz import save_confusion_matrix
from model_fm import load_fm_checkpoint, predict_probs


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained WBC FM run")
    p.add_argument("--data-root", default="data")
    p.add_argument("--name", required=True, help="run name (checkpoints/<name>/)")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--results-dir", default=".results")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--gpuid", type=int, default=0, help="GPU index cuda:0..7")
    p.add_argument("--tta", choices=["auto", "on", "off"], default="auto",
                   help="auto = on for kfold, off for holdout")
    return p.parse_args()


def resolve_tta(flag: str, mode: str) -> bool:
    if flag == "on":
        return True
    if flag == "off":
        return False
    return mode == "kfold"          # auto


def _report(y_true, y_pred):
    """Return (macro_f1, text_report, dict_report)."""
    labels = list(range(NUM_CLASSES))
    macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    txt = classification_report(y_true, y_pred, labels=labels,
                                target_names=CLASSES, digits=4, zero_division=0)
    dct = classification_report(y_true, y_pred, labels=labels,
                                target_names=CLASSES, output_dict=True,
                                zero_division=0)
    return macro, txt, dct


def evaluate_holdout(args, meta, device, use_tta, out_dir):
    _, val_df = get_holdout_dfs(args.data_root)
    tfm = build_transforms(meta["mean"], meta["std"], meta["img_size"], train=False)
    loader = make_loader(val_df, tfm, args.batch_size, shuffle=False,
                         num_workers=args.num_workers, has_labels=True)

    model, _ = load_fm_checkpoint(Path(args.ckpt_dir) / args.name / "holdout_best.pt",
                                  map_location=device)
    probs, ids = predict_probs(model, loader, device, tta=use_tta)
    id2true = dict(zip(val_df["ID"], val_df["labels"].map(CLASS_TO_IDX)))
    y_true = np.array([id2true[i] for i in ids])
    y_pred = probs.argmax(axis=1)

    macro, txt, dct = _report(y_true, y_pred)
    print(f"\n=== Holdout eval (phase2_eval, TTA={use_tta}) ===")
    print(f">>> Macro-averaged F1 = {macro:.4f} <<<\n")
    print(txt)

    (out_dir / "classification_report.txt").write_text(txt, encoding="utf-8")
    cm = save_confusion_matrix(
        y_true, y_pred, out_dir / "confusion_matrix.png",
        f"Confusion matrix — holdout (phase2_eval, TTA={use_tta})")
    return {"mode": "holdout", "tta": use_tta, "n_val": int(len(y_true)),
            "macro_f1": float(macro), "report": dct, "confusion_matrix": cm}


def evaluate_kfold(args, meta, device, use_tta, out_dir):
    k = meta["k"]
    folds = get_kfold_dfs(args.data_root, k=k, seed=meta["seed"])
    tfm = build_transforms(meta["mean"], meta["std"], meta["img_size"], train=False)

    fold_f1, fold_records = [], []
    oof_true, oof_pred = [], []
    for i, (_tr_df, va_df) in enumerate(folds):
        ckpt = Path(args.ckpt_dir) / args.name / f"fold{i}_best.pt"
        model, _ = load_fm_checkpoint(ckpt, map_location=device)
        loader = make_loader(va_df, tfm, args.batch_size, shuffle=False,
                             num_workers=args.num_workers, has_labels=True)
        probs, ids = predict_probs(model, loader, device, tta=use_tta)

        id2true = dict(zip(va_df["ID"], va_df["labels"].map(CLASS_TO_IDX)))
        y_true = np.array([id2true[i_] for i_ in ids])
        y_pred = probs.argmax(axis=1)      # argmax of TTA-averaged softmax

        macro, txt, dct = _report(y_true, y_pred)
        cm = save_confusion_matrix(
            y_true, y_pred, out_dir / f"fold{i}_confusion_matrix.png",
            f"Confusion matrix — fold {i} (TTA={use_tta})")
        fold_f1.append(float(macro))
        fold_records.append({"fold": i, "n_val": int(len(y_true)),
                             "macro_f1": float(macro), "report": dct,
                             "confusion_matrix": cm})
        oof_true.append(y_true)
        oof_pred.append(y_pred)
        print(f"[fold {i}] macro-F1 = {macro:.4f}  (n={len(y_true)})")
        (out_dir / f"fold{i}_report.txt").write_text(txt, encoding="utf-8")

    mean_f1, std_f1 = float(np.mean(fold_f1)), float(np.std(fold_f1))
    oof_true = np.concatenate(oof_true)
    oof_pred = np.concatenate(oof_pred)
    oof_macro, oof_txt, oof_dct = _report(oof_true, oof_pred)

    print(f"\n=== K-fold CV eval (k={k}, TTA={use_tta}) ===")
    print(f"per-fold macro-F1 : {[round(f, 4) for f in fold_f1]}")
    print(f">>> Mean macro-F1 over folds = {mean_f1:.4f} ± {std_f1:.4f} <<<")
    print(f">>> Out-of-fold (pooled) macro-F1 = {oof_macro:.4f} <<<\n")
    print(oof_txt)

    (out_dir / "oof_report.txt").write_text(oof_txt, encoding="utf-8")
    oof_cm = save_confusion_matrix(
        oof_true, oof_pred, out_dir / "oof_confusion_matrix.png",
        f"Confusion matrix — OOF pooled (k={k}, TTA={use_tta})")
    return {"mode": "kfold", "k": k, "tta": use_tta,
            "fold_macro_f1": fold_f1, "mean_macro_f1": mean_f1,
            "std_macro_f1": std_f1, "oof_macro_f1": float(oof_macro),
            "folds": fold_records, "oof_report": oof_dct,
            "oof_confusion_matrix": oof_cm}


def main():
    args = parse_args()
    meta_path = Path(args.ckpt_dir) / args.name / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Run metadata not found: {meta_path}. Train first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    set_seed(meta.get("seed", 42))
    device = resolve_device(args.device, args.gpuid)
    use_tta = resolve_tta(args.tta, meta["mode"])

    out_dir = Path(args.results_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | run: {args.name} | mode: {meta['mode']} "
          f"| TTA: {use_tta}\nLogging to: {out_dir}")

    if meta["mode"] == "holdout":
        result = evaluate_holdout(args, meta, device, use_tta, out_dir)
    else:
        result = evaluate_kfold(args, meta, device, use_tta, out_dir)

    result["run_name"] = args.name
    result["is_fm"] = True
    result["timestamp"] = datetime.now().isoformat(timespec="seconds")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
