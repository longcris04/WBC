"""inference.py — predict phase2_test and write a Kaggle submission.

  * ``holdout``  single trained model -> argmax of softmax.

  * ``kfold``    the K fold models are combined by MAJORITY VOTING of their
    per-model argmax predictions.  Ties are broken by the highest summed
    probability across models.

The submission preserves the exact ID order/format of ``phase2_test.csv``
(columns ``ID,labels`` with ``labels`` filled by the predicted class code, e.g.
``SNE``).

Example
-------
    python inference.py --name res50_holdout --out submission_res50.csv
    python inference.py --name cvx_k5 --tta on --out submission_cvx_k5.csv
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import (
    IDX_TO_CLASS, NUM_CLASSES, TEST_SPLIT, SPLIT_CONFIG,
    set_seed, resolve_device, build_transforms, make_loader, get_test_df,
)
from model import load_checkpoint, predict_probs


def parse_args():
    p = argparse.ArgumentParser(description="Inference -> Kaggle submission")
    p.add_argument("--data-root", default="data")
    p.add_argument("--name", required=True, help="run name (checkpoints/<name>/)")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--gpuid", type=int, default=0, help="GPU index cuda:0..7")
    p.add_argument("--tta", choices=["auto", "on", "off"], default="auto",
                   help="auto = on for kfold, off for holdout")
    p.add_argument("--out", default=None, help="submission csv path")
    return p.parse_args()


def resolve_tta(flag: str, mode: str) -> bool:
    if flag == "on":
        return True
    if flag == "off":
        return False
    return mode == "kfold"


def majority_vote(votes_by_id, probsum_by_id, ids):
    """Hard majority vote per ID; tie-break by highest summed probability."""
    preds = {}
    for _id in ids:
        counts = np.bincount(votes_by_id[_id], minlength=NUM_CLASSES)
        top = counts.max()
        cand = np.where(counts == top)[0]
        if len(cand) == 1:
            preds[_id] = int(cand[0])
        else:
            preds[_id] = int(cand[np.argmax(probsum_by_id[_id][cand])])
    return preds


def infer_holdout(args, meta, device, use_tta, loader, test_ids):
    model, _ = load_checkpoint(Path(args.ckpt_dir) / args.name / "holdout_best.pt",
                               map_location=device)
    probs, ids = predict_probs(model, loader, device, tta=use_tta)
    id2prob = dict(zip(ids, probs))
    return {i: int(np.argmax(id2prob[i])) for i in test_ids}


def infer_kfold(args, meta, device, use_tta, loader, test_ids):
    k = meta["k"]
    votes = defaultdict(list)
    probsum = defaultdict(lambda: np.zeros(NUM_CLASSES, dtype=np.float64))
    for i in range(k):
        ckpt = Path(args.ckpt_dir) / args.name / f"fold{i}_best.pt"
        model, _ = load_checkpoint(ckpt, map_location=device)
        probs, ids = predict_probs(model, loader, device, tta=use_tta)
        for _id, p in zip(ids, probs):
            votes[_id].append(int(np.argmax(p)))
            probsum[_id] += p
        print(f"[fold {i}] predicted {len(ids)} test images")
    return majority_vote(votes, probsum, test_ids)


def main():
    args = parse_args()
    meta_path = Path(args.ckpt_dir) / args.name / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Run metadata not found: {meta_path}. Train first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    set_seed(meta.get("seed", 42))
    device = resolve_device(args.device, args.gpuid)
    use_tta = resolve_tta(args.tta, meta["mode"])
    print(f"Device: {device} | run: {args.name} | mode: {meta['mode']} "
          f"| TTA: {use_tta}")

    # test loader (unlabeled, deterministic order)
    test_df = get_test_df(args.data_root)
    tfm = build_transforms(meta["mean"], meta["std"], meta["img_size"], train=False)
    loader = make_loader(test_df, tfm, args.batch_size, shuffle=False,
                         num_workers=args.num_workers, has_labels=False)
    test_ids = list(test_df["ID"])

    if meta["mode"] == "holdout":
        preds = infer_holdout(args, meta, device, use_tta, loader, test_ids)
    else:
        preds = infer_kfold(args, meta, device, use_tta, loader, test_ids)

    # build submission in the ORIGINAL phase2_test.csv order/format
    test_csv = pd.read_csv(Path(args.data_root) / SPLIT_CONFIG[TEST_SPLIT][0],
                           usecols=lambda c: c in ("ID", "labels"))
    if "labels" not in test_csv.columns:
        test_csv["labels"] = ""
    test_csv["labels"] = test_csv["ID"].map(
        lambda i: IDX_TO_CLASS[preds[i]] if i in preds else "")
    missing = int((test_csv["labels"] == "").sum())
    if missing:
        print(f"WARNING: {missing} test IDs have no prediction!")

    out = args.out or f"submission_{args.name}.csv"
    test_csv[["ID", "labels"]].to_csv(out, index=False)
    print(f"\nWrote submission ({len(test_csv)} rows) -> {out}")
    print("Class distribution:\n",
          test_csv["labels"].value_counts().to_string())


if __name__ == "__main__":
    main()
