#!/usr/bin/env python3
"""split_data.py — partition ./data into ./data_clean and ./data_noise by IMAGE QUALITY.

"Noise" here means *low-quality images*, judged from cheap, label-free pixel
statistics (the very features cached by EDA/eda.ipynb):

    * blur                 -> low  ``sharpness`` (variance of Laplacian)
    * blank / low-info     -> low  ``entropy``   (Shannon entropy of gray hist)
    * bad exposure         -> extreme ``brightness`` (too dark OR too bright)
    * corrupt / unreadable -> image fails to decode

Any image tripping one or more active gates is "noise"; everything else is "clean".

Both output trees mirror ``./data`` exactly — same sub-folders
(``phase1/``, ``phase2/{train,eval,test}``) and the same per-split CSVs
(``phase1_label.csv``, ``phase2_train.csv``, ``phase2_eval.csv``,
``phase2_test.csv``), each holding only the rows that landed in that group.
Images are COPIED by default (``--link-mode symlink|none`` to change that).
``data_noise/noise_report.csv`` records, for every flagged image, which rule(s)
fired plus its feature values, so the split is fully auditable.

Two ways to decide the thresholds:

  * QUANTILE mode (default) — data quantiles (``--q``, default 0.02 per tail)
    computed over the whole feature pool; tune with ``--gates`` and
    ``--set-lo/--set-hi``.
  * RULE mode — you state the thresholds explicitly, e.g.
    ``--rule entropy gt 7.5 --rule sharpness lt 10`` (noise where entropy > 7.5
    OR sharpness < 10). Any ``--rule`` turns the quantile gates off entirely.
    Rules may reference ANY feature in ALL_FEATURES.
  * THRESHOLD mode (``--threshold``) — same idea as RULE mode but the thresholds
    live in the hand-edited ``THRESHOLD`` dict at the top of this file
    (noise_below / noise_above per feature), so you edit numbers, not the command.

Features are read from the EDA cache when present (fast); any image missing from
the cache is decoded and measured on the fly with the exact same routine.

Run inside the ``STS`` conda env (needs cv2, numpy, pandas, tqdm).
"""
from __future__ import annotations

import argparse
import operator
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Data layout — split name -> (csv file, image sub-directory) under --data-root
# (kept in sync with dataset.py::SPLIT_CONFIG)
# --------------------------------------------------------------------------- #
SPLIT_CONFIG = {
    "phase1_train": ("phase1_label.csv", "phase1"),
    "phase2_train": ("phase2_train.csv", "phase2/train"),
    "phase2_eval":  ("phase2_eval.csv",  "phase2/eval"),
    "phase2_test":  ("phase2_test.csv",  "phase2/test"),
}
ALL_SPLITS = list(SPLIT_CONFIG)

# original column order per split (filled by _read_split_csv), so each output
# CSV mirrors its source exactly (e.g. phase2_test.csv has no ``split`` column)
ORIG_COLS: dict[str, list[str]] = {}

# every pixel feature extract_features() produces — any of these can be used in
# a --rule (e.g. ``--rule entropy gt 7.5`` for added-noise-looking images)
ALL_FEATURES = ["height", "width", "brightness", "contrast", "rms_contrast",
                "entropy", "sharpness", "colorfulness", "saturation",
                "mean_r", "mean_g", "mean_b", "edge_density"]

# comparison operators usable in --rule (word or symbol form; symbols need shell
# quoting, so the word forms are the safe default)
OPS = {"lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge,
       "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}
SYM = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=",
       "<": "<", "<=": "<=", ">": ">", ">=": ">="}

# feature -> gate direction for the DEFAULT quantile mode: which tail(s) count as
# "bad quality".  low = flag the lower tail, high = upper tail, both = either.
GATE_DIRECTION = {
    "sharpness":    "low",   # low  = blurry
    "entropy":      "low",   # low  = blank / little information
    "brightness":   "both",  # tails = under- / over-exposed
    "contrast":     "low",   # low  = flat / washed out
    "rms_contrast": "low",
    "colorfulness": "low",
    "saturation":   "both",
}
DEFAULT_GATES = ["sharpness", "entropy", "brightness"]

# =========================================================================== #
# NGƯỠNG THỦ CÔNG  —  chỉnh trực tiếp ở đây rồi chạy:  python split_data.py --threshold
# ---------------------------------------------------------------------------
# Một ảnh là NOISE nếu feature của nó rơi RA NGOÀI vùng "clean":
#     "noise_below": X   -> noise nếu  feature <  X    (None = không xét cận dưới)
#     "noise_above": Y   -> noise nếu  feature >  Y    (None = không xét cận trên)
# Đặt cả hai (vd sharpness) nghĩa là clean chỉ khi  X <= feature <= Y.
# Feature nào không liệt kê ở đây thì bỏ qua (ảnh corrupt luôn tính là noise).
# Đơn vị theo đúng thang trong EDA/eda_feature.ipynb (sharpness = var Laplacian,
# thang log: 10^1 = 10, 10^3 = 1000).
# =========================================================================== #
THRESHOLD = {
    "brightness": {"noise_below": 140,  "noise_above": None},   # noise nếu < 160
    "contrast":   {"noise_below": None, "noise_above": 80},     # noise nếu > 60
    "entropy":    {"noise_below": None, "noise_above": 8},      # noise nếu > 7
    "sharpness":  {"noise_below": 5,   "noise_above": 3000},   # noise nếu < 10 HOẶC > 1000
}


# --------------------------------------------------------------------------- #
# Low-level image features (identical to EDA/eda.ipynb::extract_features)
# --------------------------------------------------------------------------- #
def extract_features(path: str) -> dict | None:
    """Cheap, label-free pixel statistics for one image. ``None`` if unreadable."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)   # BGR
    if img is None:
        return None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img.astype(np.float32))

    mean_i = float(gray.mean())
    std_i = float(gray.std())

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    ent = float(-(p * np.log2(p)).sum())

    rg = r - g
    yb = 0.5 * (r + g) - b
    colorful = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) +
                     0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = float(hsv[:, :, 1].mean())

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float((edges > 0).mean())

    return dict(
        height=h, width=w,
        brightness=mean_i, contrast=std_i,
        rms_contrast=std_i / (mean_i + 1e-6),
        entropy=ent, sharpness=lap_var,
        colorfulness=colorful, saturation=sat,
        mean_r=float(r.mean()), mean_g=float(g.mean()), mean_b=float(b.mean()),
        edge_density=edge_density,
    )


# --------------------------------------------------------------------------- #
# Metadata + feature assembly
# --------------------------------------------------------------------------- #
def _read_split_csv(data_root: Path, split: str) -> pd.DataFrame:
    """Original per-split CSV, junk ``Unnamed:*`` columns dropped, + path/exists."""
    csv_name, img_sub = SPLIT_CONFIG[split]
    df = pd.read_csv(data_root / csv_name)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    ORIG_COLS[split] = list(df.columns)
    df["ID"] = df["ID"].astype(str)
    img_dir = data_root / img_sub
    df["grp_split"] = split
    df["grp_path"] = df["ID"].map(lambda x: str(img_dir / x))
    df["grp_exists"] = df["ID"].map(lambda x: (img_dir / x).is_file())
    return df


def load_features(data_root: Path, cache_path: Path | None, meta: pd.DataFrame,
                  workers: int) -> pd.DataFrame:
    """Feature table keyed by (grp_split, ID). Cache first, decode the rest."""
    cols = ALL_FEATURES
    have: dict[tuple[str, str], dict | None] = {}

    if cache_path and cache_path.is_file():
        cache = pd.read_csv(cache_path)
        cache["ID"] = cache["ID"].astype(str)
        keep = [c for c in cols if c in cache.columns]
        for row in cache.itertuples(index=False):
            have[(row.split, row.ID)] = {c: getattr(row, c) for c in keep}
        print(f"[feat] loaded cache {cache_path.name}: {len(cache)} rows")
    else:
        print("[feat] no feature cache — decoding every image (slower)")

    # Which (split, ID) still need features? (present on disk, absent from cache)
    todo = [(r.grp_split, r.ID, r.grp_path)
            for r in meta.itertuples(index=False)
            if r.grp_exists and (r.grp_split, r.ID) not in have]
    if todo:
        print(f"[feat] extracting features for {len(todo)} uncached image(s)...")

        def _work(job):
            split, _id, path = job
            return (split, _id), extract_features(path)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for key, feat in tqdm(ex.map(_work, todo), total=len(todo)):
                have[key] = feat  # None => corrupt

    recs = []
    for (split, _id), feat in have.items():
        rec = {"grp_split": split, "ID": _id, "grp_corrupt": feat is None}
        if feat is not None:
            for c in cols:
                rec[c] = feat.get(c, np.nan)
        recs.append(rec)
    return pd.DataFrame.from_records(recs)


# --------------------------------------------------------------------------- #
# Quality gating — everything reduces to a flat list of rules.
# A rule is (feature, op_name, threshold, label): the image is noise if
# ``OPS[op_name](feature_value, threshold)`` is True. Any rule matching -> noise.
# --------------------------------------------------------------------------- #
Rule = tuple[str, str, float, str]


def build_quantile_rules(feats: pd.DataFrame, gates: list[str], q: float,
                         overrides: dict[str, tuple[float | None, float | None]]
                         ) -> list[Rule]:
    """DEFAULT mode: turn each gate's tail into rules from pooled quantiles.

    ``low`` -> ``feature <= quantile(q)``; ``high`` -> ``feature >= quantile(1-q)``.
    ``--set-lo/--set-hi`` replace the computed threshold on that side.
    """
    rules: list[Rule] = []
    for f in gates:
        direction = GATE_DIRECTION[f]
        vals = feats[f].dropna().values if f in feats.columns else np.array([])
        lo = hi = None
        if direction in ("low", "both") and len(vals):
            lo = float(np.quantile(vals, q))
        if direction in ("high", "both") and len(vals):
            hi = float(np.quantile(vals, 1.0 - q))
        o_lo, o_hi = overrides.get(f, (None, None))
        if o_lo is not None:
            lo = o_lo
        if o_hi is not None:
            hi = o_hi
        if lo is not None:
            rules.append((f, "le", lo, f"{f}_low"))
        if hi is not None:
            rules.append((f, "ge", hi, f"{f}_high"))
    return rules


def parse_rules(raw: list[list[str]]) -> list[Rule]:
    """RULE mode: explicit ``--rule FEATURE OP VALUE`` triples -> rules.

    e.g. ``--rule entropy gt 7.5`` => noise where ``entropy > 7.5``.
    """
    rules: list[Rule] = []
    for feat, op, val in raw:
        if feat not in ALL_FEATURES:
            sys.exit(f"[error] --rule: unknown feature {feat!r}; "
                     f"choose from {ALL_FEATURES}")
        if op not in OPS:
            sys.exit(f"[error] --rule: unknown operator {op!r}; "
                     "use lt/le/gt/ge (or '<','<=','>','>=' quoted)")
        try:
            v = float(val)
        except ValueError:
            sys.exit(f"[error] --rule: threshold {val!r} for {feat} is not a number")
        rules.append((feat, op, v, f"{feat}{SYM[op]}{v:g}"))
    return rules


def rules_from_threshold(threshold: dict) -> list[Rule]:
    """THRESHOLD mode: turn the hand-edited THRESHOLD dict into rules.

    ``noise_below X`` -> ``feature < X``;  ``noise_above Y`` -> ``feature > Y``.
    """
    rules: list[Rule] = []
    for feat, band in threshold.items():
        if feat not in ALL_FEATURES:
            sys.exit(f"[error] THRESHOLD: unknown feature {feat!r}; "
                     f"choose from {ALL_FEATURES}")
        lo, hi = band.get("noise_below"), band.get("noise_above")
        if lo is not None:
            rules.append((feat, "lt", float(lo), f"{feat}<{float(lo):g}"))
        if hi is not None:
            rules.append((feat, "gt", float(hi), f"{feat}>{float(hi):g}"))
    if not rules:
        sys.exit("[error] THRESHOLD has no noise_below/noise_above set.")
    return rules


def flag_reasons(row: pd.Series, rules: list[Rule]) -> list[str]:
    """Rules that fire for one image (empty list => clean)."""
    if row["grp_corrupt"]:
        return ["corrupt"]
    reasons = []
    for feat, op, thresh, label in rules:
        v = row.get(feat, np.nan)
        if pd.isna(v):
            continue
        if OPS[op](v, thresh):
            reasons.append(label)
    return reasons


# --------------------------------------------------------------------------- #
# Output (mirror ./data into data_clean / data_noise)
# --------------------------------------------------------------------------- #
def place_image(src: str, dst: str, link_mode: str) -> None:
    d = Path(dst)
    d.parent.mkdir(parents=True, exist_ok=True)
    if d.exists() or d.is_symlink():
        d.unlink()
    if link_mode == "copy":
        shutil.copy2(src, dst)
    elif link_mode == "symlink":
        d.symlink_to(Path(src).resolve())
    # "none" -> CSVs only


def write_group(name: str, out_root: Path, meta: pd.DataFrame,
                is_noise: pd.Series, want_noise: bool,
                link_mode: str, workers: int) -> None:
    """Write one group's CSVs (mirroring each split) and place its images."""
    if out_root.exists():
        shutil.rmtree(out_root)   # start clean so a re-run leaves no stale files
    out_root.mkdir(parents=True, exist_ok=True)
    sel = meta[(is_noise == want_noise) & meta["grp_exists"]]

    jobs = []
    for split in meta["grp_split"].unique():
        csv_name, img_sub = SPLIT_CONFIG[split]
        sub = sel[sel["grp_split"] == split]
        out_cols = [c for c in ORIG_COLS[split] if c in sub.columns]
        sub[out_cols].to_csv(out_root / csv_name, index=False)
        for r in sub.itertuples(index=False):
            jobs.append((r.grp_path, str(out_root / img_sub / r.ID)))

    if link_mode != "none" and jobs:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(tqdm(ex.map(lambda j: place_image(j[0], j[1], link_mode), jobs),
                      total=len(jobs), desc=f"{name} images"))
    print(f"[{name}] {len(sel)} images  ->  {out_root}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data", type=Path)
    ap.add_argument("--clean-dir", default="data_clean", type=Path)
    ap.add_argument("--noise-dir", default="data_noise", type=Path)
    ap.add_argument("--feature-cache", type=Path,
                    default=Path("EDA/eda_cache/img_features_all_seed42.csv"),
                    help="EDA feature CSV; images missing from it are measured on the fly.")
    ap.add_argument("--splits", nargs="+", default=ALL_SPLITS, choices=ALL_SPLITS,
                    help="Which splits to process (default: all).")
    ap.add_argument("--gates", nargs="+", default=DEFAULT_GATES,
                    choices=list(GATE_DIRECTION),
                    help=f"Quantile mode: which gates to apply (default: {DEFAULT_GATES}).")
    ap.add_argument("--q", type=float, default=0.02,
                    help="Quantile mode: tail fraction per side (default 0.02).")
    ap.add_argument("--set-lo", nargs=2, action="append", default=[],
                    metavar=("FEATURE", "VALUE"),
                    help="Quantile mode: override a gate's lower threshold, "
                         "e.g. --set-lo sharpness 10")
    ap.add_argument("--set-hi", nargs=2, action="append", default=[],
                    metavar=("FEATURE", "VALUE"),
                    help="Quantile mode: override a gate's upper threshold, "
                         "e.g. --set-hi brightness 230")
    ap.add_argument("--rule", nargs=3, action="append", default=[],
                    metavar=("FEATURE", "OP", "VALUE"),
                    help="RULE mode (repeatable): flag as noise when FEATURE OP VALUE "
                         "holds. OP in lt/le/gt/ge. e.g. --rule entropy gt 7.5 "
                         "--rule sharpness lt 10. Any --rule switches off the quantile "
                         "gates entirely; an image is noise if ANY rule matches.")
    ap.add_argument("--threshold", action="store_true",
                    help="THRESHOLD mode: use the hand-edited THRESHOLD dict at the top "
                         "of this file (overrides --rule and the quantile gates).")
    ap.add_argument("--link-mode", choices=["copy", "symlink", "none"], default="copy",
                    help="How to place images in the output trees (default: copy).")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true",
                    help="Allow writing into non-empty output dirs.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report the split without writing anything.")
    args = ap.parse_args()

    for d in (args.clean_dir, args.noise_dir):
        if d.exists() and any(d.iterdir()) and not args.overwrite and not args.dry_run:
            sys.exit(f"[error] {d} exists and is not empty; pass --overwrite to reuse it.")

    overrides: dict[str, tuple[float | None, float | None]] = {}
    for f, v in args.set_lo:
        cur = overrides.get(f, (None, None))
        overrides[f] = (float(v), cur[1])
    for f, v in args.set_hi:
        cur = overrides.get(f, (None, None))
        overrides[f] = (cur[0], float(v))

    # --- metadata + features ------------------------------------------------ #
    meta = pd.concat([_read_split_csv(args.data_root, s) for s in args.splits],
                     ignore_index=True)
    n_missing = int((~meta["grp_exists"]).sum())
    if n_missing:
        print(f"[warn] {n_missing} image(s) listed in CSVs are missing on disk "
              "-> excluded from both groups.")

    feats = load_features(args.data_root, args.feature_cache, meta, args.workers)
    meta = meta.merge(feats, on=["grp_split", "ID"], how="left")
    # rows with no feature match (shouldn't happen for on-disk images) -> corrupt
    meta["grp_corrupt"] = meta["grp_corrupt"].fillna(True)

    # --- classify ----------------------------------------------------------- #
    if args.threshold:
        rules = rules_from_threshold(THRESHOLD)
        print("\n[rules]  (manual THRESHOLD mode — quantile gates ignored)")
    elif args.rule:
        rules = parse_rules(args.rule)
        print("\n[rules]  (rule mode — quantile gates ignored)")
    else:
        rules = build_quantile_rules(feats, args.gates, args.q, overrides)
        print(f"\n[rules]  (quantile mode, q={args.q})")
    if not rules:
        sys.exit("[error] no rules to apply (empty --gates and no --rule).")
    for feat, op, thresh, label in rules:
        print(f"  noise if {feat:<13} {SYM[op]:>2} {thresh:<12.4g}  -> {label}")

    reasons = meta.apply(lambda r: flag_reasons(r, rules), axis=1)
    meta["grp_reasons"] = reasons.map(lambda xs: "|".join(xs))
    is_noise = reasons.map(len) > 0

    # --- summary ------------------------------------------------------------ #
    present_mask = meta["grp_exists"]
    noise_present = is_noise & present_mask
    n = int(present_mask.sum())
    n_noise = int(noise_present.sum())
    print(f"\n[summary] {n} images present | clean {n - n_noise} "
          f"({100*(n-n_noise)/max(n,1):.1f}%) | noise {n_noise} "
          f"({100*n_noise/max(n,1):.1f}%)")
    print("  per split (clean / noise):")
    for s in args.splits:
        m = present_mask & (meta["grp_split"] == s)
        sn = int((noise_present & m).sum())
        print(f"    {s:<13} {int(m.sum()) - sn:>6} / {sn:<6}")
    rule_counts = (reasons[present_mask].explode().dropna()
                   .value_counts().to_dict())
    print("  by rule:", rule_counts or "{}")

    if args.dry_run:
        print("\n[dry-run] nothing written.")
        return

    # --- write both trees --------------------------------------------------- #
    print()
    write_group("clean", args.clean_dir, meta, is_noise, False,
                args.link_mode, args.workers)
    write_group("noise", args.noise_dir, meta, is_noise, True,
                args.link_mode, args.workers)

    # auditable report of every flagged image (feature cols = those the rules use)
    rule_feats = list(dict.fromkeys(feat for feat, *_ in rules))
    report_cols = (["ID", "grp_split", "labels", "grp_reasons"]
                   + [c for c in rule_feats if c in meta.columns])
    report_cols = [c for c in report_cols if c in meta.columns]
    rep = meta[noise_present][report_cols].rename(
        columns={"grp_split": "split", "grp_reasons": "reasons"})
    rep_path = args.noise_dir / "noise_report.csv"
    rep.to_csv(rep_path, index=False)
    print(f"[noise] wrote audit report -> {rep_path}  ({len(rep)} rows)")


if __name__ == "__main__":
    main()
