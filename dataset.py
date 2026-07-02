"""dataset.py — WBC data loading, transforms and split logic.

Two dataloader options (per project spec):

  * ``holdout``  (train WITHOUT cross-validation)
        train = phase1_train + phase2_train
        val   = phase2_eval                     <- used to score in test.py

  * ``kfold``    (train WITH cross-validation)
        pool  = phase1_train + phase2_train + phase2_eval   (all labeled data)
        pool  is split into K stratified folds; every fold -> (train, val).

Normalization: per-channel mean/std are computed on the TRAINING images and
applied as ``(x - mean) / std`` (albumentations.Normalize with
max_pixel_value=255).  The stats are saved into the run's ``meta.json`` so that
test.py / inference.py normalize identically.
"""
from __future__ import annotations

import contextlib
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold


@contextlib.contextmanager
def _suppress_c_stderr():
    """Silence libjpeg/OpenCV C-level warnings (e.g. 'Corrupt JPEG data:
    N extraneous bytes before marker ...') for slightly malformed but still
    decodable JPEGs. Only the C-level fd 2 is muted, briefly, around the decode
    call — Python-level stderr (tqdm, real tracebacks) is untouched."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def imread_color(path):
    """cv2.imread as BGR, without the noisy libjpeg stderr warnings."""
    with _suppress_c_stderr():
        return cv2.imread(path, cv2.IMREAD_COLOR)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CLASSES = ["BA", "BL", "BNE", "EO", "LY", "MMY", "MO",
           "MY", "PC", "PLY", "PMY", "SNE", "VLY"]          # 13 classes, sorted
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# split name -> (csv file, image sub-directory) relative to --data-root
SPLIT_CONFIG = {
    "phase1_train": ("phase1_label.csv", "phase1"),
    "phase2_train": ("phase2_train.csv", "phase2/train"),
    "phase2_eval":  ("phase2_eval.csv",  "phase2/eval"),
    "phase2_test":  ("phase2_test.csv",  "phase2/test"),
}
HOLDOUT_TRAIN_SPLITS = ["phase1_train", "phase2_train"]
HOLDOUT_VAL_SPLIT = "phase2_eval"
KFOLD_SPLITS = ["phase1_train", "phase2_train", "phase2_eval"]
TEST_SPLIT = "phase2_test"


# --------------------------------------------------------------------------- #
# Reproducibility / device helpers
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_device(device: str = "gpu", gpuid: int = 0) -> torch.device:
    """Resolve ``--device {cpu,gpu}`` + ``--gpuid`` into a torch.device.

    Falls back to CPU (with a warning) when CUDA is unavailable, and clamps an
    out-of-range gpuid to cuda:0.
    """
    if device == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        print("[warn] --device gpu requested but CUDA is unavailable -> using CPU.")
        return torch.device("cpu")
    n = torch.cuda.device_count()
    if gpuid < 0 or gpuid >= n:
        print(f"[warn] gpuid {gpuid} out of range (found {n} GPU(s)) -> using cuda:0.")
        gpuid = 0
    torch.cuda.set_device(gpuid)
    return torch.device(f"cuda:{gpuid}")


# --------------------------------------------------------------------------- #
# Metadata loading
# --------------------------------------------------------------------------- #
def _read_split_csv(data_root, split: str) -> pd.DataFrame:
    csv_name, img_sub = SPLIT_CONFIG[split]
    df = pd.read_csv(Path(data_root) / csv_name,
                     usecols=lambda c: c in ("ID", "labels"))
    if "labels" not in df.columns:
        df["labels"] = pd.NA
    lab = df["labels"].astype("string").str.strip()
    lab = lab.where(~lab.isin(["", "nan", "None", "NaN"]), other=pd.NA)
    df["labels"] = lab
    df["split"] = split
    img_dir = Path(data_root) / img_sub
    df["path"] = df["ID"].map(lambda x: str(img_dir / x))
    return df[["ID", "labels", "split", "path"]]


def load_metadata(data_root, splits) -> pd.DataFrame:
    """Merge the requested splits into one dataframe (ID, labels, split, path)."""
    frames = [_read_split_csv(data_root, s) for s in splits]
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Torch Dataset
# --------------------------------------------------------------------------- #
class WBCDataset(Dataset):
    """Reads an image with OpenCV, applies an albumentations transform.

    Returns ``(image_tensor, label_idx, image_id)``.  ``label_idx`` is ``-1``
    when the split is unlabeled (test set).
    """

    def __init__(self, df: pd.DataFrame, transform=None, has_labels: bool = True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.has_labels = has_labels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = imread_color(row["path"])
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {row['path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        label = CLASS_TO_IDX[row["labels"]] if self.has_labels else -1
        return img, label, row["ID"]


# --------------------------------------------------------------------------- #
# Transforms  (albumentations; Normalize = (x - mean) / std per channel)
# --------------------------------------------------------------------------- #
def build_transforms(mean, std, img_size: int = 224, train: bool = True):
    import albumentations as A                       # lazy import (optional dep)
    from albumentations.pytorch import ToTensorV2

    if train:
        # Cell images are rotation/flip invariant. Only ops that are stable
        # across albumentations 1.x and 2.x are used here (ShiftScaleRotate was
        # removed in 2.x); add A.Affine(...) yourself if you want scale/shift.
        ops = [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    else:
        ops = [
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    return A.Compose(ops)


# --------------------------------------------------------------------------- #
# Per-channel training statistics (mean/std in [0, 1])
# --------------------------------------------------------------------------- #
def compute_channel_stats(df: pd.DataFrame, img_size: int = 224,
                          max_images=3000, seed: int = 42):
    """Mean/std per RGB channel over (a sample of) the training images."""
    sub = df
    if max_images is not None and len(df) > max_images:
        sub = df.sample(max_images, random_state=seed)

    s = np.zeros(3, dtype=np.float64)
    sq = np.zeros(3, dtype=np.float64)
    count = 0
    for p in sub["path"]:
        img = imread_color(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size)).astype(np.float64) / 255.0
        flat = img.reshape(-1, 3)
        s += flat.sum(axis=0)
        sq += (flat ** 2).sum(axis=0)
        count += flat.shape[0]

    if count == 0:                                   # fallback
        return list(IMAGENET_MEAN), list(IMAGENET_STD)
    mean = s / count
    var = np.clip(sq / count - mean ** 2, 1e-8, None)
    return mean.tolist(), np.sqrt(var).tolist()


# --------------------------------------------------------------------------- #
# Class imbalance helpers
# --------------------------------------------------------------------------- #
def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """Inverse-frequency weights (normalized to mean 1) for CrossEntropyLoss."""
    counts = np.ones(NUM_CLASSES, dtype=np.float64)
    vc = df["labels"].map(CLASS_TO_IDX).value_counts()
    for idx, n in vc.items():
        counts[int(idx)] = n
    w = counts.sum() / (NUM_CLASSES * counts)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def make_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """Balanced sampling so rare classes appear as often as frequent ones."""
    y = df["labels"].map(CLASS_TO_IDX).values
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float64)
    counts[counts == 0] = 1.0
    sample_w = (1.0 / counts)[y]
    return WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                                 num_samples=len(sample_w), replacement=True)


# --------------------------------------------------------------------------- #
# DataLoader factory
# --------------------------------------------------------------------------- #
def make_loader(df, transform, batch_size, shuffle, num_workers,
                has_labels=True, sampler=None):
    ds = WBCDataset(df, transform, has_labels=has_labels)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


# --------------------------------------------------------------------------- #
# Split builders
# --------------------------------------------------------------------------- #
def get_holdout_dfs(data_root):
    """Option 1: (phase1_train + phase2_train) for train, phase2_eval for val."""
    df = load_metadata(data_root, HOLDOUT_TRAIN_SPLITS + [HOLDOUT_VAL_SPLIT])
    df = df.dropna(subset=["labels"])
    train_df = df[df["split"].isin(HOLDOUT_TRAIN_SPLITS)].reset_index(drop=True)
    val_df = df[df["split"] == HOLDOUT_VAL_SPLIT].reset_index(drop=True)
    return train_df, val_df


def get_kfold_pool(data_root):
    """The merged, sorted, labeled pool used for cross-validation."""
    df = load_metadata(data_root, KFOLD_SPLITS)
    df = df.dropna(subset=["labels"]).sort_values("ID").reset_index(drop=True)
    return df


def get_kfold_dfs(data_root, k=5, seed=42):
    """Deterministic stratified K folds -> list of (train_df, val_df).

    Sorting by ID + fixed seed guarantees train.py and test.py build the exact
    same folds, so each fold's validation set is well defined at test time.
    """
    df = get_kfold_pool(data_root)
    y = df["labels"].map(CLASS_TO_IDX).values
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    for tr_idx, va_idx in skf.split(df, y):
        folds.append((df.iloc[tr_idx].reset_index(drop=True),
                      df.iloc[va_idx].reset_index(drop=True)))
    return folds


def get_test_df(data_root):
    """Unlabeled test set (phase2_test), in original CSV order."""
    return load_metadata(data_root, [TEST_SPLIT]).reset_index(drop=True)
