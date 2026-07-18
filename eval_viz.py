"""eval_viz.py — evaluation visualisations shared by test.py / test_fm.py.

Currently a single helper: a 13-class confusion-matrix heatmap saved as a PNG.
Cells are COLOURED by row-normalized recall (so the diagonal stays comparable
across classes despite heavy imbalance — SNE has ~14k samples, PLY ~14) and each
non-zero cell is OVERLAID with its raw sample count.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from dataset import CLASSES, NUM_CLASSES


def save_confusion_matrix(y_true, y_pred, out_path, title):
    """Write a confusion-matrix heatmap to ``out_path``.

    Colour encodes the row-normalized value (per-class recall) on a single-hue
    ramp; every non-zero cell is annotated with its raw sample count. Returns the
    raw integer matrix as a nested list so callers can stash it in summary.json.
    Plotting failures never abort evaluation — the raw matrix is still returned.
    """
    out_path = Path(out_path)
    labels = list(range(NUM_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    try:
        import matplotlib
        matplotlib.use("Agg")            # headless: render to file, no display
        import matplotlib.pyplot as plt
    except Exception as e:               # never let plotting break evaluation
        print(f"[warn] matplotlib unavailable ({type(e).__name__}: {e}); "
              f"skipping {out_path.name}")
        return cm.tolist()

    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros(cm.shape, dtype=float),
                     where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(9.6, 8.4))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized (per-class recall)", fontsize=9)

    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CLASSES, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)

    # overlay raw counts; white ink on dark cells, dark ink on light cells
    for i in labels:
        for j in labels:
            n = int(cm[i, j])
            if n == 0:
                continue
            ax.text(j, i, f"{n:d}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "#0b0b0b")

    # thin white gridlines separating the cells
    ax.set_xticks(np.arange(-0.5, NUM_CLASSES, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, NUM_CLASSES, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix -> {out_path}")
    return cm.tolist()
