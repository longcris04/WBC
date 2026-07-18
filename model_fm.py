"""model_fm.py — medical / white-blood-cell foundation-model encoders + a small
adaptive head, for the 13-class WBC task.

Design
------
Each entry in ``FM_ZOO`` is a *foundation-model encoder* loaded through timm's
hf-hub integration (weights are fetched + cached automatically).  The encoder is
created with ``num_classes=0`` (i.e. a pure feature extractor) and paired with a
small trainable **adaptive head** (linear or MLP) that maps the pooled embedding
to ``NUM_CLASSES`` (13).  The encoder itself may be fine-tuned or frozen
(linear-probe) via ``freeze_encoder``.

Foundation models
-----------------
* **DinoBloom** (S / B) — DINOv2 ViT pretrained on ~380k blood & bone-marrow
  *single-cell* images (Koch et al., 2024).  Directly on-domain for WBC.  (The
  Large / Giant variants are intentionally left out — too heavy for this bench.)
* **Phikon** — ViT-B/16 DINO pretrained on pan-cancer histopathology (Owkin).
* **Lunit-DINO** — ViT-S/16 DINO pretrained on histopathology (Lunit).

The DinoBloom checkpoints are DINOv2 patch-14 models whose native input is
518x518; we rebuild them at ``img_size`` (default 224, the size DinoBloom was
trained at) and let timm interpolate the position embeddings
(``dynamic_img_size=True``).

Checkpoint I/O mirrors :mod:`model`: the ``FMClassifier`` state + a ``meta`` dict
are saved together so ``test_fm.py`` / ``inference_fm.py`` can rebuild the exact
model and normalization without extra flags.  Prediction helpers
(``predict_probs`` / ``tta_probs``) are reused from :mod:`model`.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from dataset import NUM_CLASSES, IMAGENET_MEAN, IMAGENET_STD
# re-exported so downstream scripts can import prediction helpers from here too
from model import predict_probs, tta_probs  # noqa: F401


# --------------------------------------------------------------------------- #
# Foundation-model zoo
# --------------------------------------------------------------------------- #
# alias -> config
#   hf     : timm hf-hub identifier (weights auto-downloaded + cached)
#   dinov2 : True for DINOv2 patch-14 encoders (native 518px) -> rebuild at
#            img_size with position-embedding interpolation.
FM_ZOO = {
    # --- blood-cell foundation model (DinoBloom, DINOv2) ---
    "dinobloom_small": {"hf": "hf-hub:1aurent/vit_small_patch14_224.dinobloom", "dinov2": True},
    "dinobloom_base":  {"hf": "hf-hub:1aurent/vit_base_patch14_224.dinobloom",  "dinov2": True},
    # (dinobloom_large / dinobloom_giant omitted — too large for this benchmark)
    # --- general pathology / medical foundation models ---
    "phikon":     {"hf": "hf-hub:1aurent/vit_base_patch16_224.owkin_pancancer", "dinov2": False},
    "lunit_dino": {"hf": "hf-hub:1aurent/vit_small_patch16_224.lunit_dino",     "dinov2": False},
}


def list_fm_models():
    return list(FM_ZOO)


def _hf_name(name: str) -> str:
    cfg = FM_ZOO.get(name)
    return cfg["hf"] if cfg else name


# --------------------------------------------------------------------------- #
# Encoder + adaptive head
# --------------------------------------------------------------------------- #
def build_encoder(name: str, img_size: int = 224, pretrained: bool = True):
    """Create a foundation-model encoder (feature extractor, ``num_classes=0``).

    For DINOv2 patch-14 backbones we pass ``img_size`` + ``dynamic_img_size`` so
    the model accepts ``img_size`` inputs (position embeddings are interpolated).
    """
    import timm

    cfg = FM_ZOO.get(name, {})
    kwargs = dict(pretrained=pretrained, num_classes=0, img_size=img_size)
    if cfg.get("dinov2"):
        kwargs["dynamic_img_size"] = True
    return timm.create_model(_hf_name(name), **kwargs)


class FMClassifier(nn.Module):
    """Foundation-model encoder + a small adaptive classification head.

    Parameters
    ----------
    encoder        : timm model returning a pooled embedding (``num_classes=0``).
    feat_dim       : embedding dimension of ``encoder``.
    head           : ``"linear"`` (LayerNorm->Dropout->Linear) or ``"mlp"``
                     (LayerNorm->Linear->GELU->Dropout->Linear).
    freeze_encoder : if True, the encoder is frozen (kept in eval / no-grad) and
                     only the head is trained — a lightweight linear probe.
    """

    def __init__(self, encoder: nn.Module, feat_dim: int,
                 num_classes: int = NUM_CLASSES, head: str = "linear",
                 hidden_dim: int = 512, drop_rate: float = 0.0,
                 freeze_encoder: bool = False):
        super().__init__()
        self.encoder = encoder
        self.feat_dim = feat_dim
        self.head_type = head
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        if head == "mlp":
            self.head = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(hidden_dim, num_classes),
            )
        else:  # linear
            self.head = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Dropout(drop_rate),
                nn.Linear(feat_dim, num_classes),
            )

    def forward(self, x):
        if self.freeze_encoder:
            with torch.no_grad():
                feat = self.encoder(x)
        else:
            feat = self.encoder(x)
        return self.head(feat)

    def train(self, mode: bool = True):
        """Keep a frozen encoder in eval mode (no dropout / BN stat updates)."""
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)


def build_fm_model(name: str, num_classes: int = NUM_CLASSES,
                   pretrained: bool = True, img_size: int = 224,
                   head: str = "linear", hidden_dim: int = 512,
                   drop_rate: float = 0.0, freeze_encoder: bool = False
                   ) -> FMClassifier:
    """Build an encoder from ``FM_ZOO`` and attach the small adaptive head."""
    if name not in FM_ZOO:
        raise ValueError(
            f"Unknown FM '{name}'. Known: {list_fm_models()} "
            f"(or pass a timm hf-hub identifier prefixed with 'hf-hub:')."
        )
    encoder = build_encoder(name, img_size=img_size, pretrained=pretrained)
    feat_dim = encoder.num_features
    return FMClassifier(encoder, feat_dim, num_classes=num_classes, head=head,
                        hidden_dim=hidden_dim, drop_rate=drop_rate,
                        freeze_encoder=freeze_encoder)


# --------------------------------------------------------------------------- #
# Native normalization stats (read from the FM's own pretrained_cfg)
# --------------------------------------------------------------------------- #
def fm_norm_stats(name: str):
    """Return the encoder's native ``(mean, std)`` without building the model.

    Reads ``config.json`` from the hf-hub repo (``pretrained_cfg.mean/std``), so
    even the giant backbone is not instantiated just to resolve normalization.
    Falls back to ImageNet stats if anything is missing.
    """
    try:
        from huggingface_hub import hf_hub_download
        repo = _hf_name(name).replace("hf-hub:", "")
        cfg_path = hf_hub_download(repo, "config.json")
        cfg = json.loads(Path(cfg_path).read_text())
        pc = cfg.get("pretrained_cfg", cfg)
        mean = list(pc.get("mean", IMAGENET_MEAN))
        std = list(pc.get("std", IMAGENET_STD))
        return mean, std
    except Exception as e:  # offline / unexpected schema -> safe default
        print(f"[fm_norm_stats] falling back to ImageNet stats for '{name}': {e}")
        return list(IMAGENET_MEAN), list(IMAGENET_STD)


# --------------------------------------------------------------------------- #
# Checkpoint helpers (FM-aware; parallel to model.save/load_checkpoint)
# --------------------------------------------------------------------------- #
def save_fm_checkpoint(path, model: FMClassifier, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "meta": meta}, path)


def load_fm_checkpoint(path, map_location="cpu"):
    """Rebuild an ``FMClassifier`` from a checkpoint. Returns ``(model, meta)``."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    meta = ckpt["meta"]
    model = build_fm_model(
        meta["model_name"],
        num_classes=meta.get("num_classes", NUM_CLASSES),
        pretrained=False,                       # weights come from the state_dict
        img_size=meta.get("img_size", 224),
        head=meta.get("head", "linear"),
        hidden_dim=meta.get("hidden_dim", 512),
        drop_rate=0.0,
        freeze_encoder=meta.get("freeze_encoder", False),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, meta
