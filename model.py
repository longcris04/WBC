"""model.py — baseline backbones (CNN / ViT / hybrid) for 13-class WBC.

All models are built through ``timm`` so the classifier head is automatically
re-mapped to ``num_classes`` (13).  A small curated zoo of well-known
architectures is exposed via short aliases; any raw ``timm`` model name also
works.

Checkpoint I/O keeps the model + a ``meta`` dict together so that test.py /
inference.py can rebuild the exact model and normalization without extra flags.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from dataset import NUM_CLASSES

# short alias -> timm model name
MODEL_ZOO = {
    # --- CNN ---
    "resnet18":        "resnet18",
    "resnet50":        "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b3": "efficientnet_b3",
    "convnext_tiny":   "convnext_tiny",
    "convnext_small":  "convnext_small",
    # --- Vision Transformers / hybrids ---
    "vit_tiny":        "vit_tiny_patch16_224",
    "vit_small":       "vit_small_patch16_224",
    "vit_base":        "vit_base_patch16_224",
    "deit_small":      "deit_small_patch16_224",
    "swin_tiny":       "swin_tiny_patch4_window7_224",
    "swin_small":      "swin_small_patch4_window7_224",
}

# models that require a fixed 224x224 input
FIXED_224 = {"vit_tiny", "vit_small", "vit_base", "deit_small",
             "swin_tiny", "swin_small"}


def list_models():
    return list(MODEL_ZOO)


def build_model(name: str, num_classes: int = NUM_CLASSES,
                pretrained: bool = True, drop_rate: float = 0.0) -> nn.Module:
    """Create a backbone with a fresh ``num_classes`` head."""
    import timm

    timm_name = MODEL_ZOO.get(name, name)
    if timm_name not in timm.list_models() and not timm.is_model(timm_name):
        raise ValueError(
            f"Unknown model '{name}'. Known aliases: {list_models()} "
            f"(or pass any valid timm model name)."
        )
    return timm.create_model(
        timm_name, pretrained=pretrained,
        num_classes=num_classes, drop_rate=drop_rate,
    )


# --------------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------------- #
def save_checkpoint(path, model: nn.Module, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path, map_location="cpu"):
    """Rebuild model from a checkpoint. Returns ``(model, meta)``."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    meta = ckpt["meta"]
    model = build_model(meta["model_name"],
                        num_classes=meta.get("num_classes", NUM_CLASSES),
                        pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, meta


# --------------------------------------------------------------------------- #
# Prediction helpers (shared by test.py / inference.py)
# --------------------------------------------------------------------------- #
def tta_probs(model: nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    """Test-time augmentation: average softmax over flips + 180° rotation."""
    views = [imgs,
             torch.flip(imgs, dims=[3]),      # horizontal flip
             torch.flip(imgs, dims=[2]),      # vertical flip
             torch.flip(imgs, dims=[2, 3])]   # 180 rotation
    prob = None
    for v in views:
        p = model(v).softmax(dim=1)
        prob = p if prob is None else prob + p
    return prob / len(views)


@torch.no_grad()
def predict_probs(model: nn.Module, loader, device, tta: bool = False):
    """Run inference over a loader. Returns ``(probs [N, C], ids [N])``."""
    model.eval().to(device)
    all_probs, all_ids = [], []
    for imgs, _labels, ids in loader:
        imgs = imgs.to(device, non_blocking=True)
        probs = tta_probs(model, imgs) if tta else model(imgs).softmax(dim=1)
        all_probs.append(probs.cpu())
        all_ids.extend(list(ids))
    return torch.cat(all_probs).numpy(), all_ids
