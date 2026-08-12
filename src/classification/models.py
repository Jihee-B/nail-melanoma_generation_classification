"""Classification model definitions for nail unit melanoma (NUM) vs.
nail melanonychia (NM).

Four backbones share a custom fully-connected (FC) head:

    Dropout(p) -> Linear(in_features, 256) -> Activation
               -> Dropout(p) -> Linear(256, num_classes)

The activation is chosen to match each backbone's native non-linearity
(paper Methods S3: "The activation function was selected to match the
respective backbone."):

    ResNeXt-50       -> ReLU
    EfficientNet-b0  -> SiLU (Swish)
    ViT-Base         -> GELU
    Swin-Base        -> GELU

Quick smoke test
----------------
$ python -m src.classification.models
"""
from __future__ import annotations

from typing import Dict, Type

import torch.nn as nn
from torchvision import models as tv_models
from efficientnet_pytorch import EfficientNet
from transformers import (
    SwinForImageClassification,
    ViTForImageClassification,
)


# ---------------------------------------------------------------------------
# Shared custom FC head
# ---------------------------------------------------------------------------
def build_custom_head(
    in_features: int,
    activation: nn.Module,
    dropout_rate: float = 0.3,
    hidden_dim: int = 256,
    num_classes: int = 2,
) -> nn.Sequential:
    """Construct the custom FC head used by every backbone.

    Args:
        in_features: Number of input features from the backbone.
        activation: An *instance* of an activation module
            (e.g., ``nn.ReLU()``, ``nn.GELU()``). Pass a fresh instance
            per model — do not share the same module across models.
        dropout_rate: Dropout probability applied before each linear layer.
        hidden_dim: Width of the intermediate linear layer.
        num_classes: Number of output classes.

    Returns:
        ``nn.Sequential`` implementing the custom head.
    """
    return nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, hidden_dim),
        activation,
        nn.Dropout(p=dropout_rate),
        nn.Linear(hidden_dim, num_classes),
    )


# ---------------------------------------------------------------------------
# Backbone-specific wrappers
# ---------------------------------------------------------------------------
class ResNeXt50Binary(nn.Module):
    """ResNeXt-50 (32x4d) with a custom FC head (ReLU activation).

    Backbone: ``torchvision.models.resnext50_32x4d`` initialised with the
    default ImageNet-1k pretrained weights (IMAGENET1K_V2).
    """

    def __init__(self, dropout_rate: float = 0.3, num_classes: int = 2) -> None:
        super().__init__()
        self.backbone = tv_models.resnext50_32x4d(
            weights=tv_models.ResNeXt50_32X4D_Weights.DEFAULT
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = build_custom_head(
            in_features=in_features,
            activation=nn.ReLU(inplace=True),
            dropout_rate=dropout_rate,
            num_classes=num_classes,
        )

    def forward(self, x):  # x: (B, 3, 224, 224)
        return self.backbone(x)


class EfficientNetB0Binary(nn.Module):
    """EfficientNet-b0 with a custom FC head (SiLU/Swish activation).

    Backbone: ``efficientnet_pytorch.EfficientNet`` ("efficientnet-b0",
    ImageNet pretrained).

    Note
    ----
    We use the third-party ``efficientnet_pytorch`` package (rather than
    ``torchvision.models.efficientnet_b0``) to keep the pretrained weights
    identical to those used during the paper's experiments. Migrating to
    the torchvision implementation would require re-training to maintain
    reproducibility.
    """

    def __init__(
        self,
        model_name: str = "efficientnet-b0",
        dropout_rate: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.backbone = EfficientNet.from_pretrained(model_name)
        in_features = self.backbone._fc.in_features
        self.backbone._fc = build_custom_head(
            in_features=in_features,
            activation=nn.SiLU(inplace=True),
            dropout_rate=dropout_rate,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.backbone(x)


class ViTBinary(nn.Module):
    """ViT-Base with a custom FC head (GELU activation).

    Backbone: HuggingFace ``ViTForImageClassification`` initialised from
    ``google/vit-base-patch16-224-in21k`` (ImageNet-21k pretrained).
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224-in21k",
        dropout_rate: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.backbone = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        in_features = self.backbone.config.hidden_size
        self.backbone.classifier = build_custom_head(
            in_features=in_features,
            activation=nn.GELU(),
            dropout_rate=dropout_rate,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.backbone(pixel_values=x).logits


class SwinBinary(nn.Module):
    """Swin-Base with a custom FC head (GELU activation).

    Backbone: HuggingFace ``SwinForImageClassification`` initialised from
    ``microsoft/swin-base-patch4-window7-224-in22k`` (ImageNet-22k pretrained).
    """

    def __init__(
        self,
        model_name: str = "microsoft/swin-base-patch4-window7-224-in22k",
        dropout_rate: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.backbone = SwinForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = build_custom_head(
            in_features=in_features,
            activation=nn.GELU(),
            dropout_rate=dropout_rate,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.backbone(pixel_values=x).logits


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "resnext50": ResNeXt50Binary,
    "efficientnet_b0": EfficientNetB0Binary,
    "vit": ViTBinary,
    "swin": SwinBinary,
}


def build_model(name: str, **kwargs) -> nn.Module:
    """Build a model by name (used by training/eval scripts via YAML config).

    Args:
        name: One of ``MODEL_REGISTRY`` keys
            ({'resnext50', 'efficientnet_b0', 'vit', 'swin'}).
        **kwargs: Forwarded to the chosen model's ``__init__``
            (e.g., ``dropout_rate``, ``num_classes``, or ``model_name``
            for HuggingFace-backed models).

    Returns:
        An initialised ``nn.Module``.

    Raises:
        KeyError: If ``name`` is not a registered backbone.
    """
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. "
            f"Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[key](**kwargs)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch

    dummy = torch.randn(2, 3, 224, 224)
    print("Building all backbones and running a forward pass on dummy input...\n")
    for name in MODEL_REGISTRY:
        model = build_model(name)
        model.eval()
        with torch.no_grad():
            out = model(dummy)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name:>18s}  output: {tuple(out.shape)}  "
              f"trainable params: {n_params:,}")