"""Dataset utilities for nail melanoma classification.

Two dataset classes participate in the training/evaluation pipeline:

- ``torchvision.datasets.ImageFolder`` (built-in) for the training set,
  which is organised into class subfolders (e.g., ``benign_train_aug/``,
  ``melanoma_train_aug/``).

- ``FlatImageDataset`` (defined here) for the test sets, which are flat
  folders of images with the class label encoded in the filename prefix
  (e.g., ``0_*.png`` -> label 0, ``1_*.png`` -> label 1).

``TransformedSubset`` wraps a ``torch.utils.data.Subset`` so that the
train and validation splits produced by ``StratifiedKFold`` can carry
*different* transforms even though they share the same underlying
``ImageFolder``.

``build_transforms`` and ``get_hf_normalization`` produce the (train,
val) transform pair used by every backbone, with normalisation
statistics adapted to each backbone's pretrained weights.
"""
from __future__ import annotations

import os
from typing import Callable, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Normalisation constants
# ---------------------------------------------------------------------------
# ImageNet-1k statistics used by torchvision-backed models (ResNet18,
# EfficientNet-b0) and Swin (whose HuggingFace processor exposes the
# same values).
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Subset wrapper for K-fold cross-validation
# ---------------------------------------------------------------------------
class TransformedSubset(Dataset):
    """Apply an independent transform to a ``torch.utils.data.Subset``.

    Why this exists:
        After ``StratifiedKFold`` splits an ``ImageFolder``, both the
        train and validation subsets reference the *same* underlying
        dataset object. Setting ``ImageFolder.transform`` would change
        both splits at once, so a per-subset wrapper is needed to give
        train and val different augmentation pipelines.
    """

    def __init__(self, subset: Subset, transform: Optional[Callable] = None) -> None:
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int):
        image, label = self.subset[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self) -> int:
        return len(self.subset)


# ---------------------------------------------------------------------------
# Flat-folder test dataset
# ---------------------------------------------------------------------------
class FlatImageDataset(Dataset):
    """A flat folder of images with labels encoded in the filename prefix.

    Filename convention
    -------------------
    ``<label>_<anything>.<ext>``  e.g. ``0_case_137.png``, ``1_NUM_22.jpg``

    Used for the test sets in this project (internal, external A,
    external B) which are stored as flat directories rather than
    class-subfolder layouts.

    Each item returns a 3-tuple ``(image_tensor, label, filename)``;
    the filename is preserved so per-image predictions can be saved
    to Excel/PKL for downstream analysis (e.g., misclassification
    review, Grad-CAM lookup).

    Args:
        folder_path: Directory containing the images.
        transform: Optional callable applied to each PIL image
            (typically ``build_transforms(...)[1]`` — the val/test
            transform without augmentation).
        valid_labels: Acceptable integer labels parsed from filenames.
            Raises ``ValueError`` if a filename prefix lies outside
            this set, which guards against silent label-parsing bugs.
        extensions: File extensions to include (case-insensitive).
    """

    def __init__(
        self,
        folder_path: str,
        transform: Optional[Callable] = None,
        valid_labels: Sequence[int] = (0, 1),
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
    ) -> None:
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"Not a directory: {folder_path}")

        self.folder_path = folder_path
        self.transform = transform
        self.valid_labels = set(valid_labels)

        # Sorted listing for deterministic ordering across OSes.
        candidates = sorted(
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
            and f.lower().endswith(extensions)
        )
        if not candidates:
            raise FileNotFoundError(
                f"No image files with extensions {extensions} found in {folder_path}"
            )

        # Validate the filename convention up front so any error surfaces
        # at construction time rather than during the first __getitem__.
        for f in candidates:
            label = self._parse_label(f)
            if label not in self.valid_labels:
                raise ValueError(
                    f"Filename '{f}' parses to label {label}, but "
                    f"valid_labels={sorted(self.valid_labels)}. "
                    f"Expected naming convention: '<label>_<anything>.<ext>'."
                )
        self.image_files = candidates

    @staticmethod
    def _parse_label(filename: str) -> int:
        """Extract the integer label that precedes the first '_' in a filename."""
        try:
            return int(filename.split("_")[0])
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Could not parse a label prefix from filename '{filename}'. "
                f"Expected '<int>_<rest>.<ext>'."
            ) from exc

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        filename = self.image_files[idx]
        path = os.path.join(self.folder_path, filename)
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self._parse_label(filename)
        return image, label, filename


# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------
def build_transforms(
    image_size: int = 224,
    image_mean: Tuple[float, float, float] = IMAGENET_MEAN,
    image_std: Tuple[float, float, float] = IMAGENET_STD,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Build the (train, val) transform pair shared by all backbones.

    Train transform (matches paper Methods S1 / S3):
        Resize → RandomHorizontalFlip → RandomVerticalFlip
            → ToTensor → Normalize(mean, std)

    Val/test transform (no augmentation):
        Resize → ToTensor → Normalize(mean, std)

    Args:
        image_size: Target square size for ``Resize``.
        image_mean: Per-channel normalisation mean (length-3 tuple).
        image_std: Per-channel normalisation std (length-3 tuple).

    Returns:
        ``(train_transform, val_transform)``.
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std),
        ]
    )
    return train_transform, val_transform


def get_hf_normalization(model_name: str) -> Tuple[Tuple[float, float, float],
                                                   Tuple[float, float, float],
                                                   int]:
    """Read normalisation statistics from a HuggingFace image processor.

    Used for ViT and Swin, whose pretrained weights expect model-specific
    normalisation (e.g., ViT uses (0.5, 0.5, 0.5) rather than ImageNet).
    Handles both dict-style ``size`` (``{"height": 224, "width": 224}``)
    and plain-integer ``size``.

    Args:
        model_name: HuggingFace model identifier, e.g.,
            ``"google/vit-base-patch16-224-in21k"``.

    Returns:
        ``(image_mean, image_std, image_size)``.
    """
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(model_name)
    image_mean = tuple(processor.image_mean)
    image_std = tuple(processor.image_std)

    size = processor.size
    if isinstance(size, dict):
        image_size = size.get("height", size.get("shortest_edge", 224))
    else:
        image_size = int(size)

    return image_mean, image_std, int(image_size)