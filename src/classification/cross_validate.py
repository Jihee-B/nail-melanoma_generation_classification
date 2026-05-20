"""End-to-end 10-fold cross-validation for nail melanoma classification.

This is the main training entry point. It:

    1. Loads the training set (``ImageFolder``) and the test set
       (``FlatImageDataset``).
    2. Runs ``StratifiedKFold`` across the training set.
    3. For each fold, trains the configured backbone and saves the
       best-validation-accuracy checkpoint.
    4. After all folds, evaluates each checkpoint on the test set.
    5. Saves per-fold predictions (.pkl + .xlsx) and aggregated
       validation / test summaries (.xlsx).

Usage
-----
    python -m src.classification.cross_validate \\
        --model efficientnet_b0 \\
        --train-path /path/to/train_data \\
        --test-path  /path/to/test_data \\
        --output-dir runs/efficientnet_b0

Where:
    - --train-path points to an ImageFolder layout with class subfolders
      (e.g., benign/, melanoma/). Class names are assigned integer labels
      in alphabetical order by ImageFolder, so benign -> 0 and melanoma -> 1.
    - --test-path points to a flat folder whose filenames encode the label
      as a prefix before the first underscore (e.g., 0_xxx.png, 1_yyy.jpg).
    - --output-dir will hold checkpoints/, performance/, and config.json.

Per-backbone hyperparameter defaults live in ``DEFAULT_HPARAMS`` below.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from .dataset import (
    FlatImageDataset,
    TransformedSubset,
    build_transforms,
    get_hf_normalization,
)
from .lr_scheduler import LinearWarmupCosineAnnealingLR
from .models import build_model
from .train import evaluate_on_test, train_fold


# ---------------------------------------------------------------------------
# Per-backbone hyperparameter defaults
# (TODO: migrate to configs/classification/*.yaml in next milestone)
# ---------------------------------------------------------------------------
DEFAULT_HPARAMS: Dict[str, Dict[str, Any]] = {
    "resnet18": {
        "optimizer": "adam",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "eta_min": 1e-5,
        "use_hf_normalization": False,
    },
    "efficientnet_b0": {
        "optimizer": "adam",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "eta_min": 1e-5,
        "use_hf_normalization": False,
    },
    "vit": {
        "optimizer": "adamw",
        "lr": 5e-5,
        "weight_decay": 0.01,
        "eta_min": 1e-6,
        "use_hf_normalization": True,
        "hf_model_name": "google/vit-base-patch16-224-in21k",
    },
    "swin": {
        "optimizer": "adamw",
        "lr": 5e-5,
        "weight_decay": 0.05,
        "eta_min": 1e-7,
        "use_hf_normalization": True,
        "hf_model_name": "microsoft/swin-base-patch4-window7-224-in22k",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed Python / NumPy / PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(
    model: nn.Module,
    name: str,
    lr: float,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{name}'. Expected 'adam' or 'adamw'.")


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_excel(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)


def save_lr_curve(
    num_epochs: int,
    warmup_epochs: int,
    base_lr: float,
    eta_min: float,
    save_path: Path,
) -> None:
    """Plot the LR schedule by running a dummy optimiser through it.

    Useful as a sanity check that warmup duration and ``eta_min`` match
    the configured hyperparameters.
    """
    dummy = torch.nn.Linear(1, 1).cpu()
    opt = Adam(dummy.parameters(), lr=base_lr)
    sched = LinearWarmupCosineAnnealingLR(
        opt,
        warmup_epochs=warmup_epochs,
        max_epochs=num_epochs,
        warmup_start_lr=0.0,
        eta_min=eta_min,
    )
    lrs = [sched.get_last_lr()[0]]
    for _ in range(num_epochs - 1):
        opt.step()
        sched.step()
        lrs.append(sched.get_last_lr()[0])

    plt.figure(figsize=(8, 4))
    plt.plot(lrs, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title("Warmup + Cosine Annealing LR Schedule")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_cross_validation(args: argparse.Namespace) -> None:
    # --- Resolve hyperparameters (defaults + CLI overrides) ---
    hp = dict(DEFAULT_HPARAMS[args.model])
    if args.lr is not None:
        hp["lr"] = args.lr
    if args.weight_decay is not None:
        hp["weight_decay"] = args.weight_decay
    if args.eta_min is not None:
        hp["eta_min"] = args.eta_min

    # --- Output layout: <output_dir>/{checkpoints, performance} ---
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    perf_dir = output_dir / "performance"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    perf_dir.mkdir(parents=True, exist_ok=True)

    # --- Save run config snapshot for reproducibility ---
    config_snapshot = {
        "model": args.model,
        "train_path": args.train_path,
        "test_path": args.test_path,
        "num_epochs": args.epochs,
        "patience": args.patience,
        "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "device": args.device,
        **hp,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2)
    print(f"Run config snapshot: {output_dir / 'config.json'}\n")

    # --- Seed + device ---
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Transforms (HF-aware for ViT/Swin) ---
    if hp.get("use_hf_normalization", False):
        mean, std, image_size = get_hf_normalization(hp["hf_model_name"])
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        image_size = 224
    print(f"Image normalisation: mean={mean}  std={std}  size={image_size}")
    train_transform, val_transform = build_transforms(
        image_size=image_size, image_mean=mean, image_std=std
    )

    # --- Datasets ---
    full_dataset = ImageFolder(root=args.train_path, transform=None)
    test_dataset = FlatImageDataset(
        folder_path=args.test_path, transform=val_transform
    )
    print(f"Train/val size: {len(full_dataset)}")
    print(f"Test size:      {len(test_dataset)}")
    print(f"Classes:        {full_dataset.classes}")
    print(f"Label mapping:  {full_dataset.class_to_idx}\n")

    # --- LR schedule preview ---
    save_lr_curve(
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        base_lr=hp["lr"],
        eta_min=hp["eta_min"],
        save_path=perf_dir / "lr_schedule_curve.png",
    )

    # --- K-fold cross-validation ---
    skf = StratifiedKFold(
        n_splits=args.n_splits, shuffle=True, random_state=args.seed
    )
    targets = [label for _, label in full_dataset.samples]
    fold_summaries = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(targets)), targets)
    ):
        print(f"\n{'=' * 60}\nFold {fold_idx + 1}/{args.n_splits}\n{'=' * 60}")

        train_ds = TransformedSubset(
            Subset(full_dataset, train_idx), train_transform
        )
        val_ds = TransformedSubset(Subset(full_dataset, val_idx), val_transform)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        model = build_model(args.model).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = build_optimizer(
            model,
            name=hp["optimizer"],
            lr=hp["lr"],
            weight_decay=hp["weight_decay"],
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            max_epochs=args.epochs,
            warmup_start_lr=0.0,
            eta_min=hp["eta_min"],
        )

        result = train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=args.epochs,
            patience=args.patience,
            fold_idx=fold_idx,
        )

        ckpt_path = ckpt_dir / f"fold_{fold_idx + 1}.pth.tar"
        torch.save(result.best_state_dict, ckpt_path)
        print(
            f"Fold {fold_idx + 1}  val_acc={result.best_val_acc:.4f}  "
            f"val_auc={result.best_val_auc:.4f}  "
            f"stopped_epoch={result.stopped_epoch}  "
            f"-> {ckpt_path}"
        )

        fold_summaries.append(
            {
                "Fold": fold_idx + 1,
                "Validation Accuracy": result.best_val_acc,
                "Validation AUC": result.best_val_auc,
                "EarlyStopped_Epoch": result.stopped_epoch,
            }
        )
        # Save the validation summary incrementally so partial progress
        # survives if training is interrupted.
        save_excel(
            pd.DataFrame(fold_summaries),
            perf_dir / "validation_results.xlsx",
        )

    # --- Test-set evaluation for every fold ---
    print(f"\n{'=' * 60}\nTest-set evaluation\n{'=' * 60}")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_results = []
    for fold_idx in range(args.n_splits):
        ckpt_path = ckpt_dir / f"fold_{fold_idx + 1}.pth.tar"
        model = build_model(args.model).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        tr = evaluate_on_test(model, test_loader, device)
        print(
            f"Fold {fold_idx + 1}  test_acc={tr.accuracy:.4f}  "
            f"test_auc={tr.auc:.4f}"
        )

        # .pkl format compatible with auto_perf_top3fold_excel_*.py
        # (downstream ensembling step).
        save_pickle(
            {
                "test_y_true": tr.y_true.tolist(),
                "test_y_pred": tr.y_pred.tolist(),
                "test_y_prob": tr.y_prob.tolist(),
            },
            perf_dir / f"fold_{fold_idx + 1}_test_results.pkl",
        )
        save_excel(
            pd.DataFrame(
                {
                    "File Name": tr.filenames,
                    "True Label": tr.y_true,
                    "Predicted Label": tr.y_pred,
                    "Probability": tr.y_prob,
                }
            ),
            perf_dir / f"fold_{fold_idx + 1}_test_results.xlsx",
        )

        test_results.append(
            {
                "Fold": fold_idx + 1,
                "Test Accuracy": tr.accuracy,
                "Test AUC": tr.auc,
            }
        )

    save_excel(pd.DataFrame(test_results), perf_dir / "test_results.xlsx")
    print(f"\n✅ Done. Outputs in: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "10-fold cross-validation training + test evaluation for "
            "nail melanoma classification."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        required=True,
        choices=sorted(DEFAULT_HPARAMS.keys()),
        help="Backbone architecture.",
    )
    p.add_argument(
        "--train-path",
        required=True,
        help="ImageFolder root for train/val (must contain class subfolders).",
    )
    p.add_argument(
        "--test-path",
        required=True,
        help="Flat folder of test images (filenames: '<label>_*.ext').",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory for checkpoints/, performance/, and config.json.",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    # Optional hyperparameter overrides.
    p.add_argument("--lr", type=float, default=None, help="Override default LR.")
    p.add_argument(
        "--weight-decay", type=float, default=None, help="Override default WD."
    )
    p.add_argument(
        "--eta-min", type=float, default=None, help="Override default eta_min."
    )
    return p.parse_args()


if __name__ == "__main__":
    run_cross_validation(parse_args())