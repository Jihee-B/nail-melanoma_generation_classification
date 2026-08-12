"""Training and evaluation functions for nail melanoma classification.

These functions are model-agnostic — they take an already-built
``nn.Module`` and any PyTorch ``DataLoader`` and run:

    ``train_one_epoch``  : a single training epoch
    ``validate``         : a single validation pass
    ``train_fold``       : one full CV fold (loop + best-model tracking
                           + early stopping)
    ``evaluate_on_test`` : trained model → test-set predictions

Orchestration (10-fold cross-validation, optimiser/scheduler setup,
output paths) lives in ``cross_validate.py``.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class FoldResult:
    """Result of training a single CV fold."""

    best_state_dict: dict
    best_val_loss: float
    best_val_acc: float
    best_val_auc: float
    best_val_true: List[int] = field(default_factory=list)
    best_val_pred: List[int] = field(default_factory=list)
    best_val_prob: List[float] = field(default_factory=list)
    stopped_epoch: int = 0


@dataclass
class TestResult:
    """Result of evaluating a single trained fold on a test set."""

    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    filenames: List[str]
    accuracy: float
    auc: float


# ---------------------------------------------------------------------------
# Core loops
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Run one training epoch.

    Returns:
        ``(epoch_loss, epoch_accuracy, epoch_auc)``. AUC is computed on
        hard predictions (as in the original code) and is primarily a
        sanity-check diagnostic — the trustworthy AUC is computed on
        softmax probabilities during validation.
    """
    model.train()
    running_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    try:
        epoch_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        # Happens when a mini-batch covers only one class.
        epoch_auc = 0.5
    return epoch_loss, epoch_acc, epoch_auc


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, List[int], List[int], List[float]]:
    """Run one validation pass.

    Returns:
        ``(loss, acc, auc, y_true, y_pred, y_prob)`` where ``loss`` is the
        mean cross-entropy over the validation set and ``y_prob`` is the
        softmax probability for the positive class (index 1).
    """
    model.eval()
    running_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    val_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5
    return val_loss, acc, auc, all_labels, all_preds, all_probs


# ---------------------------------------------------------------------------
# Full fold training
# ---------------------------------------------------------------------------
def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: torch.device,
    num_epochs: int = 100,
    patience: int = 15,
    fold_idx: int = 0,
    verbose: bool = True,
) -> FoldResult:
    """Train one CV fold with early stopping by validation loss.

    Selection rule (matches the paper / original code):
        The "best" checkpoint is the epoch with the lowest *validation
        loss*. Early stopping triggers if validation loss has not improved
        for ``patience`` consecutive epochs. Validation accuracy and AUC at
        that best epoch are recorded alongside for reporting.

    Note on best-weights capture:
        Uses ``copy.deepcopy(model.state_dict())`` rather than
        ``model.state_dict().copy()`` — the latter is a shallow copy
        whose tensor values remain shared with the live model and would
        be silently overwritten by subsequent training steps.
    """
    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_val_auc = -1.0
    best_state = None
    best_true: List[int] = []
    best_pred: List[int] = []
    best_prob: List[float] = []
    epochs_no_improve = 0
    stopped_epoch = num_epochs

    pbar = tqdm(
        range(num_epochs),
        desc=f"Fold {fold_idx + 1}",
        disable=not verbose,
        leave=False,
    )
    for epoch in pbar:
        train_loss, train_acc, _ = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        scheduler.step()
        val_loss, val_acc, val_auc, val_true, val_pred, val_prob = validate(
            model, val_loader, criterion, device
        )

        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix(
            {
                "lr": f"{current_lr:.2e}",
                "tr_loss": f"{train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_acc": f"{val_acc:.4f}",
                "val_auc": f"{val_auc:.4f}",
            }
        )

        # Best checkpoint = lowest validation loss (small epsilon guards
        # against selecting a numerically-equal later epoch).
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            best_true, best_pred, best_prob = val_true, val_pred, val_prob
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                stopped_epoch = epoch + 1
                if verbose:
                    pbar.write(
                        f"  Fold {fold_idx + 1}: early stopping at epoch "
                        f"{stopped_epoch} (best val_loss={best_val_loss:.4f})."
                    )
                break
    else:
        # Loop completed without early stopping.
        stopped_epoch = num_epochs

    if best_state is None:
        raise RuntimeError(
            f"Fold {fold_idx + 1} produced no best state — validation loss "
            "never improved below the initial sentinel. Check that "
            "validation data and labels are correctly configured."
        )

    return FoldResult(
        best_state_dict=best_state,
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc,
        best_val_auc=best_val_auc,
        best_val_true=best_true,
        best_val_pred=best_pred,
        best_val_prob=best_prob,
        stopped_epoch=stopped_epoch,
    )


# ---------------------------------------------------------------------------
# Test-set evaluation
# ---------------------------------------------------------------------------
def evaluate_on_test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> TestResult:
    """Run a trained model on a flat-folder test set.

    Expects ``test_loader`` to yield 3-tuples ``(image, label, filename)``
    — the default output of ``FlatImageDataset``.
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[float] = []
    all_filenames: List[str] = []

    with torch.no_grad():
        for inputs, labels, filenames in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_filenames.extend(list(filenames))

    y_true = np.asarray(all_labels, dtype=int)
    y_pred = np.asarray(all_preds, dtype=int)
    y_prob = np.asarray(all_probs, dtype=float)

    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5

    return TestResult(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        filenames=all_filenames,
        accuracy=acc,
        auc=auc,
    )