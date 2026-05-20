"""Binary-classification metrics with bootstrap 95% confidence intervals.

Computes for an ensembled prediction:
    accuracy, AUC, precision/PPV, sensitivity (recall), specificity,
    F1, NPV, positive/negative likelihood ratios, and the confusion
    matrix counts (tn, fp, fn, tp).

Bootstrap CIs use ``n_boot`` resamples with replacement (the paper
uses N=1000) and the 2.5th / 97.5th percentiles of the empirical
distribution as the 95% interval.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class MetricCI:
    """A scalar point estimate together with its 95% bootstrap CI."""

    point: float
    low: float
    high: float


def metricci_to_str(m: MetricCI, digits: int = 5) -> str:
    """Format as ``'point (low-high)'``, or just ``'point'`` if CI is NaN."""
    if np.isnan(m.low) or np.isnan(m.high):
        return f"{m.point:.{digits}f}"
    return f"{m.point:.{digits}f} ({m.low:.{digits}f}-{m.high:.{digits}f})"


# ---------------------------------------------------------------------------
# Confusion-matrix–derived metrics
# ---------------------------------------------------------------------------
def _specificity_from_cm(tn: int, fp: int) -> float:
    denom = tn + fp
    return float(tn / denom) if denom > 0 else 0.0


def _npv_from_cm(tn: int, fn: int) -> float:
    denom = tn + fn
    return float(tn / denom) if denom > 0 else 0.0


def _lr_from_sen_spec(
    sen: float,
    spec: float,
    eps: float = 1e-15,
) -> Tuple[float, float]:
    """Positive and negative likelihood ratios.

    ``eps`` avoids division-by-zero when ``spec == 1`` or ``sen == 1``.
    """
    lr_pos = (sen + eps) / ((1 - spec) + eps)
    lr_neg = ((1 - sen) + eps) / (spec + eps)
    return float(lr_pos), float(lr_neg)


def _percentile_ci(values: List[float]) -> Tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------
def compute_metrics_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 1000,
    seed: int = 1234,
    eps: float = 1e-15,
) -> Dict[str, MetricCI]:
    """Compute every reported metric together with its 95% bootstrap CI.

    Args:
        y_true: 1-D array of integer labels (0 or 1).
        y_pred: 1-D array of integer predictions (0 or 1).
        y_prob: 1-D array of positive-class probabilities.
        n_boot: Number of bootstrap resamples (paper uses 1000).
        seed: RNG seed for reproducible bootstrap.
        eps: Numerical-stability term for likelihood ratios.

    Returns:
        Dict mapping metric name to :class:`MetricCI`. Keys::

            acc, auc,
            precision, sensitivity, specificity, f1_pos,
            ppv, npv, lr_pos, lr_neg,
            tn, fp, fn, tp     (confusion-matrix counts; CI fields NaN)
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob).astype(float).reshape(-1)

    # -------- Point estimates --------
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    sen = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    spec = _specificity_from_cm(tn, fp)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    npv = _npv_from_cm(tn, fn)
    lr_pos, lr_neg = _lr_from_sen_spec(sen, spec, eps=eps)

    # -------- Bootstrap --------
    rng = np.random.RandomState(seed=seed)
    idx = np.arange(len(y_true))
    boot: Dict[str, List[float]] = {
        k: [] for k in
        ("acc", "auc", "prec", "sen", "spec", "f1", "npv", "lr_pos", "lr_neg")
    }
    for _ in range(n_boot):
        sample = rng.choice(idx, size=len(idx), replace=True)
        yt, yp, ypr = y_true[sample], y_pred[sample], y_prob[sample]

        tni, fpi, fni, tpi = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        boot["acc"].append(accuracy_score(yt, yp))
        try:
            boot["auc"].append(roc_auc_score(yt, ypr))
        except ValueError:
            # Resample contained only one class -> AUC undefined.
            pass
        boot["prec"].append(
            precision_score(yt, yp, pos_label=1, zero_division=0)
        )
        sen_i = recall_score(yt, yp, pos_label=1, zero_division=0)
        spec_i = _specificity_from_cm(tni, fpi)
        boot["sen"].append(sen_i)
        boot["spec"].append(spec_i)
        boot["f1"].append(f1_score(yt, yp, pos_label=1, zero_division=0))
        boot["npv"].append(_npv_from_cm(tni, fni))
        lp_i, ln_i = _lr_from_sen_spec(sen_i, spec_i, eps=eps)
        boot["lr_pos"].append(lp_i)
        boot["lr_neg"].append(ln_i)

    def pack(name: str, point_val: float) -> MetricCI:
        vals = boot[name]
        if not vals or np.isnan(point_val):
            return MetricCI(
                point=float(point_val), low=float("nan"), high=float("nan")
            )
        lo, hi = _percentile_ci(vals)
        return MetricCI(point=float(point_val), low=lo, high=hi)

    return {
        "acc": pack("acc", acc),
        "auc": pack("auc", auc),
        "precision": pack("prec", prec),
        "sensitivity": pack("sen", sen),
        "specificity": pack("spec", spec),
        "f1_pos": pack("f1", f1),
        "ppv": pack("prec", prec),  # PPV == precision in the binary case
        "npv": pack("npv", npv),
        "lr_pos": pack("lr_pos", lr_pos),
        "lr_neg": pack("lr_neg", lr_neg),
        # Confusion-matrix counts: point estimates only.
        "tn": MetricCI(float(tn), float("nan"), float("nan")),
        "fp": MetricCI(float(fp), float("nan"), float("nan")),
        "fn": MetricCI(float(fn), float("nan"), float("nan")),
        "tp": MetricCI(float(tp), float("nan"), float("nan")),
    }