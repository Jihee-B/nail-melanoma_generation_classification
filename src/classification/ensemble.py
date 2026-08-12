"""Full-ensemble utilities and out-of-fold (OOF) threshold selection.

After 10-fold cross-validation, every fold's checkpoint is evaluated on
the test set and its per-image positive-class probabilities are saved.
This module:

    1. ``load_all_fold_pkls``       — load the per-fold test-prediction
                                      ``*.pkl`` files produced by
                                      ``cross_validate.py`` (all folds).
    2. ``ensemble_all_folds``       — average per-image probabilities
                                      across *all* folds, then threshold.
    3. ``load_all_val_pkls`` +
       ``compute_oof_youden_cutoff`` — pool each fold's best-epoch
                                      validation predictions and derive a
                                      single classification threshold via
                                      the Youden index.
    4. ``build_fold_auc_table``     — a transparency table listing every
                                      fold's validation and test AUC.

Design note (why the full ensemble, and no fold selection)
----------------------------------------------------------
Earlier versions selected the "best" folds by the smallest
\\|val_auc - test_auc\\| gap. That criterion inspects the *test* AUC to
choose folds, which leaks test information into model selection. To avoid
this, all folds are ensembled unconditionally, and the decision threshold
is chosen from validation (out-of-fold) predictions only — the test set is
never involved in either fold selection or threshold selection.
"""
from __future__ import annotations

import glob
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


# ---------------------------------------------------------------------------
# Excel column inference (for the transparency AUC table)
# ---------------------------------------------------------------------------
# These helpers exist because the validation/test xlsx files were produced
# by several iterations of the training scripts and the AUC/fold column
# names varied slightly. The inference falls back to fuzzy matching.
def _find_first_col(
    df: pd.DataFrame,
    include: List[str],
    exclude: Optional[List[str]] = None,
) -> Optional[str]:
    """Return the first column whose name contains all ``include`` tokens
    and none of the ``exclude`` tokens (case-insensitive)."""
    exclude = exclude or []
    for c in df.columns:
        name = str(c).lower()
        if (
            all(tok.lower() in name for tok in include)
            and not any(tok.lower() in name for tok in exclude)
        ):
            return c
    return None


def _infer_fold_col(df: pd.DataFrame) -> Optional[str]:
    for tokens in (["fold"], ["split"], ["k"]):
        c = _find_first_col(df, tokens)
        if c is not None:
            return c
    return None


def _infer_auc_col(df: pd.DataFrame) -> Optional[str]:
    c = _find_first_col(df, ["auc"], exclude=["ci", "lower", "upper"])
    if c is not None:
        return c
    return _find_first_col(df, ["roc"], exclude=["ci", "lower", "upper"])


def _load_fold_auc(xlsx_path: str) -> pd.DataFrame:
    """Read a validation/test xlsx and return a clean ``(fold, auc)`` table."""
    df = pd.read_excel(xlsx_path)
    fold_col = _infer_fold_col(df)
    auc_col = _infer_auc_col(df)

    if auc_col is None:
        raise ValueError(
            f"[{xlsx_path}] No AUC column found. Expected a column whose "
            f"name contains 'auc'. Columns: {list(df.columns)}"
        )

    out = df.copy()
    if fold_col is None:
        out["fold"] = np.arange(1, len(out) + 1)
    else:
        out["fold"] = out[fold_col]

    def _to_int_fold(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, np.integer)):
            return int(x)
        m = re.search(r"(\d+)", str(x))
        return int(m.group(1)) if m else np.nan

    out["fold"] = out["fold"].apply(_to_int_fold).astype("Int64")
    out["auc"] = pd.to_numeric(out[auc_col], errors="coerce")
    out = out[["fold", "auc"]].dropna().copy()
    out["fold"] = out["fold"].astype(int)
    return out


def build_fold_auc_table(val_xlsx: str, test_xlsx: str) -> pd.DataFrame:
    """Return a per-fold ``(fold, val_auc, test_auc)`` transparency table.

    Unlike the previous ``select_topk_folds``, this performs *no* selection
    — it simply reports each fold's validation and test AUC so that fold
    variability is visible in the summary output. All folds are ensembled.
    """
    df_val = _load_fold_auc(val_xlsx).rename(columns={"auc": "val_auc"})
    df_test = _load_fold_auc(test_xlsx).rename(columns={"auc": "test_auc"})
    df = pd.merge(df_val, df_test, on="fold", how="outer").sort_values("fold")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Probability helpers
# ---------------------------------------------------------------------------
def _flatten_prob(prob: Any) -> np.ndarray:
    """Accept a list/array of shape ``(N,)`` or ``(N, 1)`` and return ``(N,)``."""
    arr = np.asarray(prob)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr.astype(float)


def _discover_folds(run_dir: str, kind: str) -> List[int]:
    """Return sorted 1-based fold IDs for which a ``*_{kind}_results.pkl``
    file exists under ``run_dir`` (``kind`` is 'test' or 'val')."""
    folds = []
    for p in glob.glob(os.path.join(run_dir, f"fold_*_{kind}_results.pkl")):
        m = re.search(rf"fold_(\d+)_{kind}_results\.pkl$", os.path.basename(p))
        if m:
            folds.append(int(m.group(1)))
    return sorted(folds)


def _find_fold_pkl(run_dir: str, fold: int, kind: str) -> str:
    """Locate a fold's prediction pkl (``kind`` = 'test' or 'val').

    Searches a few common filename patterns, then falls back to a glob.
    """
    patterns = [
        os.path.join(run_dir, f"fold_{fold}_{kind}_results.pkl"),
        os.path.join(run_dir, f"fold_{fold}_{kind}.pkl"),
        os.path.join(run_dir, f"fold{fold}_{kind}_results.pkl"),
        os.path.join(run_dir, f"fold{fold}_{kind}.pkl"),
    ]
    for p in patterns:
        if os.path.exists(p):
            return p

    matches = glob.glob(os.path.join(run_dir, f"*{fold}*{kind}*.pkl"))
    if matches:
        return sorted(matches)[0]

    raise FileNotFoundError(
        f"[{run_dir}] No {kind}-pkl found for fold={fold}. "
        f"Tried these names: {[os.path.basename(p) for p in patterns]}"
    )


# ---------------------------------------------------------------------------
# Test-prediction loading (all folds)
# ---------------------------------------------------------------------------
def load_all_fold_pkls(run_dir: str) -> Dict[str, List[Any]]:
    """Load per-fold *test* predictions for every fold under ``run_dir``.

    Each pkl is expected to be a dict with keys:
        ``test_y_true`` (required),
        ``test_y_prob`` (required),
        ``test_y_pred`` (optional — recomputed from prob if absent).

    Returns:
        Dict with keys ``y_true``, ``y_pred``, ``y_prob``, ``pkl_path``,
        ``folds``; the array values are lists with one entry per fold, in
        ascending fold order.
    """
    folds = _discover_folds(run_dir, kind="test")
    if not folds:
        raise FileNotFoundError(
            f"[{run_dir}] No fold_*_test_results.pkl files found."
        )

    out: Dict[str, List[Any]] = {
        "y_true": [],
        "y_pred": [],
        "y_prob": [],
        "pkl_path": [],
        "folds": folds,
    }
    for f in folds:
        pkl_path = _find_fold_pkl(run_dir, f, kind="test")
        with open(pkl_path, "rb") as fp:
            d = pickle.load(fp)
        if "test_y_true" not in d or "test_y_prob" not in d:
            raise KeyError(
                f"[{pkl_path}] Missing required keys. "
                f"Found: {list(d.keys())}; required: test_y_true, test_y_prob"
            )

        y_true = np.asarray(d["test_y_true"]).astype(int).reshape(-1)
        y_prob = _flatten_prob(d["test_y_prob"]).reshape(-1)
        y_pred = np.asarray(
            d.get("test_y_pred", (y_prob >= 0.5).astype(int))
        ).astype(int).reshape(-1)

        out["y_true"].append(y_true)
        out["y_pred"].append(y_pred)
        out["y_prob"].append(y_prob)
        out["pkl_path"].append(pkl_path)
    return out


# ---------------------------------------------------------------------------
# Probability ensembling (all folds)
# ---------------------------------------------------------------------------
def ensemble_all_folds(
    loaded: Dict[str, List[Any]],
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average per-image probabilities across *all* folds, then threshold.

    Folds must share the same test-sample order; otherwise probabilities
    correspond to different images and averaging is meaningless. This
    is checked and the function raises ``ValueError`` on mismatch.

    Args:
        loaded: Output of :func:`load_all_fold_pkls`.
        threshold: Probability cutoff for the positive class. Pass the
            OOF-derived cutoff from :func:`compute_oof_youden_cutoff`.

    Returns:
        ``(y_true, y_pred_ensemble, y_prob_ensemble)`` — all 1-D arrays.
    """
    y_true_0 = loaded["y_true"][0]
    for i, yt in enumerate(loaded["y_true"][1:], start=1):
        if len(yt) != len(y_true_0) or not np.all(yt == y_true_0):
            raise ValueError(
                "Folds have inconsistent test_y_true ordering "
                f"(fold0 len={len(y_true_0)} vs fold{i} len={len(yt)}). "
                "Ensembling requires identical test-sample order across folds."
            )

    prob_mat = np.vstack(
        [p.reshape(1, -1) for p in loaded["y_prob"]]
    )  # shape (n_folds, N)
    avg_prob = np.mean(prob_mat, axis=0)
    y_pred = (avg_prob >= threshold).astype(int)
    return y_true_0, y_pred, avg_prob


# ---------------------------------------------------------------------------
# Out-of-fold (OOF) validation predictions + Youden threshold
# ---------------------------------------------------------------------------
def load_all_val_pkls(run_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Pool best-epoch *validation* predictions across all folds.

    Each ``fold_*_val_results.pkl`` (written by ``cross_validate.py``) holds
    that fold's held-out validation predictions. Because the folds partition
    the training set, concatenating them yields out-of-fold (OOF) predictions
    covering every training sample exactly once.

    Returns:
        ``(oof_y_true, oof_y_prob)`` — concatenated 1-D arrays. Returns
        empty arrays if no validation pkls are present.
    """
    folds = _discover_folds(run_dir, kind="val")
    all_true: List[np.ndarray] = []
    all_prob: List[np.ndarray] = []
    for f in folds:
        pkl_path = _find_fold_pkl(run_dir, f, kind="val")
        with open(pkl_path, "rb") as fp:
            d = pickle.load(fp)
        if "val_y_true" not in d or "val_y_prob" not in d:
            raise KeyError(
                f"[{pkl_path}] Missing required keys. "
                f"Found: {list(d.keys())}; required: val_y_true, val_y_prob"
            )
        all_true.append(np.asarray(d["val_y_true"]).astype(int).reshape(-1))
        all_prob.append(_flatten_prob(d["val_y_prob"]).reshape(-1))

    if not all_true:
        return np.array([], dtype=int), np.array([], dtype=float)
    return np.concatenate(all_true), np.concatenate(all_prob)


def compute_oof_youden_cutoff(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """Return the probability threshold that maximises Youden's J = TPR - FPR.

    The threshold is computed from out-of-fold *validation* predictions, so
    it never sees the test set. ``sklearn.roc_curve`` prepends an infinite
    threshold; it is clipped to 1.0 so the returned cutoff is always a valid
    probability in ``[0, 1]``.

    Args:
        y_true: 1-D array of 0/1 OOF validation labels.
        y_prob: 1-D array of positive-class probabilities.

    Returns:
        The Youden-optimal cutoff. Falls back to ``0.5`` if the threshold
        cannot be computed (e.g., only one class present).
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob).astype(float).reshape(-1)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.5

    fpr, tpr, thr = roc_curve(y_true, y_prob)
    thr = np.clip(thr, 0.0, 1.0)  # roc_curve sets thr[0] = inf
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))
    return float(thr[best_idx])