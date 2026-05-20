"""Top-K fold selection and probability ensembling.

After 10-fold cross-validation, the paper selects the three folds with
the smallest |val_auc - test_auc| gap and averages their per-image
softmax probabilities for the positive class. This module implements
that selection + ensembling pipeline:

    1. ``select_topk_folds``   — read validation/test xlsx, rank folds
                                 by |val_auc - test_auc|, return top-K
                                 fold IDs and a transparency table.
    2. ``load_topk_fold_pkls`` — load the per-fold ``*.pkl`` files
                                 produced by ``cross_validate.py``.
    3. ``ensemble_from_topk``  — average probabilities across folds and
                                 threshold to obtain hard predictions.
"""
from __future__ import annotations

import glob
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Excel column inference
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


# ---------------------------------------------------------------------------
# Top-K fold selection
# ---------------------------------------------------------------------------
def select_topk_folds(
    val_xlsx: str,
    test_xlsx: str,
    topk: int = 3,
) -> Tuple[List[int], pd.DataFrame]:
    """Select the ``topk`` folds with the smallest \\|val_auc - test_auc\\|.

    Args:
        val_xlsx: Path to ``validation_results.xlsx``.
        test_xlsx: Path to ``test_results.xlsx``.
        topk: Number of folds to keep (paper: 3).

    Returns:
        ``(selected_folds, fold_table)``:
            - ``selected_folds`` (List[int]): 1-based fold IDs.
            - ``fold_table`` (DataFrame): every fold with its val/test
              AUC, absolute gap, rank, and a boolean ``selected_topk``.

    Tie-breaking order (when |val-test| gaps are equal):
        higher test AUC  ->  higher val AUC  ->  lower fold number.
    """
    df_val = _load_fold_auc(val_xlsx).rename(columns={"auc": "val_auc"})
    df_test = _load_fold_auc(test_xlsx).rename(columns={"auc": "test_auc"})

    df = pd.merge(df_val, df_test, on="fold", how="inner")
    df["auc_gap_abs"] = (df["val_auc"] - df["test_auc"]).abs()

    df = df.sort_values(
        ["auc_gap_abs", "test_auc", "val_auc", "fold"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)

    selected = df.head(topk)["fold"].astype(int).tolist()
    df["rank_by_gap"] = np.arange(1, len(df) + 1)
    df["selected_topk"] = df["fold"].isin(selected)
    return selected, df


# ---------------------------------------------------------------------------
# Probability loading
# ---------------------------------------------------------------------------
def _flatten_prob(prob: Any) -> np.ndarray:
    """Accept a list/array of shape ``(N,)`` or ``(N, 1)`` and return ``(N,)``."""
    arr = np.asarray(prob)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr.astype(float)


def _find_fold_pkl(run_dir: str, fold: int) -> str:
    """Locate a fold's test-prediction pkl under ``run_dir``.

    Searches a few common filename patterns, then falls back to a glob.
    """
    patterns = [
        os.path.join(run_dir, f"fold_{fold}_test_results.pkl"),
        os.path.join(run_dir, f"fold_{fold}_test.pkl"),
        os.path.join(run_dir, f"fold{fold}_test_results.pkl"),
        os.path.join(run_dir, f"fold{fold}_test.pkl"),
    ]
    for p in patterns:
        if os.path.exists(p):
            return p

    matches = glob.glob(os.path.join(run_dir, f"*{fold}*test*.pkl"))
    if matches:
        return sorted(matches)[0]

    raise FileNotFoundError(
        f"[{run_dir}] No test-pkl found for fold={fold}. "
        f"Tried these names: {[os.path.basename(p) for p in patterns]}"
    )


def load_topk_fold_pkls(
    run_dir: str,
    folds: List[int],
) -> Dict[str, List[np.ndarray]]:
    """Load per-fold test predictions for the chosen folds.

    Each pkl is expected to be a dict with keys:
        ``test_y_true`` (required),
        ``test_y_prob`` (required),
        ``test_y_pred`` (optional — recomputed from prob if absent).

    Args:
        run_dir: Directory containing ``fold_*_test_results.pkl`` files.
        folds: Sequence of 1-based fold IDs to load.

    Returns:
        Dict with keys ``y_true``, ``y_pred``, ``y_prob``, ``pkl_path``;
        each value is a list with one entry per fold, in the same order
        as the input ``folds``.
    """
    out: Dict[str, List[np.ndarray]] = {
        "y_true": [],
        "y_pred": [],
        "y_prob": [],
        "pkl_path": [],
    }
    for f in folds:
        pkl_path = _find_fold_pkl(run_dir, f)
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
# Probability ensembling
# ---------------------------------------------------------------------------
def ensemble_from_topk(
    loaded: Dict[str, List[np.ndarray]],
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average per-image probabilities across folds, then threshold.

    Folds must share the same test-sample order; otherwise probabilities
    correspond to different images and averaging is meaningless. This
    is checked and the function raises ``ValueError`` on mismatch.

    Args:
        loaded: Output of :func:`load_topk_fold_pkls`.
        threshold: Probability cutoff for the positive class.

    Returns:
        ``(y_true, y_pred_ensemble, y_prob_ensemble)`` — all 1-D arrays.
    """
    y_true_0 = loaded["y_true"][0]
    for i, yt in enumerate(loaded["y_true"][1:], start=1):
        if len(yt) != len(y_true_0) or not np.all(yt == y_true_0):
            raise ValueError(
                "Top-K folds have inconsistent test_y_true ordering "
                f"(fold0 len={len(y_true_0)} vs fold{i} len={len(yt)}). "
                "Ensembling requires identical test-sample order across folds."
            )

    prob_mat = np.vstack(
        [p.reshape(1, -1) for p in loaded["y_prob"]]
    )  # shape (K, N)
    avg_prob = np.mean(prob_mat, axis=0)
    y_pred = (avg_prob >= threshold).astype(int)
    return y_true_0, y_pred, avg_prob