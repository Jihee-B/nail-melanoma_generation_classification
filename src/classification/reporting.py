"""Per-run reporting: discover runs, ensemble top-K folds, write Excel.

This is the entry point for evaluating the cross-validation outputs
produced by ``cross_validate.py``. It walks a project root, finds every
directory that holds both ``validation_results.xlsx`` and
``test_results.xlsx``, and for each one:

    1. Picks the top-K folds by \\|val_auc - test_auc\\|.
    2. Loads their per-image probabilities (``fold_*_test_results.pkl``).
    3. Averages probabilities -> thresholds -> ensemble predictions.
    4. Computes all binary metrics with bootstrap 95% CIs.
    5. Writes one row of a summary Excel sheet (plus a fold-selection
       sheet for transparency).

Usage
-----
    python -m src.classification.reporting \\
        --project-root runs/ \\
        --out runs/_summary.xlsx \\
        --topk 3 \\
        --threshold 0.5 \\
        --n-boot 1000 \\
        --seed 1234
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .ensemble import (
    ensemble_from_topk,
    load_topk_fold_pkls,
    select_topk_folds,
)
from .metrics import compute_metrics_with_ci, metricci_to_str


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------
def discover_runs(project_root: str) -> List[str]:
    """Return all directories containing both validation + test xlsx files.

    A "run directory" is the folder that holds the per-fold predictions
    (e.g., ``runs/efficientnet_b0/originaug/performance/``).
    """
    runs = []
    for dirpath, _, filenames in os.walk(project_root):
        lower = {f.lower() for f in filenames}
        if (
            "validation_results.xlsx" in lower
            and "test_results.xlsx" in lower
        ):
            runs.append(dirpath)
    return sorted(runs)


def _parse_model_and_exp(perf_dir: str) -> Tuple[str, str]:
    """Heuristically extract ``(model, experiment)`` tags from a path.

    Recognises two layouts:

        - new (``cross_validate.py``): ``.../<model>/<experiment>/performance/``
        - original-script layout:      ``.../<model>/Perf/<experiment>``
    """
    parts = os.path.normpath(perf_dir).split(os.sep)

    # New: last component is 'performance' (or 'Perf') -> model is two
    # levels up, experiment is one level up.
    if parts and parts[-1].lower() in ("performance", "perf"):
        if len(parts) >= 3:
            return parts[-3], parts[-2]

    # Original-script: .../<Model>/Perf/<experiment>
    for i, p in enumerate(parts):
        if p.lower() == "perf" and 0 < i < len(parts) - 1:
            return parts[i - 1], os.path.join(*parts[i + 1:])

    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return "unknown_model", os.path.basename(perf_dir)


# ---------------------------------------------------------------------------
# Confusion-matrix plot
# ---------------------------------------------------------------------------
def _save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    label_names: Tuple[str, str] = ("NM", "NUM"),
    title: str = "Confusion Matrix",
    figsize: Tuple[float, float] = (5.5, 5.0),
    dpi: int = 300,
) -> None:
    """Save a 2x2 confusion-matrix heatmap with count + percent annotations.

    Args:
        y_true: 1-D array of 0/1 ground-truth labels.
        y_pred: 1-D array of 0/1 predictions (post-ensemble).
        save_path: Output PNG path; parent dirs are created if missing.
        label_names: Display labels for class 0 and class 1
            (default: ``("NM", "NUM")``).
        title: Figure title.
        figsize: Matplotlib figure size in inches.
        dpi: Resolution.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    total = cm.sum()
    cm_pct = (cm / total * 100.0) if total > 0 else cm.astype(float)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=50, aspect="equal")

    # Cell annotations: "count\n(xx.x%)"
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Switch text colour for legibility on darker cells.
            text_color = "white" if cm_pct[i, j] > 25 else "black"
            ax.text(
                j, i,
                f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                ha="center", va="center",
                color=text_color, fontsize=12,
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(list(label_names), fontsize=11)
    ax.set_yticklabels(list(label_names), fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=11, labelpad=8)
    ax.set_ylabel("True label", fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=13, pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Percent of total (%)", fontsize=10)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-run report
# ---------------------------------------------------------------------------
_METRIC_KEYS = (
    "acc", "auc", "precision", "sensitivity", "specificity",
    "f1_pos", "ppv", "npv", "lr_pos", "lr_neg",
)
_COUNT_KEYS = ("tn", "fp", "fn", "tp")


def build_report_for_run(
    run_dir: str,
    topk: int,
    threshold: float,
    n_boot: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute ensemble metrics + the fold-selection table for one run.

    Returns:
        ``(metric_row_df, fold_table_df)`` — single-run frames ready to
        be concatenated across runs.
    """
    val_xlsx = os.path.join(run_dir, "validation_results.xlsx")
    test_xlsx = os.path.join(run_dir, "test_results.xlsx")

    selected_folds, fold_table = select_topk_folds(
        val_xlsx, test_xlsx, topk=topk
    )
    loaded = load_topk_fold_pkls(run_dir, selected_folds)
    y_true, y_pred, y_prob = ensemble_from_topk(loaded, threshold=threshold)
    metrics = compute_metrics_with_ci(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        n_boot=n_boot,
        seed=seed,
    )

    model, experiment = _parse_model_and_exp(run_dir)

    # Save per-run confusion matrix PNG next to the predictions.
    _save_confusion_matrix_plot(
        y_true=y_true,
        y_pred=y_pred,
        save_path=Path(run_dir) / "confusion_matrix.png",
        title=f"{model} — {experiment}",
    )

    row = {
        "model": model,
        "experiment": experiment,
        "run_dir": run_dir,
        "topk": topk,
        "threshold": threshold,
        "selected_folds": ",".join(map(str, selected_folds)),
        "pkl_paths": " | ".join(loaded["pkl_path"]),
    }
    for k in _METRIC_KEYS:
        row[k] = metrics[k].point
        row[f"{k}_ci95"] = metricci_to_str(metrics[k])
    for k in _COUNT_KEYS:
        row[k] = int(metrics[k].point)

    fold_table = fold_table.copy()
    fold_table.insert(0, "model", model)
    fold_table.insert(1, "experiment", experiment)
    fold_table.insert(2, "run_dir", run_dir)
    return pd.DataFrame([row]), fold_table


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    runs = discover_runs(args.project_root)
    if not runs:
        raise FileNotFoundError(
            f"No run directories found under '{args.project_root}'. "
            "Looking for folders that contain both "
            "validation_results.xlsx and test_results.xlsx."
        )

    print(f"Found {len(runs)} run director{'y' if len(runs) == 1 else 'ies'}. "
          "Processing...\n")

    all_rows: List[pd.DataFrame] = []
    all_folds: List[pd.DataFrame] = []

    for run_dir in runs:
        try:
            df_row, df_fold = build_report_for_run(
                run_dir=run_dir,
                topk=args.topk,
                threshold=args.threshold,
                n_boot=args.n_boot,
                seed=args.seed,
            )
            all_rows.append(df_row)
            all_folds.append(df_fold)
            print(f"  [OK]   {run_dir}")
        except Exception as e:  # noqa: BLE001
            model, exp = _parse_model_and_exp(run_dir)
            all_rows.append(
                pd.DataFrame([{
                    "model": model,
                    "experiment": exp,
                    "run_dir": run_dir,
                    "error": repr(e),
                }])
            )
            print(f"  [FAIL] {run_dir}: {e}", file=sys.stderr)

    summary_df = pd.concat(all_rows, ignore_index=True)
    fold_df = (
        pd.concat(all_folds, ignore_index=True)
        if all_folds else pd.DataFrame()
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        summary_df.to_excel(w, index=False, sheet_name="ensemble_metrics")
        if not fold_df.empty:
            fold_df.to_excel(w, index=False, sheet_name="fold_selection")

    print(f"\n✅ Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Discover CV runs, ensemble the top-K folds, and write a "
            "summary Excel with bootstrap-CI metrics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--project-root",
        default=".",
        help="Directory to walk searching for runs.",
    )
    p.add_argument(
        "--out",
        default="cnn_perf_summary.xlsx",
        help="Output Excel file path.",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Number of best folds to ensemble (paper: 3).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability cutoff for ensemble predictions.",
    )
    p.add_argument(
        "--n-boot",
        type=int,
        default=1000,
        help="Bootstrap resamples for 95%% CIs.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="RNG seed for bootstrap reproducibility.",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())