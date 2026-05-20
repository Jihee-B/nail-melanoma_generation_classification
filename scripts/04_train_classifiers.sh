#!/usr/bin/env bash
# =============================================================================
# Train and evaluate all classification configurations for the nail
# melanoma study: 4 backbones × 3 training conditions = 12 runs,
# followed by top-3 fold ensembling + bootstrap-CI aggregation.
#
# Required environment variables
# ------------------------------
#   DATA_ROOT  Path to the directory holding the four data folders
#              referenced in the FOLDER MAPPING section below.
#
# Optional environment variables
# ------------------------------
#   RUNS_ROOT  Output root (default: ./runs).
#   MODELS     Space-separated subset of backbones to train. Default:
#              "resnet18 efficientnet_b0 vit swin". Useful for partial
#              re-runs, e.g. MODELS="vit swin".
#
# Example
# -------
#   DATA_ROOT=/data/nail_melanoma ./scripts/04_train_classifiers.sh
#   DATA_ROOT=/data/nail_melanoma MODELS="resnet18" \
#       ./scripts/04_train_classifiers.sh
#
# Output layout
# -------------
#   $RUNS_ROOT/
#     ├── <model>/<condition>/            # checkpoints + per-fold metrics
#     │   ├── checkpoints/fold_*.pth.tar
#     │   ├── performance/{validation,test}_results.xlsx + fold_*.pkl/.xlsx
#     │   └── config.json
#     └── _summary.xlsx                   # top-3 ensemble + 95% bootstrap CIs
# =============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------
: "${DATA_ROOT:?DATA_ROOT must be set (root of the data folders)}"
: "${RUNS_ROOT:=./runs}"
: "${MODELS:=resnet18 efficientnet_b0 vit swin}"

# --- FOLDER MAPPING ----------------------------------------------------------
# Adjust the four paths below to match your local data folder names.
# Each TRAIN_* must contain class subfolders for ImageFolder
# (e.g., benign/, melanoma/). TEST_INTERNAL is a flat folder where
# filenames encode the label as a prefix ('0_xxx.png', '1_yyy.png').
# -----------------------------------------------------------------------------
TRAIN_ORIGINAL="${DATA_ROOT}/train_original"
TRAIN_GAN_AUG="${DATA_ROOT}/train_with_gan4000"
TRAIN_DIFF_AUG="${DATA_ROOT}/train_with_diff4000"
TEST_INTERNAL="${DATA_ROOT}/test_internal"
# -----------------------------------------------------------------------------

CONDITIONS=(
    "original:${TRAIN_ORIGINAL}"
    "gan4000:${TRAIN_GAN_AUG}"
    "diff4000:${TRAIN_DIFF_AUG}"
)

# --- Sanity check that data folders exist before any training ---
for path in "${TRAIN_ORIGINAL}" "${TRAIN_GAN_AUG}" "${TRAIN_DIFF_AUG}" "${TEST_INTERNAL}"; do
    if [[ ! -d "${path}" ]]; then
        echo "ERROR: data folder not found: ${path}" >&2
        echo "Edit the FOLDER MAPPING section in this script to match your layout." >&2
        exit 1
    fi
done

# --- Training loop -----------------------------------------------------------
for model in $MODELS; do
    for cond_spec ...
        python -m src.classification.cross_validate \
            --model        "${model}" \
            --train-path   "${cond_path}" \
            --test-path    "${TEST_INTERNAL}" \
            --output-dir   "${RUNS_ROOT}/${model}/${cond_name}"
    done
done

# --- Aggregate metrics -------------------------------------------------------
echo
echo "============================================================"
echo "Aggregating: top-3 fold ensemble + bootstrap 95% CI"
echo "============================================================"
python -m src.classification.reporting \
    --project-root "${RUNS_ROOT}" \
    --out          "${RUNS_ROOT}/_summary.xlsx" \
    --topk         3 \
    --threshold    0.5 \
    --n-boot       1000 \
    --seed         1234

echo
echo "✅ Done. Summary: ${RUNS_ROOT}/_summary.xlsx"