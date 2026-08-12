#!/usr/bin/env bash
# =============================================================================
# Train and evaluate all classification configurations for the nail
# melanoma study: 4 backbones x 3 training conditions = 12 runs,
# followed by full-fold ensembling + OOF-Youden thresholding +
# bootstrap-CI aggregation.
#
# Required environment variables
# ------------------------------
#   DATA_ROOT  Path to the directory holding the data folders referenced
#              in the FOLDER MAPPING section below.
#
# Optional environment variables
# ------------------------------
#   RUNS_ROOT  Output root (default: ./runs).
#   MODELS     Space-separated subset of backbones to train. Default:
#              "resnext50 efficientnet_b0 vit swin". Useful for partial
#              re-runs, e.g. MODELS="vit swin".
#
# Example
# -------
#   DATA_ROOT=/data/nail_melanoma ./scripts/04_train_classifiers.sh
#   DATA_ROOT=/data/nail_melanoma MODELS="resnext50" \
#       ./scripts/04_train_classifiers.sh
#
# Output layout
# -------------
#   $RUNS_ROOT/
#     |-- <model>/<condition>/            # checkpoints + per-fold metrics
#     |   |-- checkpoints/fold_*.pth.tar
#     |   |-- performance/{validation,test}_results.xlsx
#     |   |               + fold_*_{val,test}_results.pkl/.xlsx
#     |   +-- config.json
#     +-- _summary.xlsx                   # full-fold ensemble + 95% CIs
# =============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------
: "${DATA_ROOT:?DATA_ROOT must be set (root of the data folders)}"
: "${RUNS_ROOT:=./runs}"
: "${MODELS:=resnext50 efficientnet_b0 vit swin}"

# --- FOLDER MAPPING ----------------------------------------------------------
# Adjust the paths below to match your local data folder names.
#
# Each TRAIN_* must be an ImageFolder layout with class subfolders
# (e.g., benign/, melanoma/). The amount of synthetic augmentation is
# defined entirely by the *contents* of these folders — populate each
# augmented folder with the desired mix of real and generated images.
# In the paper, the augmented conditions used a synthetic-to-real ratio of
# 0.8 (800 synthetic images per class; 1,600 in total), selected via the
# ratio sweep in Supplementary Methods S4; substitute any amount you wish.
#
# TEST_INTERNAL is a flat folder where filenames encode the label as a
# prefix before the first underscore ('0_xxx.png', '1_yyy.png').
# -----------------------------------------------------------------------------
TRAIN_ORIGINAL="${DATA_ROOT}/train_original"       # real images only
TRAIN_GAN_AUG="${DATA_ROOT}/train_with_gan"        # real + StyleGAN2 synthetic
TRAIN_DIFF_AUG="${DATA_ROOT}/train_with_diff"      # real + diffusion synthetic
TEST_INTERNAL="${DATA_ROOT}/test_internal"
# -----------------------------------------------------------------------------

# Condition label -> training folder. Labels ('original', 'gan', 'diff')
# become the <condition> directory name under each model's run folder.
CONDITIONS=(
    "original:${TRAIN_ORIGINAL}"
    "gan:${TRAIN_GAN_AUG}"
    "diff:${TRAIN_DIFF_AUG}"
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
    for cond_spec in "${CONDITIONS[@]}"; do
        cond_name="${cond_spec%%:*}"   # text before the first ':'
        cond_path="${cond_spec#*:}"    # text after the first ':'

        echo
        echo "============================================================"
        echo "Training: model=${model}  condition=${cond_name}"
        echo "  train: ${cond_path}"
        echo "  test:  ${TEST_INTERNAL}"
        echo "============================================================"

        python -m src.classification.cross_validate \
            --model        "${model}" \
            --train-path   "${cond_path}" \
            --test-path    "${TEST_INTERNAL}" \
            --output-dir   "${RUNS_ROOT}/${model}/${cond_name}"
    done
done

# --- Aggregate metrics -------------------------------------------------------
# Ensembles all folds (no fold selection) and derives the classification
# threshold per run from out-of-fold validation predictions (Youden index).
# Omit --threshold to use the OOF cutoff; pass e.g. --threshold 0.5 to force
# a fixed cutoff instead.
echo
echo "============================================================"
echo "Aggregating: full-fold ensemble + OOF-Youden threshold + bootstrap 95% CI"
echo "============================================================"
python -m src.classification.reporting \
    --project-root "${RUNS_ROOT}" \
    --out          "${RUNS_ROOT}/_summary.xlsx" \
    --n-boot       1000 \
    --seed         1234

echo
echo "Done. Summary: ${RUNS_ROOT}/_summary.xlsx"