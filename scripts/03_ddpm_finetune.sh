#!/usr/bin/env bash
# DDPM — step a: fine-tune from google/ddpm-celebahq-256 for NM and NUM.
#
# Config matches paper Methods S2: AdamW, lr 1e-5, batch 16, 256x256.
# Per-epoch checkpoints (diffusion_epoch_N/) and a fid_scores_*.json are
# written to the run dir; select the best checkpoint by FID + visual review.
#
# Edit the folder variables below (or override via environment variables).
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./data}"
RUNS_ROOT="${RUNS_ROOT:-./runs_ddpm}"

# Per-class training image folders (HF datasets imagefolder)
NM_DATA="${NM_DATA:-${DATA_ROOT}/nm_train}"     # NM (benign)
NUM_DATA="${NUM_DATA:-${DATA_ROOT}/num_train}"  # NUM (melanoma)

BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-5}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"

# NM (benign)
python -m src.generative.ddpm.finetune \
    --dataset_name "${NM_DATA}" \
    --batch_size "${BATCH_SIZE}" --lr "${LR}" --image_size "${IMAGE_SIZE}" \
    --save_dir "${RUNS_ROOT}/nm"

# NUM (melanoma)
python -m src.generative.ddpm.finetune \
    --dataset_name "${NUM_DATA}" \
    --batch_size "${BATCH_SIZE}" --lr "${LR}" --image_size "${IMAGE_SIZE}" \
    --save_dir "${RUNS_ROOT}/num"

# To resume an interrupted run, re-run with e.g.:
#   --resume_from "${RUNS_ROOT}/nm/diffusion_epoch_19"