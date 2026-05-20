#!/usr/bin/env bash
# StyleGAN2-ADA — step b: train separate models for NM and NUM.
#
# Run from inside your stylegan2-ada-pytorch checkout (train.py is NVIDIA's).
# Config matches paper Methods S2: --cfg=stylegan2 sets Adam (lr 0.002) and
# R1 gamma 10; batch 16; 256x256; single GPU. Max training length defaults
# to 25,000 kimg (NVIDIA default). FID is computed by NVIDIA's built-in
# metrics during training.
#
# The TFRecords paths must match those written by 02a — edit below or set
# the same environment variables.
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-./dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output}"

# --- must match 02a's outputs ---
NM_TFRECORDS="${NM_TFRECORDS:-${DATASET_ROOT}/tfrecords_nm}"
NUM_TFRECORDS="${NUM_TFRECORDS:-${DATASET_ROOT}/tfrecords_num}"
# ---------------------------------

# NM (benign)
python train.py \
    --outdir="${OUTPUT_ROOT}/training-runs_nm" \
    --data="${NM_TFRECORDS}" \
    --gpus=1 --cfg=stylegan2 --batch=16

# NUM (melanoma)
python train.py \
    --outdir="${OUTPUT_ROOT}/training-runs_num" \
    --data="${NUM_TFRECORDS}" \
    --gpus=1 --cfg=stylegan2 --batch=16

# For the sequential-morph model, train once more on the COMBINED NM+NUM
# dataset (NUM amelanotic cases excluded), e.g.:
#   COMBINED_TFRECORDS="${DATASET_ROOT}/tfrecords_combined"
#   python train.py \
#       --outdir="${OUTPUT_ROOT}/training-runs_combined" \
#       --data="${COMBINED_TFRECORDS}" \
#       --gpus=1 --cfg=stylegan2 --batch=16