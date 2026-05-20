#!/usr/bin/env bash
# StyleGAN2-ADA — step a: convert image folders to TFRecords (256x256).
#
# Run from inside your stylegan2-ada-pytorch checkout (dataset_tool.py is
# NVIDIA's; see the module README). Inputs are the RGB-converted, augmented
# training folders produced by the preprocessing pipeline.
#
# Folder names are study-specific. Edit the variables below (or override
# them via environment variables) to match your own dataset layout.
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-./dataset}"

# --- edit these to match your folders ---
NM_SOURCE="${NM_SOURCE:-${DATASET_ROOT}/benign_train_aug}"       # NM (benign) input images
NUM_SOURCE="${NUM_SOURCE:-${DATASET_ROOT}/melanoma_train_aug}"   # NUM (melanoma) input images
NM_TFRECORDS="${NM_TFRECORDS:-${DATASET_ROOT}/tfrecords_nm}"     # NM output (TFRecords)
NUM_TFRECORDS="${NUM_TFRECORDS:-${DATASET_ROOT}/tfrecords_num}"  # NUM output (TFRecords)
# -----------------------------------------

# NM (benign)
python dataset_tool.py \
    --source "${NM_SOURCE}/" \
    --dest   "${NM_TFRECORDS}/" \
    --width=256 --height=256

# NUM (melanoma)
python dataset_tool.py \
    --source "${NUM_SOURCE}/" \
    --dest   "${NUM_TFRECORDS}/" \
    --width=256 --height=256