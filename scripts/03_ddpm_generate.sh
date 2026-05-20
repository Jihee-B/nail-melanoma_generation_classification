#!/usr/bin/env bash
# DDPM — step b: generate synthetic images from the selected checkpoints.
#
# NUM_IMAGES defaults to 2000 per class (2000 NM + 2000 NUM = the "DIFF4000"
# augmentation set). STEPS defaults to 1000 (paper Methods S2) and is
# user-adjustable. Set NM_CKPT / NUM_CKPT to the checkpoints chosen by FID +
# dermatologist review.
set -euo pipefail

RUNS_ROOT="${RUNS_ROOT:-./runs_ddpm}"
FAKE_ROOT="${FAKE_ROOT:-./fake}"

NM_CKPT="${NM_CKPT:?Set NM_CKPT to your selected NM checkpoint, e.g. ${RUNS_ROOT}/nm/diffusion_epoch_67}"
NUM_CKPT="${NUM_CKPT:?Set NUM_CKPT to your selected NUM checkpoint, e.g. ${RUNS_ROOT}/num/diffusion_epoch_37}"

NUM_IMAGES="${NUM_IMAGES:-2000}"
STEPS="${STEPS:-1000}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"

# NM (benign)
python -m src.generative.ddpm.generate \
    --model_path "${NM_CKPT}" \
    --num_images "${NUM_IMAGES}" --image_size "${IMAGE_SIZE}" --steps "${STEPS}" \
    --output_dir "${FAKE_ROOT}/nm"

# NUM (melanoma)
python -m src.generative.ddpm.generate \
    --model_path "${NUM_CKPT}" \
    --num_images "${NUM_IMAGES}" --image_size "${IMAGE_SIZE}" --steps "${STEPS}" \
    --output_dir "${FAKE_ROOT}/num"