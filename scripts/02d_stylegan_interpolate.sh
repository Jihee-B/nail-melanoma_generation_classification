#!/usr/bin/env bash
# StyleGAN2-ADA — step d: generate a SEQUENTIAL NM->NUM morph by
# interpolating between two latent seeds. Uses interpolate_images.py
# (this module). Run from inside the stylegan2-ada-pytorch checkout
# (copy interpolate_images.py there first).
#
# This is SEPARATE from bulk generation (see 02c).
#
# The sequential model is trained on the COMBINED NM+NUM dataset
# (paper Methods S2). NUM_STEPS defaults to 10 (paper value); adjust as
# needed. SEED1/SEED2 select the start/end images — pick them visually.
set -euo pipefail

NETWORK_PKL="${NETWORK_PKL:?Set NETWORK_PKL to your combined-dataset .pkl checkpoint}"
OUTDIR="${OUTDIR:-./morph/results}"
SEED1="${SEED1:-18}"
SEED2="${SEED2:-13}"
NUM_STEPS="${NUM_STEPS:-10}"

python interpolate_images.py \
    --network "${NETWORK_PKL}" \
    --seed1 "${SEED1}" --seed2 "${SEED2}" \
    --num-steps "${NUM_STEPS}" \
    --outdir "${OUTDIR}"