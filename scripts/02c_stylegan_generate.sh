#!/usr/bin/env bash
# StyleGAN2-ADA — step c: BULK-generate synthetic images from a trained
# checkpoint. Uses NVIDIA's generate.py. Run from inside the
# stylegan2-ada-pytorch checkout.
#
# This is SEPARATE from the sequential interpolation (see 02d).
#
# NUM_IMAGES defaults to 2000; adjust to whatever the paper reports.
# checkpoint selected by FID + dermatologist review. The final models are
# non-conditional, so --class is intentionally omitted.
set -euo pipefail

NETWORK_PKL="${NETWORK_PKL:?Set NETWORK_PKL to your selected .pkl checkpoint}"
OUTDIR="${OUTDIR:-./fake}"
NUM_IMAGES="${NUM_IMAGES:-2000}"

mkdir -p "${OUTDIR}"

# generate.py uses --seeds; the range 0..(N-1) yields N images.
python generate.py \
    --network "${NETWORK_PKL}" \
    --outdir "${OUTDIR}" \
    --seeds "0-$((NUM_IMAGES - 1))"