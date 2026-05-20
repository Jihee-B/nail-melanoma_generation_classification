#!/usr/bin/env bash
# Preprocessing pipeline (paper Methods S1).
#
# This script runs the SCRIPTED steps only (RGB conversion, Color Jitter).
# Several steps are MANUAL or use EXTERNAL tools — they are shown below as
# commented commands for reference. Full details: src/preprocessing/README.md
#
# Edit the folder variables below (or override via environment variables).
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./data}"

# Per-class raw input folders (after manual center-crop / LaMa / upscaling)
NM_RAW="${NM_RAW:-${DATA_ROOT}/nm_raw}"      # NM (benign)
NUM_RAW="${NUM_RAW:-${DATA_ROOT}/num_raw}"   # NUM (melanoma)

# ---------------------------------------------------------------------
# Step 1. Center crop (1:1)         — MANUAL (image editor)
# Step 2. Artifact removal (LaMa)   — MANUAL  https://github.com/advimman/lama
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Step 3. Up-scaling (Real-ESRGAN)  — EXTERNAL executable
#   4x for images < 100x100 px, 3x for other images < 256x256 px.
#   ./realesrgan-ncnn-vulkan -i <input_folder> -o <output_folder> -s 4 -f png
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Step 4. RGB unification           — convert_to_rgb.py (this repo)
# ---------------------------------------------------------------------
python -m src.preprocessing.convert_to_rgb "${NM_RAW}"  "${DATA_ROOT}/nm_rgb"
python -m src.preprocessing.convert_to_rgb "${NUM_RAW}" "${DATA_ROOT}/num_rgb"

# ---------------------------------------------------------------------
# Step 5. Auto White Balance (AWB)  — EXTERNAL (Deep White-Balance)
#   From the Deep_White_Balance/PyTorch folder:
#   python demo_images.py --input_dir "${DATA_ROOT}/nm_rgb"  --output_dir "${DATA_ROOT}/nm_awb"  --task AWB
#   python demo_images.py --input_dir "${DATA_ROOT}/num_rgb" --output_dir "${DATA_ROOT}/num_awb" --task AWB
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Step 6. Color Jitter              — color_jitter.py (this repo)
# ---------------------------------------------------------------------
python -m src.preprocessing.color_jitter "${DATA_ROOT}/nm_rgb"  "${DATA_ROOT}/nm_jitter"
python -m src.preprocessing.color_jitter "${DATA_ROOT}/num_rgb" "${DATA_ROOT}/num_jitter"

# ---------------------------------------------------------------------
# NOTE: how the three conditions (unmodified / AWB / Color Jitter) are
# combined per class is a study-specific decision — see the paper.
# ---------------------------------------------------------------------
echo "Scripted preprocessing steps complete (RGB + Color Jitter)."