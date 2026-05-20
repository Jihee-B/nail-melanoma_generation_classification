"""Color Jitter augmentation for the training set.

Applies ``torchvision.transforms.ColorJitter`` once per input image and
saves the augmented copy. This is one of the color-augmentation
conditions used to expand the training set (the others being the
unmodified image and Auto White Balance).

Paper parameters (Methods S1):
    brightness, contrast, saturation each randomly sampled within a
    factor of (0.8, 1.2). Hue is left unchanged.

Usage
-----
    python -m src.preprocessing.color_jitter INPUT_DIR OUTPUT_DIR
    python -m src.preprocessing.color_jitter INPUT_DIR OUTPUT_DIR --seed 42
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def apply_color_jitter(
    input_dir: str,
    output_dir: str,
    brightness: tuple[float, float] = (0.8, 1.2),
    contrast: tuple[float, float] = (0.8, 1.2),
    saturation: tuple[float, float] = (0.8, 1.2),
    suffix: str = "_jitter",
    seed: int | None = None,
) -> int:
    """Apply Color Jitter to every image in ``input_dir``.

    Args:
        input_dir: Folder of source images.
        output_dir: Destination folder (created if missing).
        brightness: ColorJitter brightness factor range.
        contrast: ColorJitter contrast factor range.
        saturation: ColorJitter saturation factor range.
        suffix: Appended to each output filename stem.
        seed: Optional RNG seed. ColorJitter samples a *random* factor
            within each range per image, so set this for reproducible
            output (the original paper run was unseeded).

    Returns:
        Number of images processed.
    """
    os.makedirs(output_dir, exist_ok=True)
    if seed is not None:
        torch.manual_seed(seed)

    jitter = transforms.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
    )

    count = 0
    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
            continue
        in_path = os.path.join(input_dir, filename)
        out_name = f"{Path(filename).stem}{suffix}.png"
        out_path = os.path.join(output_dir, out_name)

        with Image.open(in_path) as image:
            jittered = jitter(image.convert("RGB"))
            jittered.save(out_path)
        count += 1
        print(f"Jittered {filename} -> {out_path}")

    print(f"Done. Processed {count} images.")
    return count


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply Color Jitter augmentation to a folder of images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_dir", help="Folder of source images.")
    p.add_argument("output_dir", help="Destination folder.")
    p.add_argument("--brightness", type=float, nargs=2, default=(0.8, 1.2))
    p.add_argument("--contrast", type=float, nargs=2, default=(0.8, 1.2))
    p.add_argument("--saturation", type=float, nargs=2, default=(0.8, 1.2))
    p.add_argument("--suffix", default="_jitter")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible jitter.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    apply_color_jitter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        brightness=tuple(args.brightness),
        contrast=tuple(args.contrast),
        saturation=tuple(args.saturation),
        suffix=args.suffix,
        seed=args.seed,
    )