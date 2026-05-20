"""Convert images to 3-channel RGB PNG.

Some clinical photographs are stored as grayscale, palette, RGBA, or
CMYK. Both StyleGAN2-ADA's ``dataset_tool.py`` and the classification
pipeline expect consistent 3-channel RGB input, so this script
normalises every image in a folder to RGB and re-saves it as PNG.

Usage
-----
    python -m src.preprocessing.convert_to_rgb INPUT_DIR OUTPUT_DIR
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from PIL import Image

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def convert_images_to_rgb(
    input_dir: str,
    output_dir: str,
    log_path: str | None = None,
) -> tuple[int, int]:
    """Convert every supported image in ``input_dir`` to RGB PNG.

    Args:
        input_dir: Folder of source images.
        output_dir: Destination folder (created if missing). Each output
            keeps the original stem with a ``.png`` extension.
        log_path: Optional path for a conversion log file. If ``None``,
            logging goes to the console only.

    Returns:
        ``(processed_count, skipped_count)``.
    """
    os.makedirs(output_dir, exist_ok=True)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s",
        handlers=handlers,
        force=True,
    )

    processed, skipped = 0, 0
    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
            skipped += 1
            logging.warning("Skipped %s: unsupported format", filename)
            continue

        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, f"{Path(filename).stem}.png")
        try:
            with Image.open(in_path) as image:
                image.convert("RGB").save(out_path)
            processed += 1
            logging.info("Converted %s -> %s", filename, out_path)
        except Exception as exc:  # noqa: BLE001
            logging.error("Error processing %s: %s", filename, exc)

    logging.info(
        "Done. Processed: %d, Skipped: %d", processed, skipped
    )
    return processed, skipped


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a folder of images to 3-channel RGB PNG.",
    )
    p.add_argument("input_dir", help="Folder of source images.")
    p.add_argument("output_dir", help="Destination folder.")
    p.add_argument(
        "--log-path",
        default=None,
        help="Optional path to write a conversion log file.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_images_to_rgb(args.input_dir, args.output_dir, args.log_path)