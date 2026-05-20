"""Generate images from a fine-tuned DDPM checkpoint.

Loads a checkpoint saved by ``finetune.py`` and samples images with a
DDIM sampler. The inner denoising loop is the standard ``diffusers``
inference pattern (HuggingFace Diffusion Models Course, Unit 2;
Apache-2.0); the surrounding CLI/IO is study-specific.

``--steps`` defaults to 1000 to reproduce the paper (Methods S2), but can
be set to any value (DDIM allows fewer steps for faster sampling).

Usage
-----
    python -m src.generative.ddpm.generate \\
        --model_path runs_ddpm/nm/diffusion_epoch_37 \\
        --num_images 4000 --image_size 256 --steps 1000 \\
        --output_dir fake/nm
"""
from __future__ import annotations

import argparse
import os

import torch
from diffusers import DDIMScheduler, DDPMPipeline
from PIL import Image
from tqdm import tqdm


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def generate(
    model_path: str,
    num_images: int,
    image_size: int,
    steps: int,
    output_dir: str,
) -> None:
    """Sample ``num_images`` images from a saved DDPM checkpoint."""
    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading model from {model_path}")
    image_pipe = DDPMPipeline.from_pretrained(model_path).to(device)

    scheduler = DDIMScheduler.from_config(image_pipe.scheduler.config)
    scheduler.set_timesteps(num_inference_steps=steps)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {num_images} images ({steps} steps)...")
    for i in range(num_images):
        x = torch.randn(1, 3, image_size, image_size).to(device)
        for t in tqdm(
            scheduler.timesteps,
            desc=f"image {i + 1}/{num_images}",
            leave=False,
        ):
            model_input = scheduler.scale_model_input(x, t)
            noise_pred = image_pipe.unet(model_input, t)["sample"]
            x = scheduler.step(noise_pred, t, x).prev_sample

        img = (x / 2 + 0.5).clamp(0, 1)
        img = img.cpu().permute(0, 2, 3, 1).numpy()[0]
        Image.fromarray((img * 255).astype("uint8")).save(
            os.path.join(output_dir, f"generated_image_{i + 1}.png")
        )

    print(f"Done. {num_images} images saved in {output_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate images from a fine-tuned DDPM checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model_path",
        default="./model_checkpoints/diffusion_final",
        help="Path to a saved DDPM checkpoint (produced by finetune.py).",
    )
    p.add_argument("--num_images", type=int, default=8)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="DDIM denoising steps. Default 1000 reproduces the paper "
             "(Methods S2); lower values are faster at some cost to quality.",
    )
    p.add_argument("--output_dir", default="generated_images")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    generate(
        model_path=args.model_path,
        num_images=args.num_images,
        image_size=args.image_size,
        steps=args.steps,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()