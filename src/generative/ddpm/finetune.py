"""DDPM fine-tuning from a pretrained ``google/ddpm-celebahq-256`` checkpoint.

Built with HuggingFace ``diffusers``. The fine-tuning and sampling loops
are adapted from the HuggingFace Diffusion Models Course, Unit 2
("Fine-Tuning and Guidance"), which is licensed under Apache-2.0:
https://huggingface.co/learn/diffusion-course/en/unit2/2

Added on top of that template for this study: FID-based per-epoch
checkpoint selection, resume support, JSON logging of training/FID
history, an argparse CLI, and the dataset-specific fine-tuning
configuration described below.

Per paper Methods S2:
    - Base checkpoint : google/ddpm-celebahq-256
    - Fine-tuned independently for NM and NUM
    - Optimizer : AdamW;  learning rate : 1e-5;  batch size : 16;  size : 256
    - Final images generated with 1000 inference steps (see ``generate.py``)

During training, an inexpensive per-epoch FID (DDIM sampler, fewer steps,
small sample) is logged to help select the best checkpoint. Final,
full-quality generation is done separately by ``generate.py`` with the
full DDPM sampler.

External dependencies: ``diffusers``, ``datasets``, ``pytorch-fid``.

Usage
-----
    # fresh start
    python -m src.generative.ddpm.finetune \\
        --dataset_name /path/to/train_images_folder \\
        --batch_size 16 --lr 1e-5 --image_size 256 \\
        --save_dir runs_ddpm/nm

    # resume from a saved epoch
    python -m src.generative.ddpm.finetune \\
        --dataset_name /path/to/train_images_folder \\
        --save_dir runs_ddpm/nm \\
        --resume_from runs_ddpm/nm/diffusion_epoch_19
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from datasets import load_dataset
from diffusers import DDIMScheduler, DDPMPipeline
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------
def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_dataloader(dataset_name: str, image_size: int, batch_size: int):
    """Load an image folder via HF ``datasets`` and apply DDPM preprocessing."""
    dataset = load_dataset(dataset_name, split="train")

    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # scale to [-1, 1]
        ]
    )

    def transform(examples):
        images = [preprocess(img.convert("RGB")) for img in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def save_batch_preview(dataloader, path: str) -> None:
    """Save a small grid of a training batch for a quick visual sanity check."""
    batch = next(iter(dataloader))
    grid = torchvision.utils.make_grid(batch["images"], nrow=4)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5)
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close()


# ---------------------------------------------------------------------------
# Per-epoch FID (used only for checkpoint selection)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_samples_for_fid(
    image_pipe, scheduler, n: int, image_size: int, device: str, out_dir: str
) -> None:
    """Sample ``n`` images with a (fast) DDIM scheduler for FID estimation."""
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n):
        x = torch.randn(1, 3, image_size, image_size).to(device)
        for t in scheduler.timesteps:
            model_input = scheduler.scale_model_input(x, t)
            noise_pred = image_pipe.unet(model_input, t)["sample"]
            x = scheduler.step(noise_pred, t, x).prev_sample
        img = (x[0].permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5) * 255
        Image.fromarray(img.numpy().astype(np.uint8)).save(
            os.path.join(out_dir, f"generated_{i}.png")
        )


def _resize_for_fid(image_dir: str, target_size: int) -> str:
    """Make a temporary resized copy of the real images for fair FID."""
    resized = image_dir.rstrip("/\\") + "_resized"
    os.makedirs(resized, exist_ok=True)
    for name in os.listdir(image_dir):
        if name.lower().endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(os.path.join(image_dir, name))
            img.resize((target_size, target_size), Image.LANCZOS).save(
                os.path.join(resized, name)
            )
    return resized


def compute_fid(real_dir: str, generated_dir: str, image_size: int):
    """Compute FID between real and generated images via the ``pytorch_fid`` CLI."""
    real_resized = _resize_for_fid(real_dir, image_size)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytorch_fid", "--num-workers", "8",
             real_resized, generated_dir],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip().split()[-1])
    except subprocess.CalledProcessError as e:
        print("FID calculation error:", e.stderr)
        return None
    except Exception as e:  # noqa: BLE001
        print(f"Unexpected FID error: {e}")
        return None
    finally:
        shutil.rmtree(real_resized, ignore_errors=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Device: {device}")

    # DDIM sampler used ONLY for the cheap per-epoch FID generation.
    fid_scheduler = DDIMScheduler.from_pretrained("google/ddpm-celebahq-256")
    fid_scheduler.set_timesteps(num_inference_steps=args.fid_steps)

    dataloader = build_dataloader(args.dataset_name, args.image_size, args.batch_size)

    os.makedirs(args.save_dir, exist_ok=True)
    save_batch_preview(
        dataloader,
        os.path.join(
            args.save_dir,
            f"batch_preview_bs{args.batch_size}_size{args.image_size}.png",
        ),
    )

    # Load model (fresh or resumed).
    if args.resume_from:
        print(f"Resuming from: {args.resume_from}")
        image_pipe = DDPMPipeline.from_pretrained(args.resume_from).to(device)
        start_epoch = int(args.resume_from.split("_")[-1]) + 1
    else:
        print("Starting from google/ddpm-celebahq-256")
        image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256").to(device)
        start_epoch = 0

    optimizer = torch.optim.AdamW(image_pipe.unet.parameters(), lr=args.lr)

    fid_data = {
        "training_info": {
            "batch_size": args.batch_size,
            "num_epochs": "indefinite" if args.num_epochs is None else args.num_epochs,
            "learning_rate": args.lr,
            "image_size": args.image_size,
        },
        "fid_scores": {},
    }
    losses = []
    json_filename = (
        f"fid_scores_bs{args.batch_size}_lr{args.lr}_size{args.image_size}.json"
    )

    epoch = start_epoch
    while args.num_epochs is None or epoch < args.num_epochs:
        epoch_loss = 0.0
        for batch in tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch}"):
            clean = batch["images"].to(device)
            noise = torch.randn(clean.shape, device=device)
            bs = clean.shape[0]
            timesteps = torch.randint(
                0, image_pipe.scheduler.num_train_timesteps, (bs,), device=device
            ).long()

            noisy = image_pipe.scheduler.add_noise(clean, noise, timesteps)
            noise_pred = image_pipe.unet(noisy, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} average loss: {avg_loss:.6f}")

        # Save checkpoint for this epoch.
        ckpt_dir = os.path.join(args.save_dir, f"diffusion_epoch_{epoch}")
        image_pipe.save_pretrained(ckpt_dir)

        # Per-epoch FID for checkpoint selection.
        gen_dir = os.path.join(args.save_dir, f"generated_images_epoch_{epoch}")
        generate_samples_for_fid(
            image_pipe, fid_scheduler, args.fid_num_images,
            args.image_size, device, gen_dir,
        )
        fid = compute_fid(args.dataset_name, gen_dir, args.image_size)
        if fid is not None:
            fid_data["fid_scores"][epoch] = fid
            print(f"Epoch {epoch} FID: {fid:.4f}")
        else:
            print(f"Epoch {epoch} FID: N/A")
        with open(os.path.join(args.save_dir, json_filename), "w") as f:
            json.dump(fid_data, f, indent=4)

        print(f"  checkpoint: {ckpt_dir}")
        print("-" * 50)
        epoch += 1

    # Final model + loss curve.
    image_pipe.save_pretrained(os.path.join(args.save_dir, "diffusion_final"))
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(args.save_dir, "loss_curve.png"))
    plt.close()
    print(f"Done. Final model: {os.path.join(args.save_dir, 'diffusion_final')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DDPM fine-tuning from google/ddpm-celebahq-256 (diffusers).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset_name", required=True,
                   help="Path to a local image folder (HF datasets imagefolder).")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_epochs", type=int, default=None,
                   help="Default: run indefinitely (select best checkpoint by FID).")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--save_dir", type=str, default="model_checkpoints")
    p.add_argument("--resume_from", type=str, default=None,
                   help="Path to a 'diffusion_epoch_N' checkpoint to resume.")
    p.add_argument("--fid_steps", type=int, default=500,
                   help="DDIM steps for the per-epoch FID generation.")
    p.add_argument("--fid_num_images", type=int, default=20,
                   help="Number of images sampled for the per-epoch FID.")
    return p.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()