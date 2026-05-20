# DDPM (Denoising Diffusion Probabilistic Model)

DDPM-based synthetic image generation for the nail melanoma study,
corresponding to **Methods S2** of the paper. Models are fine-tuned from
the pretrained `google/ddpm-celebahq-256` checkpoint, independently for
NM and NUM, using HuggingFace `diffusers`.

## Attribution

The fine-tuning and sampling code is **adapted from the HuggingFace
Diffusion Models Course, Unit 2** ("Fine-Tuning and Guidance"), licensed
under Apache-2.0:
<https://huggingface.co/learn/diffusion-course/en/unit2/2>

Additions for this study: FID-based per-epoch checkpoint selection,
resume support, JSON logging of training/FID history, an argparse CLI,
and the dataset-specific fine-tuning configuration.

## Files

| File | Purpose |
|------|---------|
| `finetune.py` | Fine-tune the DDPM; logs per-epoch FID for checkpoint selection. |
| `generate.py` | Sample images from a saved checkpoint. |

## Configuration (Methods S2)

- Base checkpoint: `google/ddpm-celebahq-256`
- Optimizer: AdamW; learning rate: 1e-5; batch size: 16; image size: 256
- Trained separately for NM and NUM
- Checkpoint selection: four lowest-FID candidates → board-certified
  dermatologist visual review → single final checkpoint

## 1. Fine-tune

```bash
python -m src.generative.ddpm.finetune \
    --dataset_name /path/to/NM_train_images \
    --batch_size 16 --lr 1e-5 --image_size 256 \
    --save_dir runs_ddpm/nm

# resume from a saved epoch:
python -m src.generative.ddpm.finetune \
    --dataset_name /path/to/NM_train_images \
    --save_dir runs_ddpm/nm \
    --resume_from runs_ddpm/nm/diffusion_epoch_19
```

Per-epoch checkpoints (`diffusion_epoch_N/`) and a `fid_scores_*.json`
log are written to `--save_dir`. Pick the checkpoint with the best FID
and visual quality.

## 2. Generate

```bash
python -m src.generative.ddpm.generate \
    --model_path runs_ddpm/nm/diffusion_epoch_37 \
    --num_images 2000 --image_size 256 --steps 1000 \
    --output_dir fake/nm
```

`--steps` defaults to **1000** to reproduce the paper (Methods S2). It is
user-adjustable — DDIM allows fewer steps for faster sampling at some
cost to image quality. (Each per-class model generated 2,000 images;
2,000 NM + 2,000 NUM = the 4,000 "DIFF4000" augmentation set.)

## Dependencies

- `diffusers` (Apache-2.0)
- `datasets`
- `pytorch-fid` (FID-based checkpoint selection)

## Weights

Pretrained weights are **not** redistributed in this repository;
fine-tuning starts from the public `google/ddpm-celebahq-256` checkpoint.
Fine-tuned checkpoints are available from the corresponding author on
reasonable request.
