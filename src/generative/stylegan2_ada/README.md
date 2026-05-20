# StyleGAN2-ADA

StyleGAN2-ADA synthetic image generation for the nail melanoma study
(paper Methods S2). Models are trained separately for NM and NUM, plus a
combined model used for sequential NM->NUM interpolation.

## Important: external dependency (not redistributed)

This module builds on **NVIDIA's official StyleGAN2-ADA (PyTorch)**, which
is released under the NVIDIA Source Code License (research / non-commercial).
That code is **not** redistributed here. Clone it separately:

```bash
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
```

`dataset_tool.py`, `train.py`, and `generate.py` are NVIDIA's — run them
from inside that checkout. Only `interpolate_images.py` in this folder is
ours; copy it into the checkout before running step d.

## Files

| File | Origin | Purpose |
|------|--------|---------|
| `interpolate_images.py` | ours (adapted from NVIDIA `gen_video.py`) | Sequential latent interpolation -> frames |
| `dataset_tool.py` | NVIDIA (clone) | Images -> TFRecords |
| `train.py` | NVIDIA (clone) | Train StyleGAN2-ADA |
| `generate.py` | NVIDIA (clone) | Bulk image generation |

Helper shell scripts live in `scripts/` (`02a`–`02d`); run them from inside
the NVIDIA checkout.

## Attribution (`interpolate_images.py`)

Adapted from NVIDIA StyleGAN2-ADA's `gen_video.py`: instead of rendering an
interpolation video, it saves the intermediate frames as individual images.
The NVIDIA copyright header is retained. It imports `dnnlib` and `legacy`,
so it must sit inside the StyleGAN2-ADA checkout to run.

## Configuration (Methods S2)

- Implementation: official StyleGAN2-ADA PyTorch
- `--cfg=stylegan2` (Adam, lr 0.002, R1 gamma 10); batch 16; 256x256; 1 GPU
- Max training length: 25,000 kimg (NVIDIA default)
- Trained separately for NM and NUM; a third model on the combined NM+NUM
  set (NUM amelanotic cases excluded) for sequential morphing
- FID: NVIDIA's built-in metrics (`fid50k_full` during training, or
  `calc_metrics.py` post-hoc)
- Checkpoint selection: four lowest-FID candidates -> board-certified
  dermatologist review -> single final checkpoint

## Pipeline

```bash
# Copy interpolate_images.py into your stylegan2-ada-pytorch checkout, then
# run the scripts from there.

# a. prepare TFRecords (NM + NUM)
DATASET_ROOT=./dataset bash 02a_stylegan_prepare.sh

# b. train (NM + NUM; combined model separately, see comments in 02b)
bash 02b_stylegan_train.sh

# c. BULK generation (2000 images by default, adjustable)
NETWORK_PKL=output/.../network-snapshot-XXXXXX.pkl \
  NUM_IMAGES=2000 bash 02c_stylegan_generate.sh

# d. SEQUENTIAL morph (separate from bulk generation)
NETWORK_PKL=output/<combined-model>/network-snapshot-XXXXXX.pkl \
  SEED1=18 SEED2=13 NUM_STEPS=10 bash 02d_stylegan_interpolate.sh
```
