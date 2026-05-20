# Generative AI-Augmented Deep Learning for Distinguishing Nail Unit Melanoma from Nail Melanonychia

Code for the study *"Generative AI-Augmented Deep Learning for Distinguishing
Nail Unit Melanoma from Nail Melanonychia: A Multicenter Study with Human-AI
Collaboration Analysis."*

The task is binary classification of nail clinical photographs as **nail unit
melanoma (NUM)** or **nail melanonychia (NM)**. Because NUM is comparatively
rare, the training set is expanded with synthetic images from two generative
models, and four classification backbones are evaluated with and without this
augmentation. This repository provides the computational pipeline:
**preprocessing → generative modeling → classification**.

> The full study also includes a human–AI reader study and explainability
> analyses. Those produce paper figures only and are not part of this code
> release.

## Pipeline overview

```
clinical photos
   │
   ▼
1. Preprocessing      crop · artifact removal · up-scaling · RGB · color aug
   │
   ▼
2. Generative models  StyleGAN2-ADA  +  DDPM   →  synthetic NM / NUM images
   │
   ▼
3. Classification     ResNet18 · EfficientNet-b0 · ViT · Swin
                      (original / +GAN / +DIFF), 10-fold CV, top-3 ensemble
```

## Repository structure

```
.
├── LICENSE                 Apache-2.0
├── NOTICE                  third-party component notices
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── scripts/                pipeline entry points (run in order)
│   ├── 01_preprocess.sh
│   ├── 02a_stylegan_prepare.sh
│   ├── 02b_stylegan_train.sh
│   ├── 02c_stylegan_generate.sh      # bulk generation
│   ├── 02d_stylegan_interpolate.sh   # sequential morph
│   ├── 03a_ddpm_finetune.sh
│   ├── 03b_ddpm_generate.sh
│   └── 04_train_classifiers.sh
└── src/
    ├── preprocessing/      convert_to_rgb.py, color_jitter.py, README.md
    ├── generative/
    │   ├── stylegan2_ada/  interpolate_images.py, README.md
    │   └── ddpm/           finetune.py, generate.py, README.md
    └── classification/     models, dataset, training, ensemble, metrics, reporting
```

Each module has its own `README.md` with the relevant configuration and
commands. Start there for module-specific details.

## Installation

The pipeline uses **two separate environments**.

**1. Main environment** — preprocessing, DDPM, and classification:

```bash
conda env create -f environment.yml
conda activate nail-melanoma
# or, with pip:
pip install -r requirements.txt
```

Install `torch`/`torchvision` matching your CUDA from
[pytorch.org](https://pytorch.org); this study used torch 2.9.0.

**2. StyleGAN2-ADA environment** — a *separate* environment is required, since
NVIDIA's StyleGAN2-ADA needs an older PyTorch/CUDA stack. See
[`src/generative/stylegan2_ada/README.md`](src/generative/stylegan2_ada/README.md).

**External tools** (installed separately, not redistributed here): LaMa,
Real-ESRGAN, and Deep White-Balance for preprocessing; NVIDIA StyleGAN2-ADA for
generation. Links and licenses are in each module README and in `NOTICE`.

## Usage

Run the scripts from the repository root, in pipeline order.

**1. Preprocessing.** Some steps are manual or use external tools; the script
runs the scripted parts (RGB conversion, Color Jitter) and documents the rest.

```bash
bash scripts/01_preprocess.sh
```

**2a. Generative — StyleGAN2-ADA** (from inside the NVIDIA checkout):

```bash
bash scripts/02a_stylegan_prepare.sh      # images -> TFRecords
bash scripts/02b_stylegan_train.sh        # train NM, NUM (and combined) models
NETWORK_PKL=... bash scripts/02c_stylegan_generate.sh    # bulk synthetic images
NETWORK_PKL=... bash scripts/02d_stylegan_interpolate.sh # sequential morph
```

**2b. Generative — DDPM:**

```bash
bash scripts/03a_ddpm_finetune.sh         # fine-tune NM, NUM models
NM_CKPT=... NUM_CKPT=... bash scripts/03b_ddpm_generate.sh
```

**3. Classification:**

```bash
DATA_ROOT=/path/to/data bash scripts/04_train_classifiers.sh
```

Folder names in the scripts are placeholders — edit the variables at the top of
each script (or pass them as environment variables) to match your layout.

### Label convention

The classification test loader reads the label from the filename prefix:
`0_*.png` for NM and `1_*.png` for NUM.

## Data and weights availability

- **Synthetic images:** a representative subset of synthetic images is made
  publicly available to support reproducibility (see the paper).
- **Patient data:** the clinical image dataset is not publicly shared due to
  patient privacy.
- **Trained weights:** generative and classification checkpoints are available
  from the corresponding author on reasonable request. Training from scratch is
  fully reproducible with the code and configurations here.

## License

Released under the **Apache License 2.0** (see [`LICENSE`](LICENSE)).

Some components are governed by their own licenses (see [`NOTICE`](NOTICE)):
`src/generative/stylegan2_ada/interpolate_images.py` is adapted from NVIDIA
StyleGAN2-ADA and remains under NVIDIA's license; the DDPM code is adapted from
the HuggingFace Diffusion Models Course (Apache-2.0). Because some external
dependencies are restricted to non-commercial use, the pipeline as a whole
should be treated as **research / non-commercial**.

## Citation

If you use this code, please cite the repository (see [`CITATION.cff`](CITATION.cff))
and the associated paper.