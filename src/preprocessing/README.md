# Preprocessing

Data preprocessing pipeline for the nail melanoma study, corresponding to
**Methods S1** of the paper. The pipeline mixes manual steps, external
tools, and two small scripts provided here.

## Pipeline overview

```
raw clinical photo
   │
   ├─ 1. Center crop (1:1)            ── manual
   ├─ 2. Artifact removal (LaMa)      ── manual (web tool)
   ├─ 3. Up-scaling (Real-ESRGAN)     ── external executable
   ├─ 4. RGB unification              ── convert_to_rgb.py   (this repo)
   └─ 5. Color augmentation           ── AWB (external) + color_jitter.py (this repo)
            │
            ▼
   training-ready dataset  ──►  StyleGAN2-ADA  +  classifiers
```

Only steps 4 and 5 (Color Jitter) are scripted here; the others are
manual or use external tools, documented below.

---

## 1. Center crop (manual)

Images were manually center-cropped to a 1:1 aspect ratio so the nail
unit is centered. No script — performed in an image editor.

## 2. Artifact removal — LaMa (manual)

Artifacts such as markers, rulers, text overlays, and blank spaces were
manually removed using large-mask inpainting (**LaMa**).

- Tool: <https://github.com/advimman/lama> (a hosted demo was used)
- This step is manual; no script is provided.

## 3. Up-scaling — Real-ESRGAN (external executable)

Images with a resolution below 256×256 were upscaled using
**Real-ESRGAN**:

- **4× scaling** for images under 100×100 pixels
- **3× scaling** for all other sub-256×256 images

The portable `realesrgan-ncnn-vulkan` executable was used (no CUDA /
PyTorch environment required).

```bash
# Single image
./realesrgan-ncnn-vulkan -i input.jpg -o output.png -s 4

# Folder
./realesrgan-ncnn-vulkan -i input_folder -o output_folder -s 3 -f png
```

- Tool: <https://github.com/xinntao/Real-ESRGAN>
- Paper: <https://arxiv.org/abs/2107.10833>

## 4. RGB unification (`convert_to_rgb.py`)

Normalises every image to 3-channel RGB PNG. Required because
StyleGAN2-ADA's `dataset_tool.py` and the classifiers expect consistent
RGB input.

```bash
python -m src.preprocessing.convert_to_rgb /path/to/input /path/to/output_rgb
```

## 5. Color augmentation

Two color-augmentation techniques — **Auto White Balance (AWB)** and
**Color Jitter** — were applied to the training set alongside the
unmodified images, to improve robustness to lighting and imaging
conditions. The exact class-specific combination strategy is described
in the paper (Methods S1).

### 5a. Auto White Balance (external)

AWB was applied with the **Deep White-Balance Editing** PyTorch
implementation (Afifi & Brown, CVPR 2020).

```bash
# From within the Deep_White_Balance/PyTorch folder:
python demo_images.py --input_dir /path/to/input_rgb --output_dir /path/to/AWB --task AWB
```

- Tool: <https://github.com/mahmoudnafifi/Deep_White_Balance>
- **License: research purposes only; not for commercial use.**

### 5b. Color Jitter (`color_jitter.py`)

Brightness, contrast, and saturation are each randomly sampled within a
factor of (0.8, 1.2); hue is unchanged.

```bash
python -m src.preprocessing.color_jitter /path/to/input_rgb /path/to/jitter
# reproducible:
python -m src.preprocessing.color_jitter /path/to/input_rgb /path/to/jitter --seed 42
```

---

## External tools summary

| Step | Tool | License | Note |
|------|------|---------|------|
| Artifact removal | LaMa | Apache-2.0 | Manual / hosted demo |
| Up-scaling | Real-ESRGAN | BSD-3-Clause | Portable ncnn-vulkan executable |
| Auto White Balance | Deep White-Balance Editing | Research-only (non-commercial) | Separate clone required |

These tools are **not** redistributed in this repository; install them
separately from the links above.