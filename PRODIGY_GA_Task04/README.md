# PRODIGY_GA_Task04 — Image-to-Image Translation with cGAN (Pix2Pix)

## Overview

This project implements **Pix2Pix**, a conditional Generative Adversarial Network (cGAN) for paired image-to-image translation. Given a source image (e.g., a sketch or segmentation map), the model learns to generate a realistic corresponding output image.

It now also ships a **web UI** (`webapp/`) for uploading an image, running it through the generator, and visualizing the PatchGAN discriminator's patch-by-patch verdict.

---

## Project Structure

```
PRODIGY_GA_Task04/
├── README.md
├── report.md
├── requirements.txt
│
├── dataset/
│   ├── train/          ← paired training images (side-by-side: input|target)
│   ├── test/            ← paired test images
│   └── sample_data/     ← small sample for quick demos
│
├── models/
│   ├── generator.py     ← U-Net Generator
│   ├── discriminator.py ← PatchGAN Discriminator
│   └── pix2pix.py       ← Combined cGAN model (losses, weight init)
│
├── utils/
│   ├── data_loader.py        ← Dataset loading & augmentation
│   └── helper_functions.py   ← Checkpointing, metrics, loss plotting
│
├── outputs/              ← Training progress images
├── checkpoints/          ← Saved generator/discriminator weights
├── screenshots/          ← Sample inputs & translated outputs
│
├── source_code/
│   ├── train.py          ← Training script
│   └── inference.py      ← Inference / demo script (CLI)
│
└── webapp/                ← Web UI
    ├── app.py              ← Flask backend (upload, inference, patch grid)
    ├── requirements.txt
    ├── templates/
    │   └── index.html
    └── static/
        ├── style.css
        ├── app.js
        └── samples/        ← example gallery thumbnails
```

---

## Architecture

### Generator — U-Net
- **Encoder**: 8 stages of Conv → BatchNorm → LeakyReLU that downsample 256×256 down to a 1×1×512 bottleneck
- **Decoder**: 7 stages of TransposedConv → BatchNorm → ReLU that upsample back to 256×256, plus a final TransposedConv → Tanh
- **Skip Connections**: Concatenate encoder feature maps to the matching decoder layer to preserve fine details

### Discriminator — PatchGAN
- Classifies overlapping **70×70 patches** rather than the whole image
- Produces a **30×30 grid** of real/fake logits — effective at capturing local texture details
- Receptive field and output-grid size are derived directly from the conv stride/kernel configuration (verified: 5 conv layers, strides `[2,2,2,1,1]`, kernel 4 → 70×70 receptive field, 30×30 output)

### Loss Functions
| Loss | Formula | Purpose |
|------|---------|---------|
| Adversarial | BCE(D(x,G(x,z)), 1) | Generator fools discriminator |
| L1 Reconstruction | λ × ‖y − G(x,z)‖₁ | Pixel-level similarity (λ=100) |
| Discriminator | BCE(D(x,y),1) + BCE(D(x,G(x,z)),0) | Distinguish real vs fake |

---

## Setup

```bash
pip install -r requirements.txt
```

### Dataset
Place paired images (source|target side-by-side, 512×256) in `dataset/train/` and `dataset/test/`.

Popular datasets: [facades](http://cmp.felk.cvut.cz/~tylecr1/facade/), [maps](https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/), [edges2shoes](https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/).

---

## Training

```bash
python source_code/train.py \
    --data_dir dataset/train \
    --epochs 200 \
    --batch_size 1 \
    --lr 0.0002 \
    --output_dir outputs
```

### Key Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `dataset/train` | Path to training data |
| `--epochs` | `200` | Number of epochs |
| `--batch_size` | `1` | Batch size |
| `--lr` | `0.0002` | Learning rate |
| `--lambda_l1` | `100` | L1 loss weight |
| `--image_size` | `256` | Resize images to this size |
| `--save_interval` | `10` | Save sample grid every N epochs |
| `--checkpoint_interval` | `50` | Save model checkpoint every N epochs |
| `--decay_epoch` | `100` | Epoch to start linear LR decay |
| `--direction` | `AtoB` | Translation direction (`AtoB` or `BtoA`) |

Checkpoints are written to `checkpoints/generator_epoch<N>.pth` (and matching discriminator/optimizer files), and the web UI automatically picks up the latest one on startup.

---

## Inference (CLI)

```bash
python source_code/inference.py \
    --checkpoint checkpoints/generator_epoch200.pth \
    --input path/to/input.png \
    --output screenshots/translated_output.png
```

---

## Web UI

A Flask-based interface for interactively translating images and visualizing the discriminator's patch grid.

```bash
pip install -r requirements.txt
pip install -r webapp/requirements.txt
python webapp/app.py
```

Then open **http://localhost:5000**.

**How it behaves:**
- **If a trained checkpoint exists** in `checkpoints/` (any file named `generator_*.pth`), the UI loads it automatically and runs real generator inference. The status pill at the top turns green and shows the checkpoint name + device.
- **If no checkpoint is found**, the UI runs in **demo mode**: it applies a lightweight classical image-processing stand-in (edge detection / segmentation-style quantization / day-to-night recolor) so the interface is fully explorable without requiring a multi-hour training run first. This is clearly labeled in the UI (amber "demo mode" status pill and tag on the result).
- Either way, the result panel renders a **30×30 patch-grid overlay** echoing the PatchGAN discriminator's real/fake judgement across the image, toggleable on/off.

This separation means you can demo the full UX immediately, then drop in a real `checkpoints/generator_epoch*.pth` from `train.py` later and the same interface starts using it with no code changes.

---

## Results

Training outputs are saved to `outputs/` every `--save_interval` epochs. Final translated samples appear in `screenshots/`. Loss curves and a JSON history are written to `outputs/loss_curve.png` and `outputs/loss_history.json`.

---

## References

- Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks" (CVPR 2017)
- [Official Pix2Pix Repo](https://github.com/phillipi/pix2pix)
- [PyTorch Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

---

*Internship Task — Prodigy InfoTech | Generative AI Track*
