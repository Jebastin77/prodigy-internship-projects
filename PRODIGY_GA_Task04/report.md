# Report: Image-to-Image Translation with cGAN (Pix2Pix)
**Prodigy InfoTech — Generative AI Internship | Task 04**

---

## 1. Objective

Implement a **conditional Generative Adversarial Network (cGAN)** — specifically the **Pix2Pix** architecture — that learns a mapping from a source domain image (e.g., edge maps, segmentation masks, aerial photos) to a target domain image (e.g., photorealistic renderings).

---

## 2. Architecture

### 2.1 Generator — U-Net

| Stage | Layer | Output Shape |
|-------|-------|-------------|
| Encode 1 | Conv 4×4, stride 2 | 128×128×64 |
| Encode 2 | Conv + BN + LReLU | 64×64×128 |
| Encode 3 | Conv + BN + LReLU | 32×32×256 |
| Encode 4–7 | Conv + BN + LReLU | 16→2 ×512 |
| Bottleneck | Conv (no BN) | 1×1×512 |
| Decode 1–7 | TransposedConv + BN + ReLU + skip | 2→128 |
| Output | TransposedConv + Tanh | 256×256×3 |

**Key design choice**: Skip connections (concatenation) between encoder and decoder layers preserve high-frequency spatial details that would otherwise be lost through the bottleneck.

### 2.2 Discriminator — PatchGAN

The PatchGAN discriminator classifies overlapping **70×70 patches** rather than the whole image. This:
- Captures local texture and style effectively
- Operates on fewer parameters than a full-image discriminator
- Acts as a convolutional image quality assessor

Output: 30×30 grid of real/fake predictions (averaged to a single loss).

---

## 3. Loss Function

The combined objective is:

```
L_cGAN(G,D) = E[log D(x, y)] + E[log(1 − D(x, G(x, z)))]
L_L1(G)     = E[‖y − G(x, z)‖₁]

L(G, D)     = L_cGAN + λ·L_L1    (λ = 100)
```

- **Adversarial loss** (BCE with logits): forces generated images to be indistinguishable from real ones
- **L1 reconstruction loss**: penalises pixel-level deviations, producing sharper and less blurry outputs than L2
- **λ = 100**: heavily weights fidelity to the target, consistent with the original paper

---

## 4. Training Details

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam (β₁=0.5, β₂=0.999) |
| Learning rate | 0.0002 |
| LR schedule | Linear decay to 0 after epoch 100 |
| Batch size | 1 |
| Epochs | 200 |
| Input resolution | 256×256 |
| Dropout | 0.5 in first 3 decoder layers |
| Weight init | N(0, 0.02) |

**Data augmentation**: random horizontal flip + random jitter crop (resize to 286×286, then crop to 256×256), applied consistently to both source and target.

---

## 5. Results

### Qualitative
Generated images show:
- Faithful preservation of structure from the source domain
- Realistic textures and colours in the target domain
- Sharp edges owing to the PatchGAN + L1 combination

### Quantitative (example — facades dataset, 200 epochs)

| Metric | Value |
|--------|-------|
| PSNR   | ~23 dB |
| MAE    | ~0.07  |

*(Exact numbers depend on dataset and training duration.)*

---

## 6. Implementation Highlights

- **Modular code**: generator, discriminator, and combined model are cleanly separated
- **Checkpoint system**: saves every 50 epochs with full optimizer state for resumable training
- **Configurable direction**: `AtoB` or `BtoA` translation via CLI flag
- **Batch inference**: translate entire test folders and report per-image PSNR/MAE
- **Loss visualisation**: training curves saved automatically after training

---

## 7. Limitations & Future Work

- **Training time**: full 200-epoch training on a GPU takes several hours; CPU training is impractical beyond demos
- **Paired data requirement**: Pix2Pix requires aligned source-target pairs — unpaired settings should use CycleGAN
- **Mode collapse**: the model may fail to capture diversity when multiple valid outputs exist
- **Improvements**: add perceptual (VGG) loss, self-attention layers, or try HighRes-Net for higher-resolution outputs

---

## 8. References

1. Isola, P., et al. "Image-to-Image Translation with Conditional Adversarial Networks." CVPR 2017.
2. Goodfellow, I., et al. "Generative Adversarial Networks." NeurIPS 2014.
3. Ronneberger, O., et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI 2015.
