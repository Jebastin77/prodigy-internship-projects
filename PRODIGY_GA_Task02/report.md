# Task Report — PRODIGY_GA_Task02
## Image Generation with Pre-trained Models

**Intern:** iebastin30200
**Task:** Utilize pre-trained generative models to create images from text prompts.

---

## 1. Objective

Build a locally-runnable web application that allows a user to type a natural-language description and receive a high-quality AI-generated image, powered by Stable Diffusion v1.5.

---

## 2. Model Selected

| Attribute | Detail |
|-----------|--------|
| Model | Stable Diffusion v1.5 |
| Source | `runwayml/stable-diffusion-v1-5` (HuggingFace) |
| Scheduler | DPM-Solver++ (20–25 steps ≈ DDIM 50 steps quality) |
| Precision | float16 on GPU · float32 on CPU |
| Licence | CreativeML Open RAIL-M |

Stable Diffusion was chosen over DALL-E-mini because it:
- Runs fully offline after the initial weight download.
- Produces significantly higher resolution and detail (512 px default).
- Offers fine-grained control through negative prompts, guidance scale, and seeds.

---

## 3. System Architecture

```
Browser  ──POST /generate/──▶  Django View  ──▶  image_generator.py
                                                       │
                                                  SD Pipeline
                                                  (diffusers)
                                                       │
                                              images/generated_*.png
                                                       │
Browser  ◀── JSON { image_url, meta } ──────────────────
```

---

## 4. Prompt Optimisation Strategy

Raw user prompts are automatically enhanced with quality boosters:

```
ultra-detailed, high resolution, 8K, sharp focus,
professional photography, award-winning
```

A configurable negative prompt suppresses common artefacts:

```
blurry, low quality, distorted, deformed, ugly, watermark, text, nsfw
```

This two-part approach consistently improves output quality without requiring
the user to learn prompt engineering syntax.

---

## 5. Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `num_inference_steps` | 25 | Denoising iterations — quality vs speed trade-off |
| `guidance_scale` | 7.5 | Prompt adherence — higher = more literal |
| `width × height` | 512 × 512 | Output resolution |
| `seed` | random | Reproducibility |

---

## 6. Results

Two sample images are stored in `images/` demonstrating:

- **image1.png** — A photorealistic landscape prompt.
- **image2.png** — A stylised portrait prompt.

Generation times observed:
- GPU (RTX 3060, CUDA): ~8–15 seconds per image.
- CPU only: ~90–180 seconds per image.

---

## 7. Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Long CPU inference | DPM-Solver++ scheduler halves steps needed |
| Large model size (~4 GB) | Lazy-load pipeline singleton; cache via HuggingFace |
| CSRF in AJAX calls | `@csrf_exempt` on generate endpoint (local dev only) |
| Prompt quality variance | Automatic quality-booster injection |

---

## 8. Conclusion

The project successfully demonstrates text-to-image generation using a
pre-trained diffusion model served through a Django web interface. The UI
provides intuitive controls for prompt entry, parameter tuning, and image
download, meeting all Task 02 requirements.
