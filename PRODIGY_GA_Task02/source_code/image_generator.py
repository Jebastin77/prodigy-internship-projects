"""
source_code/image_generator.py
Core image generation logic using Stable Diffusion via HuggingFace Diffusers.
Supports CPU (float32) and GPU (float16) inference automatically.
"""

import os
import time
import uuid
import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

logger = logging.getLogger(__name__)

# ── Model configuration ───────────────────────────────────────────────────────
MODEL_ID = "runwayml/stable-diffusion-v1-5"   # ~4 GB download on first run
SAVE_DIR = Path(__file__).resolve().parent.parent / "images"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Prompt optimisation helpers ───────────────────────────────────────────────
QUALITY_BOOSTERS = (
    "ultra-detailed, high resolution, 8K, sharp focus, "
    "professional photography, award-winning"
)

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, ugly, watermark, "
    "text, signature, nsfw, oversaturated, bad anatomy"
)


def optimize_prompt(raw_prompt: str) -> str:
    """
    Append quality-booster keywords to the user's raw prompt
    if they are not already present.
    """
    if "ultra-detailed" not in raw_prompt.lower():
        return f"{raw_prompt.strip()}, {QUALITY_BOOSTERS}"
    return raw_prompt.strip()


# ── Pipeline singleton ────────────────────────────────────────────────────────
_pipeline: Optional[StableDiffusionPipeline] = None


def _get_pipeline() -> StableDiffusionPipeline:
    """Load the pipeline once and reuse it across requests."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    logger.info("Loading Stable Diffusion pipeline on %s …", device)
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,          # disable for local / research use
        requires_safety_checker=False,
    )

    # Faster scheduler (DPM-Solver++ with 20 steps ≈ quality of DDIM 50 steps)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    if device == "cpu":
        # Reduce memory pressure on CPU-only machines
        pipe.enable_attention_slicing()

    _pipeline = pipe
    logger.info("Pipeline ready.")
    return _pipeline


# ── Public API ────────────────────────────────────────────────────────────────
def generate_image(
    prompt: str,
    negative_prompt: str = NEGATIVE_PROMPT,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    seed: Optional[int] = None,
) -> dict:
    """
    Generate an image from *prompt* and save it to SAVE_DIR.

    Returns
    -------
    dict with keys:
        filename   – saved file name (relative to SAVE_DIR)
        filepath   – absolute path
        optimized_prompt – the prompt that was actually sent to the model
        seed       – RNG seed used
        elapsed    – generation time in seconds
    """
    optimized = optimize_prompt(prompt)

    generator = None
    if seed is None:
        seed = int(time.time()) % (2 ** 32)
    generator = torch.Generator().manual_seed(seed)

    pipe = _get_pipeline()

    t0 = time.time()
    result = pipe(
        prompt=optimized,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    )
    elapsed = round(time.time() - t0, 2)

    image: Image.Image = result.images[0]
    filename = f"generated_{uuid.uuid4().hex[:8]}.png"
    filepath = SAVE_DIR / filename
    image.save(filepath)

    logger.info("Image saved → %s  (%.1fs)", filepath, elapsed)
    return {
        "filename": filename,
        "filepath": str(filepath),
        "optimized_prompt": optimized,
        "seed": seed,
        "elapsed": elapsed,
    }
