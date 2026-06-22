"""
image_gen_project/generator/views.py
Handles the home page (GET) and image generation (POST + AJAX).
"""

import sys
import json
import logging
from pathlib import Path

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

# Add source_code to path so we can import image_generator
SOURCE_DIR = Path(__file__).resolve().parent.parent.parent / "source_code"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

logger = logging.getLogger(__name__)

# Load sample prompts from prompts.txt
PROMPTS_FILE = Path(__file__).resolve().parent.parent.parent / "prompts" / "prompts.txt"


def _load_sample_prompts():
    if not PROMPTS_FILE.exists():
        return []
    lines = PROMPTS_FILE.read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


def index(request):
    """Render the main UI page."""
    sample_prompts = _load_sample_prompts()
    return render(request, "generator/index.html", {"sample_prompts": sample_prompts})


@csrf_exempt
@require_http_methods(["POST"])
def generate(request):
    """
    AJAX endpoint – receives JSON body:
        { "prompt": "...", "steps": 25, "guidance": 7.5,
          "width": 512, "height": 512, "seed": null }
    Returns JSON with image URL and metadata.
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    prompt = data.get("prompt", "").strip()
    if not prompt:
        return JsonResponse({"error": "Prompt is required."}, status=400)

    steps = int(data.get("steps", 25))
    guidance = float(data.get("guidance", 7.5))
    width = int(data.get("width", 512))
    height = int(data.get("height", 512))
    seed = data.get("seed")
    if seed is not None:
        seed = int(seed)

    # Clamp values to safe ranges
    steps = max(10, min(steps, 50))
    guidance = max(1.0, min(guidance, 20.0))
    width = max(256, min(width, 768))
    height = max(256, min(height, 768))

    try:
        from image_generator import generate_image
        result = generate_image(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            seed=seed,
        )
    except Exception as exc:
        logger.exception("Image generation failed")
        return JsonResponse({"error": str(exc)}, status=500)

    image_url = f"/images/{result['filename']}"
    return JsonResponse({
        "image_url": image_url,
        "filename": result["filename"],
        "optimized_prompt": result["optimized_prompt"],
        "seed": result["seed"],
        "elapsed": result["elapsed"],
    })
