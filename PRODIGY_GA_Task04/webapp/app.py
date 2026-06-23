"""
Flask web UI for the Pix2Pix project.

Provides:
  - Drag/drop image upload
  - Run inference with a trained generator checkpoint (if one is provided)
  - "Demo mode" fallback: if no checkpoint is loaded, applies a lightweight
    classical image-processing pipeline (edge/segmentation-style stylization)
    so the UI is fully explorable without requiring a multi-hour GPU training run.
  - Live PatchGAN-style patch grid overlay for visualization
  - Sample gallery of pre-baked example translations

Run:
    python webapp/app.py
Then open http://localhost:5000
"""

import os
import sys
import io
import base64
import uuid

from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB

ALLOWED_EXT = {"png", "jpg", "jpeg", "webp", "bmp"}

# ---------------------------------------------------------------------------
# Model loading (real Pix2Pix generator if torch + a checkpoint are available)
# ---------------------------------------------------------------------------
MODEL_STATE = {"loaded": False, "model": None, "device": None, "checkpoint_name": None}


def try_load_model():
    """
    Attempt to load the real trained GeneratorUNet. Falls back gracefully
    (MODEL_STATE['loaded'] stays False) if torch is missing or no checkpoint
    exists yet -- this lets the UI run in classical-CV 'demo mode' out of the box.
    """
    try:
        import torch
        from models.pix2pix import Pix2Pix
    except ImportError:
        return

    ckpt_dir = os.path.join(os.path.dirname(BASE_DIR), "checkpoints")
    candidate = None
    if os.path.isdir(ckpt_dir):
        for fname in sorted(os.listdir(ckpt_dir)):
            if fname.startswith("generator_") and fname.endswith(".pth"):
                candidate = os.path.join(ckpt_dir, fname)

    if candidate is None:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Pix2Pix(device=device)
    state = torch.load(candidate, map_location=device)
    model.generator.load_state_dict(state)
    model.generator.eval()

    MODEL_STATE.update(loaded=True, model=model, device=device, checkpoint_name=os.path.basename(candidate))


try_load_model()


# ---------------------------------------------------------------------------
# Demo-mode fallback translation (no trained weights required)
# ---------------------------------------------------------------------------
def demo_translate(pil_img, style="edges"):
    """
    A deterministic, dependency-light stand-in for the trained generator,
    used only when no checkpoint is available. Mimics common Pix2Pix demo
    directions (edge map -> photo look, segmentation -> stylized) using
    classical filters, purely so the UI has something visual to show.
    This is clearly labeled as 'Demo mode' in the API response and UI.
    """
    img = pil_img.convert("RGB").resize((256, 256))

    if style == "edges":
        gray = ImageOps.grayscale(img)
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges = ImageOps.invert(edges)
        edges = ImageEnhance.Contrast(edges).enhance(1.6)
        return edges.convert("RGB")

    if style == "segmentation":
        arr = np.array(img).astype(np.float32)
        quant = (arr // 48 * 48).astype(np.uint8)
        seg = Image.fromarray(quant)
        seg = seg.filter(ImageFilter.SMOOTH_MORE)
        return seg

    if style == "night":
        arr = np.array(img).astype(np.float32)
        arr[..., 0] *= 0.55  # R
        arr[..., 1] *= 0.65  # G
        arr[..., 2] *= 1.15  # B
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        night = Image.fromarray(arr)
        night = ImageEnhance.Brightness(night).enhance(0.6)
        night = ImageEnhance.Contrast(night).enhance(1.2)
        return night

    return img


def compute_patch_scores(pil_img, grid=30):
    """
    Lightweight stand-in for PatchGAN's 30x30 'real/fake' patch grid:
    scores each patch by local gradient energy as a proxy for 'how
    plausible/textured' the patch looks, purely for the visualization
    overlay (not a real discriminator forward pass unless a trained
    model is loaded).
    """
    gray = np.array(ImageOps.grayscale(pil_img.resize((256, 256))), dtype=np.float32)
    gy, gx = np.gradient(gray)
    energy = np.sqrt(gx ** 2 + gy ** 2)

    h, w = energy.shape
    step_h, step_w = h // grid, w // grid
    scores = np.zeros((grid, grid), dtype=np.float32)
    for i in range(grid):
        for j in range(grid):
            patch = energy[i * step_h:(i + 1) * step_h, j * step_w:(j + 1) * step_w]
            scores[i, j] = patch.mean() if patch.size else 0.0

    if scores.max() > 0:
        scores = scores / scores.max()
    return scores.tolist()


def pil_to_base64(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", model_loaded=MODEL_STATE["loaded"],
                            checkpoint_name=MODEL_STATE["checkpoint_name"])


@app.route("/api/status")
def status():
    return jsonify({
        "model_loaded": MODEL_STATE["loaded"],
        "checkpoint_name": MODEL_STATE["checkpoint_name"],
        "device": str(MODEL_STATE["device"]) if MODEL_STATE["device"] else None,
    })


@app.route("/api/translate", methods=["POST"])
def translate():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Unsupported or missing file."}), 400

    style = request.form.get("style", "edges")

    try:
        pil_img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not read image file."}), 400

    uid = uuid.uuid4().hex[:10]
    input_resized = pil_img.resize((256, 256))

    used_real_model = False
    if MODEL_STATE["loaded"]:
        try:
            import torch
            import torchvision.transforms as T
            from utils.data_loader import denormalize

            transform = T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ])
            x = transform(pil_img).unsqueeze(0).to(MODEL_STATE["device"])
            with torch.no_grad():
                fake = MODEL_STATE["model"].generator(x)
            fake = denormalize(fake).clamp(0, 1).squeeze(0).cpu().numpy()
            fake = (np.transpose(fake, (1, 2, 0)) * 255).astype(np.uint8)
            output_img = Image.fromarray(fake)
            used_real_model = True
        except Exception as e:
            print(f"Model inference failed, falling back to demo mode: {e}")
            output_img = demo_translate(pil_img, style=style)
    else:
        output_img = demo_translate(pil_img, style=style)

    patch_scores = compute_patch_scores(output_img)

    in_path = os.path.join(OUTPUT_DIR, f"{uid}_input.png")
    out_path = os.path.join(OUTPUT_DIR, f"{uid}_output.png")
    input_resized.save(in_path)
    output_img.save(out_path)

    return jsonify({
        "input_image": pil_to_base64(input_resized),
        "output_image": pil_to_base64(output_img),
        "patch_scores": patch_scores,
        "used_real_model": used_real_model,
        "mode": "trained-model" if used_real_model else "demo-mode",
        "style": style,
    })


@app.route("/api/samples")
def samples():
    """Returns metadata for the static example gallery shown on first load."""
    return jsonify({
        "samples": [
            {"id": "facade", "label": "Facade \u2192 Building", "thumb": "/static/samples/facade.png"},
            {"id": "map", "label": "Aerial \u2192 Map", "thumb": "/static/samples/map.png"},
            {"id": "shoe", "label": "Edges \u2192 Shoe", "thumb": "/static/samples/shoe.png"},
        ]
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
