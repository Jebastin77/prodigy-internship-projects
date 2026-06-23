"""
Inference / demo script for Pix2Pix.

Usage (matches README.md):
    python source_code/inference.py \
        --checkpoint outputs/generator_epoch200.pth \
        --input path/to/input.png \
        --output screenshots/translated_output.png
"""

import argparse
import os
import sys

import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pix2pix import Pix2Pix
from utils.data_loader import denormalize


def parse_args():
    p = argparse.ArgumentParser(description="Run Pix2Pix inference on a single image.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to generator .pth checkpoint")
    p.add_argument("--input", type=str, required=True, help="Path to input image")
    p.add_argument("--output", type=str, default="screenshots/translated_output.png", help="Where to save result")
    p.add_argument("--image_size", type=int, default=256, help="Resize input to this size")
    p.add_argument("--device", type=str, default=None, help="cuda / cpu / mps (auto-detected if omitted)")
    p.add_argument("--side_by_side", action="store_true", help="Save input next to the output")
    return p.parse_args()


def get_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_input_image(path, image_size):
    img = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform(img).unsqueeze(0)


def run_inference(checkpoint_path, input_path, image_size=256, device=None):
    """Programmatic entry point reused by the web UI."""
    device = device or get_device()
    model = Pix2Pix(device=device)
    state = torch.load(checkpoint_path, map_location=device)
    model.generator.load_state_dict(state)
    model.generator.eval()

    real_x = load_input_image(input_path, image_size).to(device)
    with torch.no_grad():
        fake_y = model.generator(real_x)

    return denormalize(fake_y).clamp(0, 1), denormalize(real_x).clamp(0, 1)


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input image not found: {args.input}")

    fake_y, real_x = run_inference(args.checkpoint, args.input, args.image_size, device)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if args.side_by_side:
        grid = torch.cat([real_x, fake_y], dim=3)  # concat along width
        save_image(grid, args.output)
    else:
        save_image(fake_y, args.output)

    print(f"Saved translated output -> {args.output}")


if __name__ == "__main__":
    main()
