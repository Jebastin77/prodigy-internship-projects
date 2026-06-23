"""
Utility helpers: saving sample grids, checkpointing, simple metrics (PSNR/MAE),
and loss-curve plotting. Kept dependency-light (numpy, torch, matplotlib, skimage).
"""

import os
import json

import numpy as np
import torch
from torchvision.utils import make_grid, save_image

from utils.data_loader import denormalize


def save_sample_grid(real_x, real_y, fake_y, out_path, nrow=None):
    """Save a [source | generated | target] comparison grid for the first N items in a batch."""
    n = real_x.size(0) if nrow is None else min(nrow, real_x.size(0))
    imgs = torch.cat([real_x[:n], fake_y[:n], real_y[:n]], dim=0)
    imgs = denormalize(imgs).clamp(0, 1)
    grid = make_grid(imgs, nrow=n)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(grid, out_path)


def save_checkpoint(model, optimizer_g, optimizer_d, epoch, ckpt_dir, tag=None):
    """
    Save full optimizer + model state for resumable training, per report.md
    Section 6: 'Checkpoint system: saves every 50 epochs with full optimizer state.'
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    suffix = tag if tag is not None else f"epoch{epoch}"

    torch.save(model.generator.state_dict(), os.path.join(ckpt_dir, f"generator_{suffix}.pth"))
    torch.save(model.discriminator.state_dict(), os.path.join(ckpt_dir, f"discriminator_{suffix}.pth"))
    torch.save(
        {
            "epoch": epoch,
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
        },
        os.path.join(ckpt_dir, f"training_state_{suffix}.pth"),
    )


def load_generator_checkpoint(model, checkpoint_path, device="cpu"):
    state = torch.load(checkpoint_path, map_location=device)
    model.generator.load_state_dict(state)
    model.generator.eval()
    return model


def compute_psnr(fake, real, max_val=1.0):
    """PSNR between two tensors already in [0, 1]."""
    mse = torch.mean((fake - real) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10((max_val ** 2) / mse)


def compute_mae(fake, real):
    return torch.mean(torch.abs(fake - real)).item()


class LossTracker:
    """Accumulates per-epoch losses and writes them to JSON + a plotted PNG curve."""

    def __init__(self):
        self.history = {"loss_G": [], "loss_D": [], "loss_G_L1": []}

    def update(self, g_metrics, d_metrics):
        self.history["loss_G"].append(g_metrics["loss_G"])
        self.history["loss_D"].append(d_metrics["loss_D"])
        self.history["loss_G_L1"].append(g_metrics["loss_G_L1"])

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "loss_history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 5))
            plt.plot(self.history["loss_G"], label="Generator")
            plt.plot(self.history["loss_D"], label="Discriminator")
            plt.plot(self.history["loss_G_L1"], label="L1 (x100 in total loss)")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Pix2Pix Training Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "loss_curve.png"))
            plt.close()
        except ImportError:
            pass
