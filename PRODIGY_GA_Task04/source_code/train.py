"""
Training script for Pix2Pix.

Usage (matches README.md):
    python source_code/train.py \
        --data_dir dataset/train \
        --epochs 200 \
        --batch_size 1 \
        --lr 0.0002 \
        --output_dir outputs
"""

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pix2pix import Pix2Pix
from utils.data_loader import PairedImageDataset
from utils.helper_functions import save_sample_grid, save_checkpoint, LossTracker


def parse_args():
    p = argparse.ArgumentParser(description="Train a Pix2Pix cGAN.")
    p.add_argument("--data_dir", type=str, default="dataset/train", help="Path to training data")
    p.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    p.add_argument("--batch_size", type=int, default=1, help="Batch size")
    p.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    p.add_argument("--lambda_l1", type=float, default=100.0, help="L1 loss weight")
    p.add_argument("--image_size", type=int, default=256, help="Resize images to this size")
    p.add_argument("--save_interval", type=int, default=10, help="Save sample output every N epochs")
    p.add_argument("--checkpoint_interval", type=int, default=50, help="Save model checkpoint every N epochs")
    p.add_argument("--decay_epoch", type=int, default=100, help="Epoch to start linear LR decay")
    p.add_argument("--direction", type=str, default="AtoB", choices=["AtoB", "BtoA"], help="Translation direction")
    p.add_argument("--output_dir", type=str, default="outputs", help="Where to save sample grids/curves")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Where to save model checkpoints")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    p.add_argument("--device", type=str, default=None, help="cuda / cpu / mps (auto-detected if omitted)")
    return p.parse_args()


def get_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_lambda(epoch, total_epochs, decay_epoch):
    """Linear decay to 0 after `decay_epoch`, matching report.md training table."""
    if epoch < decay_epoch:
        return 1.0
    return max(0.0, 1.0 - (epoch - decay_epoch) / max(1, (total_epochs - decay_epoch)))


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    dataset = PairedImageDataset(args.data_dir, image_size=args.image_size, mode=args.direction, augment=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print(f"Loaded {len(dataset)} paired training images from {args.data_dir}")

    model = Pix2Pix(lambda_l1=args.lambda_l1, device=device)

    opt_g = torch.optim.Adam(model.generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    sched_g = LambdaLR(opt_g, lr_lambda=lambda e: lr_lambda(e, args.epochs, args.decay_epoch))
    sched_d = LambdaLR(opt_d, lr_lambda=lambda e: lr_lambda(e, args.epochs, args.decay_epoch))

    tracker = LossTracker()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    fixed_batch = next(iter(loader))  # used for consistent sample previews

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        epoch_g_loss, epoch_d_loss = 0.0, 0.0

        for real_x, real_y in loader:
            real_x, real_y = real_x.to(device), real_y.to(device)

            # --- Train Generator ---
            opt_g.zero_grad()
            loss_g, g_metrics = model.generator_loss(real_x, real_y)
            loss_g.backward()
            opt_g.step()

            # --- Train Discriminator ---
            with torch.no_grad():
                fake_y = model.generator(real_x)
            opt_d.zero_grad()
            loss_d, d_metrics = model.discriminator_loss(real_x, real_y, fake_y)
            loss_d.backward()
            opt_d.step()

            tracker.update(g_metrics, d_metrics)
            epoch_g_loss += g_metrics["loss_G"]
            epoch_d_loss += d_metrics["loss_D"]

        sched_g.step()
        sched_d.step()

        avg_g = epoch_g_loss / len(loader)
        avg_d = epoch_d_loss / len(loader)
        elapsed = time.time() - start
        print(f"[Epoch {epoch}/{args.epochs}] loss_G={avg_g:.4f} loss_D={avg_d:.4f} ({elapsed:.1f}s)")

        if epoch % args.save_interval == 0 or epoch == args.epochs:
            with torch.no_grad():
                fx, fy = fixed_batch[0].to(device), fixed_batch[1].to(device)
                fake = model.translate(fx)
            out_path = os.path.join(args.output_dir, f"sample_epoch{epoch:04d}.png")
            save_sample_grid(fx, fy, fake, out_path)
            tracker.save(args.output_dir)
            print(f"  saved sample grid -> {out_path}")

        if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
            save_checkpoint(model, opt_g, opt_d, epoch, args.checkpoint_dir)
            print(f"  saved checkpoint at epoch {epoch}")

    print("Training complete.")


if __name__ == "__main__":
    main()
