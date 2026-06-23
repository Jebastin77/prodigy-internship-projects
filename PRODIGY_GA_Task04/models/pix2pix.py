"""
Combined Pix2Pix model: wires the Generator and Discriminator together with
their respective losses, following report.md Section 3:

    L_cGAN(G, D) = E[log D(x, y)] + E[log(1 - D(x, G(x)))]
    L_L1(G)      = E[ || y - G(x) ||_1 ]
    L(G, D)      = L_cGAN + lambda * L_L1     (lambda = 100)

Implemented with BCEWithLogitsLoss (numerically stable sigmoid + BCE) since the
discriminator outputs raw logits.
"""

import torch
import torch.nn as nn

from models.generator import GeneratorUNet
from models.discriminator import PatchGANDiscriminator


def weights_init_normal(m):
    """Initialize Conv/BatchNorm weights ~ N(0, 0.02), per report.md training table."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class Pix2Pix(nn.Module):
    """
    Convenience wrapper bundling generator + discriminator + loss functions.
    Training loop logic itself lives in source_code/train.py; this class exposes
    the building blocks and loss computation so both train.py and inference.py
    (and the web UI) share one source of truth.
    """

    def __init__(self, in_channels=3, out_channels=3, lambda_l1=100.0, device="cpu"):
        super().__init__()
        self.device = device
        self.lambda_l1 = lambda_l1

        self.generator = GeneratorUNet(in_channels, out_channels).to(device)
        self.discriminator = PatchGANDiscriminator(in_channels).to(device)

        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()

    def generator_loss(self, real_x, real_y):
        """Generator loss = adversarial (fool D) + lambda * L1(real_y, fake_y)."""
        fake_y = self.generator(real_x)
        pred_fake = self.discriminator(real_x, fake_y)
        valid = torch.ones_like(pred_fake, device=self.device)

        loss_gan = self.criterion_gan(pred_fake, valid)
        loss_l1 = self.criterion_l1(fake_y, real_y)
        loss_g = loss_gan + self.lambda_l1 * loss_l1
        return loss_g, {"loss_G": loss_g.item(), "loss_G_GAN": loss_gan.item(), "loss_G_L1": loss_l1.item()}

    def discriminator_loss(self, real_x, real_y, fake_y):
        """Discriminator loss = BCE(D(x,y),1) + BCE(D(x,G(x)),0)."""
        pred_real = self.discriminator(real_x, real_y)
        valid = torch.ones_like(pred_real, device=self.device)
        loss_real = self.criterion_gan(pred_real, valid)

        pred_fake = self.discriminator(real_x, fake_y.detach())
        fake = torch.zeros_like(pred_fake, device=self.device)
        loss_fake = self.criterion_gan(pred_fake, fake)

        loss_d = 0.5 * (loss_real + loss_fake)
        return loss_d, {"loss_D": loss_d.item(), "loss_D_real": loss_real.item(), "loss_D_fake": loss_fake.item()}

    @torch.no_grad()
    def translate(self, real_x):
        """Run the generator only (inference mode)."""
        self.generator.eval()
        out = self.generator(real_x)
        self.generator.train()
        return out
