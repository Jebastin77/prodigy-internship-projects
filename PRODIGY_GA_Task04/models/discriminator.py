"""
PatchGAN Discriminator for Pix2Pix.

Per report.md Section 2.2:
    - Classifies overlapping 70x70 patches rather than the whole image.
    - Input: concatenation of source image x and (real or fake) target image y -> 6 channels.
    - Output: 30x30 grid of real/fake logits (averaged into a single scalar loss outside
      this module, typically via BCEWithLogitsLoss against an all-real / all-fake label map).
"""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN discriminator.

    Architecture: 4 strided conv blocks (Conv-BN-LeakyReLU) reducing spatial size,
    followed by a final 1-channel conv producing the patch-prediction grid.
    No sigmoid is applied here -- use BCEWithLogitsLoss for numerical stability.
    """

    def __init__(self, in_channels=3):
        super().__init__()

        def block(in_c, out_c, stride=2, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=not normalize)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Discriminator sees concatenated (source, target) -> in_channels * 2
        layers = []
        layers += block(in_channels * 2, 64, stride=2, normalize=False)  # 256 -> 128
        layers += block(64, 128, stride=2)                                # 128 -> 64
        layers += block(128, 256, stride=2)                               # 64  -> 32
        layers += block(256, 512, stride=1)                               # 32  -> 31  (stride 1 keeps receptive field at ~70x70)
        layers += [nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)]  # 31 -> 30, 1-channel patch logits

        self.model = nn.Sequential(*layers)

    def forward(self, img_x, img_y):
        """img_x: source/condition image, img_y: target (real or generated) image."""
        x = torch.cat((img_x, img_y), dim=1)
        return self.model(x)


if __name__ == "__main__":
    model = PatchGANDiscriminator()
    a = torch.randn(1, 3, 256, 256)
    b = torch.randn(1, 3, 256, 256)
    out = model(a, b)
    print("Output patch grid shape:", out.shape)  # expect (1, 1, 30, 30)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
