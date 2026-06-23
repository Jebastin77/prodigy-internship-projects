"""
U-Net Generator for Pix2Pix.

Architecture (matches report.md, Section 2.1):
    Encoder: 8 downsampling blocks   Conv(4x4, stride2) -> BatchNorm -> LeakyReLU
    Bottleneck: 1x1x512 (no BatchNorm)
    Decoder: 8 upsampling blocks     TransposedConv(4x4, stride2) -> BatchNorm -> ReLU
    Skip connections concatenate encoder features into the matching decoder layer.
    Output: TransposedConv -> Tanh, producing a 256x256x3 image in [-1, 1].
"""

import torch
import torch.nn as nn


class UNetDown(nn.Module):
    """Single encoder block: Conv -> (BatchNorm) -> LeakyReLU -> (Dropout)."""

    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not normalize)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Single decoder block: TransposedConv -> BatchNorm -> ReLU -> (Dropout), then skip-concat."""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        return torch.cat((x, skip_input), dim=1)


class GeneratorUNet(nn.Module):
    """
    U-Net generator for 256x256 RGB images.

    Channel progression (matches report.md table):
        in   ->  64  (256->128)
        64   -> 128  (128->64)
        128  -> 256  (64->32)
        256  -> 512  (32->16)
        512  -> 512  (16->8)
        512  -> 512  (8->4)
        512  -> 512  (4->2)
        512  -> 512  (2->1, bottleneck, no BN)
    Decoder mirrors this back up to 256x256x out_channels via Tanh.
    """

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        # Encoder (8 down blocks total, last one = bottleneck without BN)
        self.down1 = UNetDown(in_channels, 64, normalize=True)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512)
        self.down8 = UNetDown(512, 512, normalize=False)  # bottleneck, 1x1x512

        # Decoder (7 up blocks with skip connections + final output layer)
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)  # bottleneck

        # Decoder with skip connections
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


if __name__ == "__main__":
    # quick shape sanity check
    model = GeneratorUNet()
    dummy = torch.randn(1, 3, 256, 256)
    out = model(dummy)
    print("Input :", dummy.shape)
    print("Output:", out.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
