"""
Dataset loading & augmentation for paired image-to-image translation.

Expects each image in the dataset directory to be a single file containing
the SOURCE and TARGET side-by-side (width = 2 * height), e.g. 512x256,
which is the standard Pix2Pix paired-data convention (facades, maps, edges2shoes).
"""

import os
import random

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


class PairedImageDataset(Dataset):
    """
    Loads side-by-side paired images and splits them into (source, target).

    Args:
        root_dir: folder containing paired images.
        image_size: final square size each half is resized to (default 256).
        mode: "AtoB" or "BtoA" -- which half is treated as the conditioning input.
        augment: enable random horizontal flip + random jitter crop, as
                 described in report.md Section 4 (resize to 286, crop 256).
    """

    def __init__(self, root_dir, image_size=256, mode="AtoB", augment=True):
        self.root_dir = root_dir
        self.image_size = image_size
        self.mode = mode
        self.augment = augment

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

        self.files = sorted(
            f for f in os.listdir(root_dir) if f.lower().endswith(IMG_EXTENSIONS)
        )
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # -> [-1, 1]
        ])

    def __len__(self):
        return len(self.files)

    def _split_pair(self, img):
        w, h = img.size
        half = w // 2
        left = img.crop((0, 0, half, h))
        right = img.crop((half, 0, w, h))
        return left, right

    def _augment_pair(self, src, tgt):
        jitter = int(self.image_size * 1.12)  # ~286 when image_size=256
        src = TF.resize(src, [jitter, jitter])
        tgt = TF.resize(tgt, [jitter, jitter])

        # random crop (same crop box for both images)
        max_x = jitter - self.image_size
        max_y = jitter - self.image_size
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        src = TF.crop(src, y, x, self.image_size, self.image_size)
        tgt = TF.crop(tgt, y, x, self.image_size, self.image_size)

        if random.random() > 0.5:
            src = TF.hflip(src)
            tgt = TF.hflip(tgt)

        return src, tgt

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.files[idx])
        img = Image.open(path).convert("RGB")
        left, right = self._split_pair(img)

        src, tgt = (left, right) if self.mode == "AtoB" else (right, left)

        if self.augment:
            src, tgt = self._augment_pair(src, tgt)
        else:
            src = TF.resize(src, [self.image_size, self.image_size])
            tgt = TF.resize(tgt, [self.image_size, self.image_size])

        return self.to_tensor(src), self.to_tensor(tgt)


def denormalize(tensor):
    """Map a [-1, 1] tensor back to [0, 1] for saving/displaying."""
    return (tensor * 0.5) + 0.5
