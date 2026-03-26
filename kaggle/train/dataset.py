# MYELIN-SR v2: Zero-Barrier Ternary Reconstruction Engine
# Copyright (C) 2026 Krishna Singh
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Dataset loader for MYELIN-SR training.

Supports:
  - DIV2K (800 training images, 100 validation)
  - Flickr2K (2650 images)
  - Standard test sets: Set5, Set14, BSD100, Urban100, Manga109

All datasets use HR images downscaled via bicubic interpolation to create LR pairs.
Training uses random crops + augmentation. Validation uses center crops or full images.
"""

import os
import random
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image


class SRDataset(Dataset):
    """
    Super-Resolution paired dataset.
    
    Loads high-resolution images, creates low-resolution inputs via bicubic downscale.
    Training mode: random crop + augmentation.
    Validation mode: center crop or full image.
    """

    def __init__(
        self,
        hr_dir: str,
        scale: int = 2,
        patch_size: int = 256,
        is_train: bool = True,
        max_images: Optional[int] = None,
    ):
        super().__init__()
        self.hr_dir = Path(hr_dir)
        self.scale = scale
        self.patch_size = patch_size  # HR patch size
        self.is_train = is_train

        # Collect all image files
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        self.image_paths = sorted([
            p for p in self.hr_dir.iterdir()
            if p.suffix.lower() in extensions
        ])

        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {hr_dir}")

        print(f"[Dataset] Loaded {len(self.image_paths)} images from {hr_dir} ({'train' if is_train else 'val'})")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _random_crop_pair(
        self, hr_img: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Random crop HR patch, then downscale to create LR patch."""
        w, h = hr_img.size
        ps = self.patch_size

        # Ensure image is large enough for the requested patch
        if w < ps or h < ps:
            hr_img = hr_img.resize(
                (max(w, ps), max(h, ps)), Image.BICUBIC
            )
            w, h = hr_img.size

        # Random crop coordinates
        x = random.randint(0, w - ps)
        y = random.randint(0, h - ps)
        hr_patch = hr_img.crop((x, y, x + ps, y + ps))

        # Create LR by downscaling
        lr_size = ps // self.scale
        lr_patch = hr_patch.resize((lr_size, lr_size), Image.BICUBIC)

        return lr_patch, hr_patch

    def _center_crop_pair(
        self, hr_img: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Center crop for deterministic validation."""
        w, h = hr_img.size
        ps = self.patch_size

        # Ensure divisible by scale for clean downscale
        crop_w = (min(w, ps) // self.scale) * self.scale
        crop_h = (min(h, ps) // self.scale) * self.scale

        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        hr_patch = hr_img.crop((left, top, left + crop_w, top + crop_h))

        lr_size_w = crop_w // self.scale
        lr_size_h = crop_h // self.scale
        lr_patch = hr_patch.resize((lr_size_w, lr_size_h), Image.BICUBIC)

        return lr_patch, hr_patch

    @staticmethod
    def _augment(lr_img: Image.Image, hr_img: Image.Image):
        """Random horizontal flip + rotation (0, 90, 180, 270)."""
        # Random horizontal flip
        if random.random() > 0.5:
            lr_img = lr_img.transpose(Image.FLIP_LEFT_RIGHT)
            hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)

        # Random rotation (0, 90, 180, 270)
        rot = random.choice([0, 90, 180, 270])
        if rot != 0:
            lr_img = lr_img.rotate(rot, expand=True)
            hr_img = hr_img.rotate(rot, expand=True)

        return lr_img, hr_img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr_img = Image.open(self.image_paths[idx]).convert("RGB")

        if self.is_train:
            lr_img, hr_img = self._random_crop_pair(hr_img)
            lr_img, hr_img = self._augment(lr_img, hr_img)
        else:
            lr_img, hr_img = self._center_crop_pair(hr_img)

        # Convert to tensors [0, 1]
        to_tensor = transforms.ToTensor()
        lr_tensor = to_tensor(lr_img)
        hr_tensor = to_tensor(hr_img)

        return lr_tensor, hr_tensor


class SRFullImageDataset(Dataset):
    """
    Full-image dataset for benchmark evaluation (no cropping).
    Returns full LR/HR pairs with sizes divisible by the scale factor.
    """

    def __init__(self, hr_dir: str, scale: int = 2):
        super().__init__()
        self.hr_dir = Path(hr_dir)
        self.scale = scale

        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        self.image_paths = sorted([
            p for p in self.hr_dir.iterdir()
            if p.suffix.lower() in extensions
        ])

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {hr_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        hr_img = Image.open(self.image_paths[idx]).convert("RGB")
        w, h = hr_img.size

        # Make divisible by scale
        w = (w // self.scale) * self.scale
        h = (h // self.scale) * self.scale
        hr_img = hr_img.crop((0, 0, w, h))

        lr_img = hr_img.resize((w // self.scale, h // self.scale), Image.BICUBIC)

        to_tensor = transforms.ToTensor()
        return to_tensor(lr_img), to_tensor(hr_img), self.image_paths[idx].name


def build_train_loader(
    data_root: str,
    scale: int = 2,
    patch_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 4,
) -> DataLoader:
    """
    Build training dataloader from DIV2K + optional Flickr2K.
    
    Expected directory structure at data_root:
      data_root/
        DIV2K_train_HR/       # 800 images (0001.png - 0800.png)
        Flickr2K_HR/          # 2650 images (optional)
    """
    datasets = []

    div2k_path = Path(data_root) / "DIV2K_train_HR"
    if div2k_path.exists():
        datasets.append(SRDataset(str(div2k_path), scale, patch_size, is_train=True))
    else:
        print(f"[WARNING] DIV2K not found at {div2k_path}. Download from https://data.vision.ee.ethz.ch/cvl/DIV2K/")

    flickr2k_path = Path(data_root) / "Flickr2K_HR"
    if flickr2k_path.exists():
        datasets.append(SRDataset(str(flickr2k_path), scale, patch_size, is_train=True))

    if not datasets:
        raise FileNotFoundError(
            f"No training data found in {data_root}. "
            "Please download DIV2K_train_HR and optionally Flickr2K_HR."
        )

    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def build_val_loader(
    data_root: str,
    dataset_name: str = "Set14",
    scale: int = 2,
) -> DataLoader:
    """
    Build validation dataloader for benchmark test sets.
    
    Expected: data_root/Set14/, data_root/Set5/, etc.
    """
    val_path = Path(data_root) / dataset_name
    if not val_path.exists():
        raise FileNotFoundError(f"Validation set not found at {val_path}")

    return DataLoader(
        SRFullImageDataset(str(val_path), scale),
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )
