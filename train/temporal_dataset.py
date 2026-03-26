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
REDS (REalistic and Diverse Scenes) Dataset Loader for Phase 2 Temporal Training.

Unlike DIV2K which is static images, REDS is video sequences.
We must load pairs of consecutive frames: (Frame T-1, Frame T).
"""

import os
from pathlib import Path
import random
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class REDSSequenceDataset(Dataset):
    """
    Loads consecutive frame pairs (T-1, T) from the REDS dataset.
    Provides data augmentation (flips, rotations, crops) consistently across pairs
    to ensure optical flow vectors remain valid during training.
    """
    def __init__(
        self, 
        root_dir: str, 
        split: str = 'train', 
        scale: int = 2, 
        patch_size: int = 128
    ):
        self.root = Path(root_dir) / split
        self.hr_dir = self.root / 'train_sharp'
        self.scale = scale
        self.patch_size = patch_size
        
        # REDS typically provides X4 bicubic. We will generate our own X2
        # dynamically or use the provided X4 if scale == 4.
        
        self.sequences = []
        
        # Each folder under train_sharp is a sequence (000 to 239 for train)
        if self.hr_dir.exists():
            for seq_folder in sorted(os.listdir(self.hr_dir)):
                seq_path = self.hr_dir / seq_folder
                if not seq_path.is_dir(): continue
                
                # Frames are numbered 00000000.png to 00000099.png
                frames = sorted([f for f in os.listdir(seq_path) if f.endswith('.png')])
                
                # Create overlapping pairs: (0,1), (1,2), (2,3)...
                for i in range(len(frames) - 1):
                    self.sequences.append({
                        'seq': seq_folder,
                        't_minus_1': frames[i],
                        't': frames[i+1]
                    })

    def __len__(self):
        return len(self.sequences)

    def _load_img(self, path) -> torch.Tensor:
        img = Image.open(path).convert('RGB')
        return TF.to_tensor(img)

    def __getitem__(self, idx):
        item = self.sequences[idx]
        seq_folder = item['seq']
        
        hr_path_prev = self.hr_dir / seq_folder / item['t_minus_1']
        hr_path_cur  = self.hr_dir / seq_folder / item['t']
        
        hr_prev = self._load_img(hr_path_prev)
        hr_cur  = self._load_img(hr_path_cur)
        
        # Consistent Random Cropping across the sequence
        _, H, W = hr_prev.shape
        # Ensure patch is an even multiple of scale
        hr_patch = self.patch_size * self.scale
        
        if H > hr_patch and W > hr_patch:
            top  = random.randint(0, H - hr_patch)
            left = random.randint(0, W - hr_patch)
            
            hr_prev = TF.crop(hr_prev, top, left, hr_patch, hr_patch)
            hr_cur  = TF.crop(hr_cur,  top, left, hr_patch, hr_patch)
        else:
            # Center crop if too small
            hr_prev = TF.center_crop(hr_prev, [hr_patch, hr_patch])
            hr_cur  = TF.center_crop(hr_cur,  [hr_patch, hr_patch])
            
        # Create LR images via bicubic downsampling
        # This guarantees perfect alignment for flow training
        lr_size = [self.patch_size, self.patch_size]
        lr_prev = F.interpolate(hr_prev.unsqueeze(0), size=lr_size, mode='bicubic', align_corners=False).squeeze(0)
        lr_cur  = F.interpolate(hr_cur.unsqueeze(0),  size=lr_size, mode='bicubic', align_corners=False).squeeze(0)
        
        # Consistent Data Augmentation (H-flip, V-flip)
        if random.random() < 0.5:
            hr_prev, hr_cur = TF.hflip(hr_prev), TF.hflip(hr_cur)
            lr_prev, lr_cur = TF.hflip(lr_prev), TF.hflip(lr_cur)
        
        if random.random() < 0.5:
            hr_prev, hr_cur = TF.vflip(hr_prev), TF.vflip(hr_cur)
            lr_prev, lr_cur = TF.vflip(lr_prev), TF.vflip(lr_cur)
            
        # Return as two tuples: (LR_prev, LR_cur) and (HR_prev, HR_cur)
        return torch.stack([lr_prev, lr_cur]), torch.stack([hr_prev, hr_cur])

def build_temporal_loader(root_dir: str, scale: int, patch_size: int, batch_size: int, num_workers: int = 4):
    dataset = REDSSequenceDataset(root_dir, scale=scale, patch_size=patch_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
