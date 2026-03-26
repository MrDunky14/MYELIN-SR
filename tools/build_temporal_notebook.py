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

import json
from pathlib import Path

def create_kaggle_notebook():
    cells = []
    
    def add_md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [text]})
        
    def add_code(text):
        # ensure lines end with \n
        lines = [line + '\n' for line in text.strip().split('\n')]
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines})

    add_md("# MYELIN-SR Phase 2: Temporal Training\n\nTrains MYELIN-Flow and TemporalFusion on the REDS dataset using the frozen Phase 1 spatial brain.")
    
    add_code("""
# SETUP AND IMPORTS
import os
import time
import math
import random
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

# Ensure reproducibility
torch.manual_seed(42)
random.seed(42)

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu_count = torch.cuda.device_count()
print(f"GPUs detected:   {gpu_count}")
""")

    add_md("## Core Architecture (From Phase 1)")
    
    # Read fpsan_conv2d.py
    fpsan_src = Path('train/fpsan_conv2d.py').read_text()
    add_code(fpsan_src)
    
    # Read model.py
    model_src = Path('train/model.py').read_text()
    add_code(model_src.replace("from .fpsan_conv2d import FPSANConv2d", ""))
    
    add_md("## Phase 2 Architecture: MYELIN-Flow & TemporalFusion")
    
    # Read flow_model.py
    flow_src = Path('train/flow_model.py').read_text()
    # Remove imports
    flow_src = '\n'.join([l for l in flow_src.split('\n') if not l.startswith('from ') and not l.startswith('import ')])
    add_code(flow_src)
    
    # Read temporal_model.py
    temp_model_src = Path('train/temporal_model.py').read_text()
    temp_model_src = '\n'.join([l for l in temp_model_src.split('\n') if not l.startswith('from ') and not l.startswith('import ')])
    add_code(temp_model_src)
    
    add_md("## Datasets & Losses")
    
    # Read temporal_loss.py
    loss_src = Path('train/temporal_loss.py').read_text()
    
    # Need to append Phase 1 losses since temporal_loss imports them
    phase1_loss = Path('train/losses.py').read_text()
    phase1_loss = '\n'.join([l for l in phase1_loss.split('\n') if not l.startswith('from ') and not l.startswith('import ')])
    
    loss_src = '\n'.join([l for l in loss_src.split('\n') if not l.startswith('from ') and not l.startswith('import ')])
    add_code(phase1_loss + '\n\n' + loss_src)
    
    # Read temporal_dataset.py
    data_src = Path('train/temporal_dataset.py').read_text()
    data_src = '\n'.join([l for l in data_src.split('\n') if not l.startswith('from ') and not l.startswith('import ')])
    add_code(data_src)
    
    add_md("## Training Loop")
    
    train_src = Path('train/train_temporal.py').read_text()
    # Strip imports at top roughly
    lines = train_src.split('\n')
    start_idx = 0
    for i, l in enumerate(lines):
        if l.startswith('def train_temporal('):
            start_idx = i
            break
    add_code('\n'.join(lines[start_idx:]))
    
    add_md("## Execution")
    
    add_code("""
# PATH CONFIGURATION
# 1. You must add the downloaded Phase 1 output (myelin_sr_deploy.bin or myelin_sr_best.pt) to Kaggle
# 2. You must add the REDS dataset to Kaggle

# Example paths (adjust based on your Kaggle dataset names):
SPATIAL_CKPT = "/kaggle/input/myelin-sr-phase1/myelin_sr_best.pt"
REDS_DIR = "/kaggle/input/reds-dataset"

# Wait for Kaggle to connect to the dataset
import time
while not Path(SPATIAL_CKPT).exists() or not Path(REDS_DIR).exists():
    print("Waiting for paths to become available...")
    print(f"Checking {SPATIAL_CKPT}")
    print(f"Checking {REDS_DIR}")
    time.sleep(5)

CONFIG = {
    'scale': 2,
    'epochs': 150,
    'batch_size': 16, # Adjust based on 16GB T4 VRAM
    'patch_size': 128, 
    'num_workers': 4,
    'base_lr': 2e-4,
    'flow_radius': 4,
    'spatial_weight': 1.0,
    'temporal_weight': 0.5,
    'flow_weight': 1.0,
    'spatial_ckpt': SPATIAL_CKPT,
    'data_dir': REDS_DIR,
}

print("Starting Temporal Pipeline...")
train_temporal(CONFIG)
""")

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open('train/myelin_temporal.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
        
    print("Successfully generated train/myelin_temporal.ipynb")

if __name__ == "__main__":
    create_kaggle_notebook()
