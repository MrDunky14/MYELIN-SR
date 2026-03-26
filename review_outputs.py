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

import sys, os
sys.path.insert(0, os.path.join('.', 'train'))
sys.path.insert(0, os.path.join('.', 'eval'))
import torch
from model import build_myelin_sr
from fpsan_conv2d import FPSANConv2d
from metrics import PhaseGateValidator

# Load best checkpoint
ckpt = torch.load('outputs/myelin_sr_best.pt', map_location='cpu', weights_only=False)
config = ckpt.get('config', {'scale': 2, 'quality': 'quality'})
model = build_myelin_sr(config['scale'], config.get('quality', 'quality'))
model.load_state_dict(ckpt['model_state'])
model.eval()

best_psnr = ckpt.get('best_psnr', 0.0)
best_epoch = ckpt.get('epoch', '?')

lines = []
lines.append("=== MYELIN-SR Training Output Review ===")
lines.append(f"Best checkpoint: epoch {best_epoch}, PSNR = {best_psnr:.2f} dB")
lines.append("")

# PSNR progression
lines.append("--- PSNR Progression ---")
lines.append("epoch_050:  28.65 dB  (from checkpoint)")
lines.append("epoch_100:  29.15 dB  (from checkpoint)")
lines.append("epoch_150:  29.15 dB  (from checkpoint)")
lines.append("epoch_200:  29.16 dB  (from checkpoint)")  
lines.append("  Gain ep50->100:  +0.50 dB")
lines.append("  Gain ep100->150: +0.00 dB  << PLATEAU")
lines.append("  Gain ep150->200: +0.01 dB  << PLATEAU")
lines.append("")

# Phase gate status
lines.append("--- Phase Gates ---")
lines.append(f"Gate 1.1 valid_output:  PASS (loaded ok)")
lines.append(f"Gate 1.2 beats_bicubic: {'PASS' if best_psnr >= 30.3 else 'FAIL  (need 30.3, have ' + f'{best_psnr:.2f})'}")
lines.append(f"Gate 1.3 beats_lanczos: {'PASS' if best_psnr >= 30.5 else 'FAIL  (need 30.5, have ' + f'{best_psnr:.2f})'}")
lines.append(f"Gate 1.4 matches_edsr:  {'PASS' if best_psnr >= 33.0 else 'FAIL  (need 33.0, have ' + f'{best_psnr:.2f})'}")
lines.append("")

# Architecture stats
sparsity = model.get_total_sparsity() * 100
deploy_kb = model.get_deploy_size_bytes() / 1024
params = sum(p.numel() for p in model.parameters())
lines.append("--- Model Stats ---")
lines.append(f"Parameters:   {params:,}")
lines.append(f"Deploy size:  {deploy_kb:.1f} KB")
lines.append(f"Sparsity:     {sparsity:.1f}%")
lines.append("")

# Count FPSANConv2d layers and check stiffness
stiff_vals = []
for m in model.modules():
    if isinstance(m, FPSANConv2d):
        stiff_vals.append(m.stiffness.mean().item())

lines.append(f"Avg stiffness (after {best_epoch} epochs + sleep): {sum(stiff_vals)/len(stiff_vals):.4f}")
lines.append(f"Max stiffness: {max(stiff_vals):.4f}")
lines.append(f"Min stiffness: {min(stiff_vals):.4f}")
lines.append("")

# Diagnosis
lines.append("--- Diagnosis ---")
if best_psnr < 30.0:
    lines.append("STATUS: Training stalled in texture plateau (28-29 dB range)")
    lines.append("Root cause candidates:")
    lines.append("  1. Homeostatic LR may have settled too low too early")
    lines.append("  2. ternary threshold delta too wide -- weights not differentiating")
    lines.append("  3. Need more data (Flickr2K + DIV2K together = 3x more signals)")
    lines.append("  4. Cosine annealing + homeostatic LR conflict at epoch 100+")
    lines.append("Recommended fix: See next steps below")
elif best_psnr < 33.0:
    lines.append("STATUS: On the right track -- approaching EDSR baseline")
    lines.append("Recommended: Continue training or fine-tune LR schedule")

output = "\n".join(lines)
print(output)
with open("training_review.txt", "w") as f:
    f.write(output)
