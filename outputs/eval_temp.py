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

with open('outputs/training_log_r2.json', 'r') as f:
    log = json.load(f)

latest = log[-1]
best = max((e for e in log if 'val_psnr' in e), key=lambda x: x['val_psnr'])

print("--- FINAL EPOCH 300 ---")
print(f"Loss: {latest['loss']:.4f}")
print(f"PSNR: {latest['val_psnr']:.2f}")
print(f"SSIM: {latest['val_ssim']:.4f}")

print("\n--- BEST CHECKPOINT ---")
print(f"Epoch: {best['epoch']}")
print(f"PSNR: {best['val_psnr']:.2f} dB")
print(f"SSIM: {best['val_ssim']:.4f}")
print(f"LR: {best['lr']:.2e}")
