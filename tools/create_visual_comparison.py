import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

REPO_DIR = Path("C:/Users/Krishna/Documents/mygithub/MYELIN-SR")
DENSE_DIR = REPO_DIR / "outputs/trained_dense/dx12_bundle"
TERN_DIR = REPO_DIR / "outputs/trained_ternary/dx12_bundle"
OUT_DIR = REPO_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

lr_img = Image.open(DENSE_DIR / "sample_lr.png").convert("RGB")
gt_img = Image.open(DENSE_DIR / "sample_gt.png").convert("RGB")
w, h = gt_img.size

# Baselines
bic_img = lr_img.resize((w, h), Image.BICUBIC)
lan_img = lr_img.resize((w, h), Image.LANCZOS)

# GPU outputs
tern_gpu_img = Image.open(TERN_DIR / "gpu_ternary_fp16.png").convert("RGB")
dense_gpu_img = Image.open(DENSE_DIR / "gpu_dense_fp16.png").convert("RGB")

def rgb_to_y(arr_01):
    return 16.0 + 65.481 * arr_01[..., 0] + 128.553 * arr_01[..., 1] + 24.966 * arr_01[..., 2]

def get_metrics(pred_im, gt_im, shave=2):
    p = np.array(pred_im).astype(np.float32) / 255.0
    g = np.array(gt_im).astype(np.float32) / 255.0
    if shave > 0:
        p = p[shave:-shave, shave:-shave]
        g = g[shave:-shave, shave:-shave]
    py = rgb_to_y(p)
    gy = rgb_to_y(g)
    mse = float(np.mean((py - gy) ** 2))
    psnr = 100.0 if mse < 1e-10 else 20.0 * math.log10(255.0 / math.sqrt(mse))
    
    # SSIM
    pt = torch.from_numpy(py).float()[None, None]
    gt_t = torch.from_numpy(gy).float()[None, None]
    coords = torch.arange(11, dtype=torch.float32) - 5.0
    k1d = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    k1d /= k1d.sum()
    win = (k1d[:, None] * k1d[None, :])[None, None]
    mu1 = F.conv2d(pt, win)
    mu2 = F.conv2d(gt_t, win)
    s1 = F.conv2d(pt * pt, win) - mu1 * mu1
    s2 = F.conv2d(gt_t * gt_t, win) - mu2 * mu2
    s12 = F.conv2d(pt * gt_t, win) - mu1 * mu2
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim = float((((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2))).mean().item())
    return psnr, ssim

methods = [
    ("Bicubic", bic_img),
    ("Lanczos", lan_img),
    ("MYELIN-SR Ternary (DX12)", tern_gpu_img),
    ("MYELIN-SR Dense FP16 (DX12)", dense_gpu_img),
    ("Ground Truth", gt_img),
]

print("Metrics on sample image (butterfly.png):")
for name, im in methods:
    if name == "Ground Truth":
        print(f"  {name:28s}: GT Reference")
    else:
        psnr, ssim = get_metrics(im, gt_img)
        print(f"  {name:28s}: {psnr:.2f} dB (Y) | {ssim:.4f} SSIM")

# Create a high-detail zoomed crop (butterfly wing center)
# The image is 256x256. Center crop around (100, 80) to (180, 160) = 80x80
crop_box = (90, 80, 170, 160) # 80x80 box
zoom_factor = 3
zoom_w = (crop_box[2] - crop_box[0]) * zoom_factor
zoom_h = (crop_box[3] - crop_box[1]) * zoom_factor

# Stitch a 5-column panel
header_h = 50
panel_w = zoom_w
panel_h = zoom_h + header_h
total_w = panel_w * len(methods)
total_h = panel_h

composite = Image.new("RGB", (total_w, total_h), (25, 25, 25))
draw = ImageDraw.Draw(composite)

try:
    font = ImageFont.truetype("arial.ttf", 16)
    font_sub = ImageFont.truetype("arial.ttf", 13)
except Exception:
    font = ImageFont.load_default()
    font_sub = font

for idx, (name, im) in enumerate(methods):
    x_off = idx * panel_w
    # Crop and zoom with nearest-neighbor to show pixel-level sharpness without artificial blurring
    crop = im.crop(crop_box).resize((zoom_w, zoom_h), Image.NEAREST)
    composite.paste(crop, (x_off, header_h))
    
    # Text
    if name == "Ground Truth":
        text_title = "Ground Truth"
        text_metric = "HR 256x256"
    else:
        psnr, ssim = get_metrics(im, gt_img)
        text_title = name
        text_metric = f"{psnr:.2f} dB | {ssim:.4f}"
        
    draw.text((x_off + 10, 8), text_title, fill=(255, 255, 255), font=font)
    draw.text((x_off + 10, 28), text_metric, fill=(200, 200, 100) if "MYELIN" in name else (180, 180, 180), font=font_sub)
    
    # Border line
    if idx > 0:
        draw.line([(x_off, 0), (x_off, total_h)], fill=(60, 60, 60), width=2)

out_path = OUT_DIR / "visual_comparison_butterfly.png"
composite.save(out_path)
print(f"\n[Saved Visual Comparison] {out_path} ({total_w}x{total_h})")
