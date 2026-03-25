"""
MYELIN-SR Kaggle Training Script

Ready to run on Kaggle with T4 GPU (free tier).
Upload the entire FP-SAN_NSS/train/ and FP-SAN_NSS/eval/ directories as a dataset,
then paste this script into a Kaggle notebook code cell and run.

Estimated training time on Kaggle T4:
  - quality preset, 200 epochs, batch=16, DIV2K only: ~8-10 hours
  - Use Kaggle's 12h session limit wisely — save checkpoints every 50 epochs.

Setup steps:
  1. Go to kaggle.com → Datasets → New Dataset
  2. Upload the `train/` and `eval/` folders (zip them first if needed)
  3. Create a new notebook → Add your dataset
  4. Enable T4 GPU accelerator
  5. Paste and run this script

Kaggle dataset path will be: /kaggle/input/your-dataset-name/
"""

# ─── Cell 1: Setup ───────────────────────────────────────────────────
import subprocess, sys

def pip(pkg):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

# No extra deps needed — torch and torchvision are pre-installed on Kaggle

# ─── Cell 2: Paths ───────────────────────────────────────────────────
import os
from pathlib import Path

# THE NAME OF THE DATASET YOU UPLOADED TO KAGGLE (containing train/ and eval/)
CODEBASE_DATASET_NAME = "myelin-sr-codebase" 

# Kaggle input/output paths
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

CODE_ROOT = KAGGLE_INPUT / CODEBASE_DATASET_NAME
TRAIN_DIR = CODE_ROOT / "train"
EVAL_DIR  = CODE_ROOT / "eval"

# Writable directories for data and checkpoints
DATA_DIR = KAGGLE_WORKING / "data"
CKPT_DIR = KAGGLE_WORKING / "checkpoints"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Add code to Python path
sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(EVAL_DIR))

print(f"Code root: {CODE_ROOT}")
print(f"Data root: {DATA_DIR}")
print(f"Ckpt root: {CKPT_DIR}")

# ─── Cell 3: Download DIV2K Dataset ─────────────────────────────────
# DIV2K is available as a Kaggle dataset — search "DIV2K" in Datasets
# Or download manually:

import urllib.request
import zipfile

def download_div2k():
    """Download DIV2K training HR images."""
    url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    zip_path = DATA_DIR / "DIV2K_train_HR.zip"
    hr_dir = DATA_DIR / "DIV2K_train_HR"

    if hr_dir.exists() and len(list(hr_dir.glob("*.png"))) > 700:
        print(f"DIV2K already downloaded ({len(list(hr_dir.glob('*.png')))} images)")
        return

    print("Downloading DIV2K HR training set (~3.5GB)...")
    urllib.request.urlretrieve(url, zip_path)
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    zip_path.unlink()
    print(f"Done. {len(list(hr_dir.glob('*.png')))} images.")

def download_set14():
    """Download Set14 benchmark test set."""
    # Set14 is commonly available on GitHub
    url = "https://github.com/jbhuang0604/SelfExSR/raw/master/data/Set14.zip"
    out = DATA_DIR / "Set14"
    if out.exists():
        print("Set14 already downloaded")
        return
    zip_path = DATA_DIR / "Set14.zip"
    try:
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR)
        zip_path.unlink()
        print(f"Set14 downloaded: {len(list(out.glob('*.png')))} images")
    except Exception as e:
        print(f"Could not download Set14 automatically: {e}")
        print("Add Set14 manually to /kaggle/working/data/Set14/")

download_div2k()
download_set14()

# ─── Cell 4: Import & Verify ─────────────────────────────────────────
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

from model import build_myelin_sr
from fpsan_conv2d import FPSANConv2d
from losses import MyelinSRLoss
from metrics import calculate_psnr, calculate_ssim, PhaseGateValidator
from dataset import build_train_loader, build_val_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Quick gate check before training
model_check = build_myelin_sr(2, "quality").to(device)
validator = PhaseGateValidator()
assert validator.check_valid_output(model_check, device), "Gate 1.1 FAILED — fix before training"
print("Gate 1.1: PASS — model valid on", device)
del model_check

# ─── Cell 5: Training Configuration ──────────────────────────────────
CONFIG = {
    "scale":                 2,
    "quality":               "quality",    # performance / balanced / quality / ultra
    "epochs":                200,
    "batch_size":            16,           # T4 has 16GB — can go up to 32 if needed
    "patch_size":            256,
    "base_lr":               2e-4,
    "perceptual_weight":     0.1,
    "consolidation_rate":    0.08,
    "consolidation_interval":50,           # Sleep every 50 epochs
    "val_interval":          10,
    "val_set":               "Set14",
    "num_workers":           4,
    "resume_checkpoint":     None,         # Set to path to resume training
}

print("Training configuration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")

# ─── Cell 6: Build Everything ────────────────────────────────────────
import json, time, math
import torch.optim as optim
import torch.nn as nn
from fpsan_conv2d import FPSANConv2d

model = build_myelin_sr(CONFIG["scale"], CONFIG["quality"]).to(device)
model.print_architecture_summary()

train_loader = build_train_loader(
    str(DATA_DIR),
    CONFIG["scale"],
    CONFIG["patch_size"],
    CONFIG["batch_size"],
    CONFIG["num_workers"],
)

val_loader = None
if (DATA_DIR / CONFIG["val_set"]).exists():
    val_loader = build_val_loader(str(DATA_DIR), CONFIG["val_set"], CONFIG["scale"])
else:
    print(f"[WARNING] Validation set {CONFIG['val_set']} not found in {DATA_DIR}")

criterion = MyelinSRLoss(perceptual_weight=CONFIG["perceptual_weight"]).to(device)
optimizer = optim.AdamW(model.parameters(), lr=CONFIG["base_lr"], weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-6)

start_epoch = 1
best_psnr = 0.0

if CONFIG["resume_checkpoint"]:
    ckpt = torch.load(CONFIG["resume_checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_psnr = ckpt.get("best_psnr", 0.0)
    print(f"Resumed from epoch {start_epoch - 1}, best PSNR: {best_psnr:.2f}")

# ─── Cell 7: Homeostatic LR ──────────────────────────────────────────
class HomeostaticLR:
    def __init__(self, base_lr, lr_min=1e-6, lr_max=5e-4, alpha=0.1):
        self.base_lr = base_lr
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.alpha = alpha
        self.expected_energy = 1.0

    def step(self, energy):
        self.expected_energy = (1 - self.alpha) * self.expected_energy + self.alpha * max(energy, 1e-8)
        ratio = energy / max(self.expected_energy, 1e-8)
        lr = self.base_lr * math.pow(ratio, 0.5)
        return max(self.lr_min, min(self.lr_max, lr))

homeostatic = HomeostaticLR(CONFIG["base_lr"])

# ─── Cell 8: Training Loop ────────────────────────────────────────────
log = []

for epoch in range(start_epoch, CONFIG["epochs"] + 1):
    model.train()
    ep_loss, ep_psnr, n_batches = 0.0, 0.0, 0
    t0 = time.time()

    for lr_imgs, hr_imgs in train_loader:
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        sr_imgs = model(lr_imgs)
        loss, ld = criterion(sr_imgs, hr_imgs)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Homeostatic LR update
        ge = sum(p.grad.abs().mean().item() for p in model.parameters() if p.grad is not None)
        ge /= max(sum(1 for p in model.parameters() if p.grad is not None), 1)
        for pg in optimizer.param_groups:
            pg["lr"] = homeostatic.step(ge)

        optimizer.step()
        ep_loss += ld["total"]
        n_batches += 1

    scheduler.step()
    elapsed = time.time() - t0
    entry = {"epoch": epoch, "loss": ep_loss / n_batches, "time_s": elapsed}

    # Validation
    if val_loader and epoch % CONFIG["val_interval"] == 0:
        model.eval()
        psnr_sum, ssim_sum, nv = 0.0, 0.0, 0
        with torch.no_grad():
            for lr_v, hr_v, _ in val_loader:
                sr_v = model(lr_v.to(device))
                psnr_sum += calculate_psnr(sr_v, hr_v.to(device), crop_border=CONFIG["scale"])
                ssim_sum += calculate_ssim(sr_v, hr_v.to(device), crop_border=CONFIG["scale"])
                nv += 1
        val_psnr = psnr_sum / max(nv, 1)
        val_ssim = ssim_sum / max(nv, 1)
        entry.update({"val_psnr": val_psnr, "val_ssim": val_ssim})
        print(f"Epoch {epoch:03d} | Loss={ep_loss/n_batches:.4f} | PSNR={val_psnr:.2f}dB | SSIM={val_ssim:.4f} | {elapsed:.0f}s")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_psnr": best_psnr,
                "config": {"scale": CONFIG["scale"], "quality": CONFIG["quality"]},
            }, CKPT_DIR / "myelin_sr_best.pt")
            print(f"  >> New best: {best_psnr:.2f} dB saved")
    else:
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss={ep_loss/n_batches:.4f} | {elapsed:.0f}s")

    # Sleep consolidation
    if epoch % CONFIG["consolidation_interval"] == 0:
        model.consolidate_all(rate=CONFIG["consolidation_rate"])
        sp = model.get_total_sparsity() * 100
        print(f"  [Sleep] Epoch {epoch}: sparsity={sp:.1f}%")

    # Periodic checkpoint
    if epoch % 50 == 0:
        torch.save({
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "best_psnr": best_psnr,
        }, CKPT_DIR / f"myelin_sr_epoch_{epoch}.pt")

    log.append(entry)

# ─── Cell 9: Final Results ────────────────────────────────────────────
with open(CKPT_DIR / "training_log.json", "w") as f:
    json.dump(log, f, indent=2)

print(f"\nTraining complete. Best PSNR: {best_psnr:.2f} dB")
print(f"Checkpoints at: {CKPT_DIR}")

# Final phase gate report
validator = PhaseGateValidator()
validator.check_valid_output(model, device)
validator.check_sparsity(model)
if best_psnr > 0:
    validator.check_psnr_gate(best_psnr, "beats_bicubic")
    if best_psnr > 30.5:
        validator.check_psnr_gate(best_psnr, "beats_lanczos")
    if best_psnr > 33.0:
        validator.check_psnr_gate(best_psnr, "matches_edsr")
validator.print_report()

# ─── Cell 10: Export to Ternary Binary ───────────────────────────────
sys.path.insert(0, str(TRAIN_DIR))
from export_model import export_ternary_binary

best_ckpt = CKPT_DIR / "myelin_sr_best.pt"
if best_ckpt.exists():
    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval().cpu()
    config = ckpt.get("config", {"scale": CONFIG["scale"], "quality": CONFIG["quality"]})
    bin_path = CKPT_DIR / "myelin_sr_deploy.bin"
    size = export_ternary_binary(model, bin_path, config)
    fp32_size = sum(p.numel() * 4 for p in model.parameters())
    print(f"\nExport complete:")
    print(f"  FP32 size:      {fp32_size / 1024:.1f} KB")
    print(f"  Ternary binary: {size / 1024:.1f} KB  ({fp32_size / size:.1f}x smaller)")
    print(f"  Saved to:       {bin_path}")
