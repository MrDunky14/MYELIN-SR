"""
MYELIN-SR Training Pipeline

Trains MyelinSRNet with:
  - Standard backpropagation for initial heavy training
  - Astrocytic Stiffness consolidation between training stages
  - Phase-gate validation to prevent imperfections from propagating
  - Dual homeostatic learning rate control

Usage (local):
  python train.py --data-root ./data --epochs 100 --scale 2

Usage (Kaggle):
  Upload train/ and eval/ directories, set data-root to Kaggle dataset path.
  See train_kaggle.py for a ready-to-run Kaggle notebook script.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# Add parent directories to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from model import build_myelin_sr, MyelinSRNet
from dataset import build_train_loader, build_val_loader
from losses import MyelinSRLoss
from fpsan_conv2d import FPSANConv2d
from metrics import calculate_psnr, calculate_ssim, PhaseGateValidator


# ──────────────────────────────────────────────────────────────────────
# Homeostatic Learning Rate Controller (Decision 4)
# ──────────────────────────────────────────────────────────────────────

class HomeostaticLRController:
    """
    Dual Homeostatic Control for learning rate.
    
    Adapts LR based on gradient energy:
      - If gradients spike (new content) → increase LR for faster adaptation
      - If gradients flatline (familiar content) → decrease LR for stability
    
    Uses exponential moving average of gradient magnitude.
    Reference: V6 exponential homeostasis (fpsan_v6_atomic.cpp L50-L68)
    """

    def __init__(
        self,
        base_lr: float = 1e-4,
        lr_min: float = 1e-6,
        lr_max: float = 5e-4,
        ema_alpha: float = 0.1,
    ):
        self.base_lr = base_lr
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.ema_alpha = ema_alpha
        self.expected_grad_energy = 1.0  # Initial expectation

    def step(self, current_grad_energy: float) -> float:
        """
        Update expected energy and return adapted learning rate.
        
        homeostasis_ratio > 1.0 → gradients higher than expected → boost LR
        homeostasis_ratio < 1.0 → gradients lower than expected → dampen LR
        """
        # Update EMA
        self.expected_grad_energy = (
            (1 - self.ema_alpha) * self.expected_grad_energy
            + self.ema_alpha * max(current_grad_energy, 1e-8)
        )

        # Compute homeostasis ratio
        ratio = current_grad_energy / max(self.expected_grad_energy, 1e-8)

        # Bounded adaptive LR
        adapted_lr = self.base_lr * math.pow(ratio, 0.5)  # Sqrt for smoother adaptation
        return max(self.lr_min, min(self.lr_max, adapted_lr))


# ──────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────

def compute_gradient_energy(model: nn.Module) -> float:
    """Compute mean absolute gradient across all trainable parameters."""
    total = 0.0
    count = 0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.abs().mean().item()
            count += 1
    return total / max(count, 1)


def train_one_epoch(
    model: MyelinSRNet,
    train_loader,
    criterion: MyelinSRLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    homeostatic_lr: HomeostaticLRController,
    log_interval: int = 50,
) -> dict:
    """Train for one epoch, returns average metrics."""
    model.train()
    total_loss = 0.0
    total_l1 = 0.0
    total_perceptual = 0.0
    num_batches = 0

    for batch_idx, (lr_imgs, hr_imgs) in enumerate(train_loader):
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        # Forward pass (ternary quantization happens inside FPSANConv2d)
        sr_imgs = model(lr_imgs)
        loss, loss_dict = criterion(sr_imgs, hr_imgs)

        # Backward pass + optimize
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent instability from ternary quantization
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Homeostatic LR adaptation
        grad_energy = compute_gradient_energy(model)
        adapted_lr = homeostatic_lr.step(grad_energy)
        for param_group in optimizer.param_groups:
            param_group['lr'] = adapted_lr

        optimizer.step()

        total_loss += loss_dict["total"]
        total_l1 += loss_dict["l1"]
        total_perceptual += loss_dict.get("perceptual", 0.0)
        num_batches += 1

        if batch_idx % log_interval == 0:
            print(
                f"  Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss_dict['total']:.4f} | "
                f"L1: {loss_dict['l1']:.4f} | "
                f"LR: {adapted_lr:.2e} | "
                f"Sparsity: {model.get_total_sparsity() * 100:.1f}%"
            )

    return {
        "loss": total_loss / max(num_batches, 1),
        "l1": total_l1 / max(num_batches, 1),
        "perceptual": total_perceptual / max(num_batches, 1),
        "lr": adapted_lr,
        "sparsity": model.get_total_sparsity(),
    }


@torch.no_grad()
def validate(
    model: MyelinSRNet,
    val_loader,
    device: torch.device,
    scale: int = 2,
) -> dict:
    """Validate on benchmark set, returns PSNR and SSIM."""
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    for lr_imgs, hr_imgs, filenames in val_loader:
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        sr_imgs = model(lr_imgs)

        psnr = calculate_psnr(sr_imgs, hr_imgs, crop_border=scale)
        ssim = calculate_ssim(sr_imgs, hr_imgs, crop_border=scale)

        total_psnr += psnr
        total_ssim += ssim
        count += 1

    return {
        "psnr": total_psnr / max(count, 1),
        "ssim": total_ssim / max(count, 1),
    }


# ──────────────────────────────────────────────────────────────────────
# Main Training Pipeline
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MYELIN-SR Training Pipeline")
    parser.add_argument("--data-root", type=str, default="./data",
                        help="Root directory containing DIV2K_train_HR/, Set14/, etc.")
    parser.add_argument("--output-dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--scale", type=int, default=2, choices=[2, 4],
                        help="Upscale factor")
    parser.add_argument("--quality", type=str, default="quality",
                        choices=["performance", "balanced", "quality", "ultra"],
                        help="Quality preset")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--patch-size", type=int, default=256,
                        help="HR training patch size")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Base learning rate")
    parser.add_argument("--perceptual-weight", type=float, default=0.1,
                        help="Weight for perceptual loss")
    parser.add_argument("--consolidation-rate", type=float, default=0.08,
                        help="Astrocytic Stiffness consolidation rate during sleep")
    parser.add_argument("--consolidation-interval", type=int, default=50,
                        help="Epochs between sleep consolidation phases")
    parser.add_argument("--val-interval", type=int, default=10,
                        help="Epochs between validation runs")
    parser.add_argument("--val-set", type=str, default="Set14",
                        help="Validation benchmark set name")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if not specified)")
    args = parser.parse_args()

    # Setup
    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  MYELIN-SR Training Pipeline")
    print("=" * 60)
    print(f"  Device:      {device}")
    print(f"  Scale:       {args.scale}×")
    print(f"  Quality:     {args.quality}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Base LR:     {args.lr}")
    print(f"  Data root:   {args.data_root}")
    print("=" * 60)

    # Build model
    model = build_myelin_sr(upscale=args.scale, quality_mode=args.quality).to(device)
    model.print_architecture_summary()

    # Phase Gate: Check valid output before training
    validator = PhaseGateValidator()
    if not validator.check_valid_output(model, device):
        validator.print_report()
        raise RuntimeError("Gate 1.1 FAILED: Model produces invalid output. Fix architecture before training.")
    print("  ✅ Gate 1.1 PASSED: Model produces valid output\n")

    # Build data
    train_loader = build_train_loader(
        args.data_root, args.scale, args.patch_size, args.batch_size, args.num_workers
    )

    val_loader = None
    val_path = Path(args.data_root) / args.val_set
    if val_path.exists():
        val_loader = build_val_loader(args.data_root, args.val_set, args.scale)
    else:
        print(f"  [WARNING] Validation set {args.val_set} not found at {val_path}")

    # Loss, optimizer, scheduler
    criterion = MyelinSRLoss(perceptual_weight=args.perceptual_weight).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    homeostatic_lr = HomeostaticLRController(base_lr=args.lr)

    # Resume from checkpoint
    start_epoch = 1
    best_psnr = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_psnr = checkpoint.get("best_psnr", 0.0)
        print(f"  Resumed from epoch {start_epoch - 1}, best PSNR: {best_psnr:.2f}")

    # Training log
    training_log = []

    # ── Main Training Loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # Train one epoch
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, homeostatic_lr
        )

        # Step LR scheduler
        scheduler.step()
        epoch_time = time.time() - epoch_start

        # Log
        log_entry = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_l1": train_metrics["l1"],
            "lr": train_metrics["lr"],
            "sparsity": train_metrics["sparsity"],
            "epoch_time_s": epoch_time,
        }

        # ── Validation ──
        if val_loader and epoch % args.val_interval == 0:
            val_metrics = validate(model, val_loader, device, args.scale)
            log_entry["val_psnr"] = val_metrics["psnr"]
            log_entry["val_ssim"] = val_metrics["ssim"]

            print(
                f"\n  📊 Epoch {epoch} Validation: "
                f"PSNR={val_metrics['psnr']:.2f} dB | "
                f"SSIM={val_metrics['ssim']:.4f} | "
                f"Time={epoch_time:.1f}s"
            )

            # Save best model
            if val_metrics["psnr"] > best_psnr:
                best_psnr = val_metrics["psnr"]
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_psnr": best_psnr,
                    "config": {
                        "scale": args.scale,
                        "quality": args.quality,
                    },
                }, output_dir / "myelin_sr_best.pt")
                print(f"  💾 New best model saved (PSNR: {best_psnr:.2f})")

            # ── Phase Gate Checks ──
            if val_metrics["psnr"] > 30.3:
                validator.check_psnr_gate(val_metrics["psnr"], "beats_bicubic")
            if val_metrics["psnr"] > 30.5:
                validator.check_psnr_gate(val_metrics["psnr"], "beats_lanczos")
            if val_metrics["psnr"] > 33.0:
                validator.check_psnr_gate(val_metrics["psnr"], "matches_edsr")

        # ── Sleep Consolidation (Astrocytic Stiffness) ──
        if epoch % args.consolidation_interval == 0:
            print(f"\n  😴 Sleep Consolidation at epoch {epoch} (rate={args.consolidation_rate})")
            model.consolidate_all(rate=args.consolidation_rate)
            validator.check_sparsity(model)

        training_log.append(log_entry)

        # Save periodic checkpoint
        if epoch % 50 == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_psnr": best_psnr,
            }, output_dir / f"myelin_sr_epoch_{epoch}.pt")

    # ── Final Report ──
    print("\n")
    validator.print_report()

    # Save training log
    log_path = output_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"\n  📋 Training log saved to {log_path}")

    # Save final model
    final_path = output_dir / "myelin_sr_final.pt"
    torch.save({
        "epoch": args.epochs,
        "model_state": model.state_dict(),
        "best_psnr": best_psnr,
        "config": {"scale": args.scale, "quality": args.quality},
    }, final_path)
    print(f"  💾 Final model saved to {final_path}")
    print(f"  🏆 Best PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()
