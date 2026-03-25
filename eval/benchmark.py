"""
MYELIN-SR Benchmark Evaluation

Runs the trained model against standard SR benchmarks and produces
a formatted quality report with comparison to known baselines.

Usage:
  python benchmark.py --model ../checkpoints/myelin_sr_best.pt --data-root ../data --scale 2
"""

import argparse
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent / "train"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(TRAIN_DIR))

from metrics import calculate_psnr, calculate_ssim, PhaseGateValidator


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained MyelinSRNet from checkpoint."""
    from model import build_myelin_sr

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {"scale": 2, "quality": "quality"})

    model = build_myelin_sr(
        upscale=config["scale"],
        quality_mode=config["quality"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"  Loaded model from {checkpoint_path}")
    print(f"  Trained for {checkpoint.get('epoch', '?')} epochs, best PSNR: {checkpoint.get('best_psnr', '?')}")

    return model, config["scale"]


def benchmark_dataset(model, data_root: str, dataset_name: str, scale: int, device: torch.device) -> dict:
    """Benchmark on a single dataset."""
    from dataset import build_val_loader

    dataset_path = Path(data_root) / dataset_name
    if not dataset_path.exists():
        return {"error": f"Not found: {dataset_path}"}

    val_loader = build_val_loader(data_root, dataset_name, scale)

    total_psnr = 0.0
    total_ssim = 0.0
    total_time = 0.0
    count = 0
    per_image = []

    with torch.no_grad():
        for lr_imgs, hr_imgs, filenames in val_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            start = time.perf_counter()
            sr_imgs = model(lr_imgs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            psnr = calculate_psnr(sr_imgs, hr_imgs, crop_border=scale)
            ssim = calculate_ssim(sr_imgs, hr_imgs, crop_border=scale)

            total_psnr += psnr
            total_ssim += ssim
            total_time += elapsed
            count += 1

            per_image.append({
                "filename": filenames[0] if filenames else f"img_{count}",
                "psnr": psnr,
                "ssim": ssim,
                "time_ms": elapsed * 1000,
            })

    return {
        "dataset": dataset_name,
        "avg_psnr": total_psnr / max(count, 1),
        "avg_ssim": total_ssim / max(count, 1),
        "avg_time_ms": (total_time / max(count, 1)) * 1000,
        "num_images": count,
        "per_image": per_image,
    }


# Known baseline PSNR values on standard benchmarks (×2 scale)
BASELINES_X2 = {
    "Set5": {"Bicubic": 33.66, "EDSR-baseline": 37.99, "SwinIR-light": 38.14},
    "Set14": {"Bicubic": 30.24, "EDSR-baseline": 33.57, "SwinIR-light": 33.86},
    "BSD100": {"Bicubic": 29.56, "EDSR-baseline": 32.16, "SwinIR-light": 32.31},
    "Urban100": {"Bicubic": 26.88, "EDSR-baseline": 31.98, "SwinIR-light": 32.76},
}

BASELINES_X4 = {
    "Set5": {"Bicubic": 28.42, "EDSR-baseline": 32.09, "SwinIR-light": 32.44},
    "Set14": {"Bicubic": 26.00, "EDSR-baseline": 28.58, "SwinIR-light": 28.77},
    "BSD100": {"Bicubic": 25.96, "EDSR-baseline": 27.57, "SwinIR-light": 27.69},
    "Urban100": {"Bicubic": 23.14, "EDSR-baseline": 26.04, "SwinIR-light": 26.47},
}


def print_comparison_table(results: list, scale: int):
    """Print formatted comparison table against known baselines."""
    baselines = BASELINES_X2 if scale == 2 else BASELINES_X4

    print("\n" + "=" * 80)
    print(f"  MYELIN-SR Benchmark Results (×{scale} upscale)")
    print("=" * 80)
    print(f"  {'Dataset':<12} {'MYELIN-SR':>10} {'Bicubic':>10} {'EDSR-base':>10} {'SwinIR-lt':>10} {'Time(ms)':>10}")
    print("-" * 80)

    for result in results:
        if "error" in result:
            print(f"  {result.get('dataset', '?'):<12} {'ERROR':>10}")
            continue

        ds = result["dataset"]
        our_psnr = result["avg_psnr"]
        baseline = baselines.get(ds, {})

        bicubic = baseline.get("Bicubic", "-")
        edsr = baseline.get("EDSR-baseline", "-")
        swinir = baseline.get("SwinIR-light", "-")

        def fmt(v):
            return f"{v:.2f}" if isinstance(v, float) else str(v)

        # Color coding via text markers
        marker = ""
        if isinstance(bicubic, float):
            if our_psnr > edsr if isinstance(edsr, float) else False:
                marker = " 🏆"
            elif our_psnr > bicubic:
                marker = " ✅"
            else:
                marker = " ⚠️"

        print(
            f"  {ds:<12} {our_psnr:>9.2f}{marker}"
            f" {fmt(bicubic):>10} {fmt(edsr):>10} {fmt(swinir):>10}"
            f" {result['avg_time_ms']:>9.1f}"
        )

    print("=" * 80)
    print("  ✅ = Beats Bicubic | 🏆 = Beats EDSR-baseline | ⚠️ = Below Bicubic")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="MYELIN-SR Benchmark Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data-root", type=str, default="./data", help="Root data directory")
    parser.add_argument("--scale", type=int, default=2, choices=[2, 4])
    parser.add_argument("--datasets", nargs="+", default=["Set5", "Set14", "BSD100", "Urban100"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run-gates", action="store_true", help="Run phase-gate validation")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, scale = load_model(args.model, device)

    # Run benchmarks
    results = []
    for ds_name in args.datasets:
        print(f"\n  Benchmarking on {ds_name}...")
        result = benchmark_dataset(model, args.data_root, ds_name, scale, device)
        results.append(result)

    # Print comparison table
    print_comparison_table(results, scale)

    # Phase gate validation
    if args.run_gates:
        validator = PhaseGateValidator()
        validator.check_valid_output(model, device)
        validator.check_sparsity(model)

        for result in results:
            if "error" not in result and result["dataset"] == "Set14":
                validator.check_psnr_gate(result["avg_psnr"], "beats_bicubic")
                validator.check_psnr_gate(result["avg_psnr"], "beats_lanczos")
                validator.check_psnr_gate(result["avg_psnr"], "matches_edsr")

        validator.print_report()

    # Model stats
    print(f"\n  Model Statistics:")
    model.print_architecture_summary()


if __name__ == "__main__":
    main()
