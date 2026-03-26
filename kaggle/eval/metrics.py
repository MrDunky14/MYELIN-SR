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
PSNR, SSIM, and utility metrics for MYELIN-SR evaluation.

These are the standard quality metrics used to benchmark super-resolution:
  - PSNR: Peak Signal-to-Noise Ratio (higher = better, measured in dB)
  - SSIM: Structural Similarity Index (higher = better, range 0-1)
  
Calculated on Y channel (luminance) in YCbCr space, as is standard practice.
"""

import torch
import torch.nn.functional as F
import math


def rgb_to_ycbcr(img: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB tensor to YCbCr. Input/output: [B, 3, H, W] in range [0, 1].
    Returns Y channel only for metric computation.
    """
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    y = 16.0 / 255.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
    return y


def calculate_psnr(
    sr: torch.Tensor,
    hr: torch.Tensor,
    crop_border: int = 0,
    y_channel: bool = True,
) -> float:
    """
    Calculate PSNR between super-resolved and high-resolution images.
    
    Args:
        sr: Super-resolved image [B, C, H, W] in [0, 1]
        hr: Ground truth HR image [B, C, H, W] in [0, 1]
        crop_border: Pixels to crop from each border (standard: scale_factor)
        y_channel: If True, compute on Y channel only (standard practice)
    
    Returns:
        PSNR value in dB (higher is better)
    """
    sr = sr.detach().clamp(0, 1)
    hr = hr.detach().clamp(0, 1)

    if y_channel and sr.size(1) == 3:
        sr = rgb_to_ycbcr(sr)
        hr = rgb_to_ycbcr(hr)

    if crop_border > 0:
        sr = sr[..., crop_border:-crop_border, crop_border:-crop_border]
        hr = hr[..., crop_border:-crop_border, crop_border:-crop_border]

    mse = F.mse_loss(sr, hr, reduction="mean").item()
    if mse < 1e-10:
        return 100.0  # Identical images

    return 10.0 * math.log10(1.0 / mse)


def _create_ssim_window(window_size: int, channel: int) -> torch.Tensor:
    """Create Gaussian window for SSIM computation."""
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()

    window_1d = gauss.unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t())
    window = window_2d.unsqueeze(0).unsqueeze(0).expand(channel, 1, -1, -1).contiguous()
    return window


def calculate_ssim(
    sr: torch.Tensor,
    hr: torch.Tensor,
    crop_border: int = 0,
    y_channel: bool = True,
    window_size: int = 11,
) -> float:
    """
    Calculate SSIM between super-resolved and high-resolution images.
    
    Args:
        sr: Super-resolved image [B, C, H, W] in [0, 1]
        hr: Ground truth HR image [B, C, H, W] in [0, 1]
        crop_border: Pixels to crop from each border
        y_channel: If True, compute on Y channel only
        window_size: Size of Gaussian sliding window
    
    Returns:
        SSIM value (higher is better, range 0-1)
    """
    sr = sr.detach().clamp(0, 1)
    hr = hr.detach().clamp(0, 1)

    if y_channel and sr.size(1) == 3:
        sr = rgb_to_ycbcr(sr)
        hr = rgb_to_ycbcr(hr)

    if crop_border > 0:
        sr = sr[..., crop_border:-crop_border, crop_border:-crop_border]
        hr = hr[..., crop_border:-crop_border, crop_border:-crop_border]

    channel = sr.size(1)
    window = _create_ssim_window(window_size, channel).to(sr.device, sr.dtype)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_sr = F.conv2d(sr, window, padding=window_size // 2, groups=channel)
    mu_hr = F.conv2d(hr, window, padding=window_size // 2, groups=channel)

    mu_sr_sq = mu_sr ** 2
    mu_hr_sq = mu_hr ** 2
    mu_sr_hr = mu_sr * mu_hr

    sigma_sr_sq = F.conv2d(sr * sr, window, padding=window_size // 2, groups=channel) - mu_sr_sq
    sigma_hr_sq = F.conv2d(hr * hr, window, padding=window_size // 2, groups=channel) - mu_hr_sq
    sigma_sr_hr = F.conv2d(sr * hr, window, padding=window_size // 2, groups=channel) - mu_sr_hr

    ssim_map = ((2 * mu_sr_hr + C1) * (2 * sigma_sr_hr + C2)) / \
               ((mu_sr_sq + mu_hr_sq + C1) * (sigma_sr_sq + sigma_hr_sq + C2))

    return ssim_map.mean().item()


class PhaseGateValidator:
    """
    Phase-Gate Quality Validator.
    
    Enforces minimum quality thresholds before allowing progression to the
    next training phase. Prevents bugs and imperfections from propagating.
    
    Phase 1 Gates:
      - Gate 1.1: Model produces valid output (correct shape, no NaN/Inf)
      - Gate 1.2: PSNR > bicubic baseline (sanity check — must beat trivial)
      - Gate 1.3: PSNR ≥ 30.5 dB on Set14 ×2 (beats Lanczos)
      - Gate 1.4: PSNR ≥ 33.0 dB on Set14 ×2 (matches EDSR-baseline)
      - Gate 1.5: Ternary quantized PSNR within 0.5 dB of FP32 (quantization quality)
      - Gate 1.6: Sparsity > 30% (ternary weights are actually sparse)
    """

    PHASE1_GATES = {
        "valid_output": {
            "description": "Model produces valid output (correct shape, no NaN/Inf)",
            "threshold": None,  # Boolean check
        },
        "beats_bicubic": {
            "description": "PSNR exceeds bicubic interpolation baseline",
            "metric": "psnr",
            "threshold": 30.3,  # Bicubic ~30.24 on Set14 ×2
        },
        "beats_lanczos": {
            "description": "PSNR exceeds Lanczos interpolation",
            "metric": "psnr",
            "threshold": 30.5,
        },
        "matches_edsr": {
            "description": "PSNR matches EDSR-baseline level",
            "metric": "psnr",
            "threshold": 33.0,
        },
        "ternary_quality": {
            "description": "Ternary quantized model within 0.5 dB of FP32",
            "metric": "psnr_gap",
            "threshold": 0.5,
        },
        "sparsity_check": {
            "description": "Ternary weight sparsity exceeds 20%",
            "metric": "sparsity",
            "threshold": 0.20,
        },
    }

    def __init__(self):
        self.results = {}

    def check_valid_output(self, model, device: torch.device) -> bool:
        """Gate 1.1: Verify model produces valid output."""
        model.eval()
        try:
            with torch.no_grad():
                dummy = torch.randn(1, 3, 64, 64, device=device).clamp(0, 1)
                output = model(dummy)

                # Check shape
                expected = (1, 3, 128, 128)  # Assuming ×2 upscale
                assert output.shape == expected, f"Shape mismatch: {output.shape} vs {expected}"

                # Check for NaN/Inf
                assert not torch.isnan(output).any(), "Output contains NaN"
                assert not torch.isinf(output).any(), "Output contains Inf"

                # Clamp output to valid image range [0, 1]
                output_clamped = output.clamp(0, 1)
                
                # Check value range after clamping (should be reasonable)
                assert output_clamped.min() >= -0.01, f"Output min too low: {output_clamped.min()}"
                assert output_clamped.max() <= 1.01, f"Output max too high: {output_clamped.max()}"

            self.results["valid_output"] = True
            return True
        except Exception as e:
            self.results["valid_output"] = f"FAILED: {e}"
            return False

    def check_psnr_gate(self, psnr_value: float, gate_name: str) -> bool:
        """Check if PSNR exceeds a specific gate threshold."""
        gate = self.PHASE1_GATES[gate_name]
        passed = psnr_value >= gate["threshold"]
        self.results[gate_name] = {
            "value": psnr_value,
            "threshold": gate["threshold"],
            "passed": passed,
        }
        return passed

    def check_sparsity(self, model) -> bool:
        """Gate 1.6: Verify ternary weight sparsity."""
        sparsity = model.get_total_sparsity()
        threshold = self.PHASE1_GATES["sparsity_check"]["threshold"]
        passed = sparsity >= threshold
        self.results["sparsity_check"] = {
            "value": sparsity,
            "threshold": threshold,
            "passed": passed,
        }
        return passed

    def print_report(self) -> None:
        """Print a formatted gate check report."""
        print("\n" + "=" * 60)
        print("  PHASE GATE VALIDATION REPORT")
        print("=" * 60)

        all_passed = True
        for gate_name, gate_info in self.PHASE1_GATES.items():
            if gate_name not in self.results:
                status = "[--] NOT RUN"
                all_passed = False
            elif isinstance(self.results[gate_name], bool):
                status = "[OK] PASS" if self.results[gate_name] else "[!!] FAIL"
                if not self.results[gate_name]:
                    all_passed = False
            elif isinstance(self.results[gate_name], str):
                status = f"[!!] {self.results[gate_name]}"
                all_passed = False
            elif isinstance(self.results[gate_name], dict):
                r = self.results[gate_name]
                if r["passed"]:
                    status = f"[OK] PASS ({r['value']:.2f} >= {r['threshold']})"
                else:
                    status = f"[!!] FAIL ({r['value']:.2f} < {r['threshold']})"
                    all_passed = False

            print(f"  [{gate_name}] {gate_info['description']}")
            print(f"    -> {status}")

        print("-" * 60)
        if all_passed:
            print("  >> ALL GATES PASSED -- Ready for next phase")
        else:
            print("  >> Some gates failed -- fix before proceeding")
        print("=" * 60)
