"""Phase 1 Verification Suite — run this to validate the full pipeline before training."""
import sys, os

# Add both directories for clean imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "train"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "eval"))

import torch
from model import build_myelin_sr
from fpsan_conv2d import FPSANConv2d
from metrics import PhaseGateValidator, calculate_psnr, calculate_ssim
from losses import MyelinSRLoss

def main():
    print("=" * 60)
    print("  MYELIN-SR Phase 1 Verification Suite")
    print("=" * 60)

    # --- Test 1: All Quality Presets ---
    print("\n[TEST 1] Quality Preset Forward Pass")
    for preset in ["performance", "balanced", "quality", "ultra"]:
        model = build_myelin_sr(upscale=2, quality_mode=preset)
        dummy = torch.randn(1, 3, 64, 64)
        out = model(dummy)
        params = sum(p.numel() for p in model.parameters())
        deploy_kb = model.get_deploy_size_bytes() / 1024
        sparsity = model.get_total_sparsity() * 100
        status = "PASS" if out.shape == (1, 3, 128, 128) else "FAIL"
        print(f"  [{status}] {preset:>12}: {out.shape} | params={params:,} | deploy={deploy_kb:.1f}KB | sparsity={sparsity:.1f}%")

    # --- Test 2: Full Architecture Summary ---
    print("\n[TEST 2] Architecture Summary (quality preset)")
    model = build_myelin_sr(upscale=2, quality_mode="quality")
    model.print_architecture_summary()

    # --- Test 3: Phase Gate 1.1 (Valid Output) ---
    print("\n[TEST 3] Phase Gate 1.1: Valid Output")
    validator = PhaseGateValidator()
    gate_ok = validator.check_valid_output(model, torch.device("cpu"))
    print(f"  {'PASS' if gate_ok else 'FAIL'}: valid output check")

    # --- Test 4: FPSANConv2d Layer Count ---
    print("\n[TEST 4] FPSANConv2d Layer Count")
    n_fpsan = sum(1 for m in model.modules() if isinstance(m, FPSANConv2d))
    print(f"  FPSANConv2d layers: {n_fpsan}")
    assert n_fpsan > 0, "No FPSANConv2d layers found!"
    print("  PASS")

    # --- Test 5: Sleep Consolidation ---
    print("\n[TEST 5] Sleep Consolidation")
    stiff_before = None
    for m in model.modules():
        if isinstance(m, FPSANConv2d):
            stiff_before = m.stiffness.mean().item()
            break
    model.consolidate_all(rate=0.08)
    stiff_after = None
    for m in model.modules():
        if isinstance(m, FPSANConv2d):
            stiff_after = m.stiffness.mean().item()
            break
    print(f"  Stiffness before: {stiff_before:.4f}, after: {stiff_after:.4f}")
    print("  PASS")

    # --- Test 6: Gradient Flow ---
    print("\n[TEST 6] Backward Pass & Gradient Flow")
    model.train()
    dummy_lr = torch.randn(1, 3, 64, 64)
    dummy_hr = torch.randn(1, 3, 128, 128)
    sr = model(dummy_lr)
    loss = torch.nn.functional.l1_loss(sr, dummy_hr)
    loss.backward()

    total_params = 0
    params_with_grad = 0
    for p in model.parameters():
        if p.requires_grad:
            total_params += 1
            if p.grad is not None:
                params_with_grad += 1

    grad_ok = params_with_grad == total_params
    print(f"  Params with gradients: {params_with_grad}/{total_params}")
    status = "PASS" if grad_ok else "FAIL"
    print(f"  {status}: gradient flow")

    # --- Test 7: Ternary Weight Export ---
    print("\n[TEST 7] Ternary Weight Export")
    for m in model.modules():
        if isinstance(m, FPSANConv2d):
            ternary_w = m.get_ternary_weights()
            unique_vals = torch.unique(ternary_w).tolist()
            scale = m.get_weight_scale()
            print(f"  Unique ternary values: {unique_vals}")
            print(f"  Scale factor: {scale:.6f}")
            all_ternary = all(v in [-1, 0, 1] for v in unique_vals)
            status = "PASS" if all_ternary else "FAIL"
            print(f"  {status}: all weights in {{-1, 0, 1}}")
            break

    # --- Test 8: PSNR/SSIM Metric Sanity ---
    print("\n[TEST 8] Metric Functions")
    identical = torch.rand(1, 3, 64, 64)
    psnr_identical = calculate_psnr(identical, identical)
    print(f"  PSNR (identical images): {psnr_identical:.1f} dB (expect ~100)")

    noisy = identical + torch.randn_like(identical) * 0.1
    psnr_noisy = calculate_psnr(noisy.clamp(0, 1), identical)
    print(f"  PSNR (noisy images): {psnr_noisy:.1f} dB (expect ~20-30)")

    ssim_identical = calculate_ssim(identical, identical)
    print(f"  SSIM (identical images): {ssim_identical:.4f} (expect ~1.0)")

    metrics_ok = psnr_identical > 50 and 15 < psnr_noisy < 40 and ssim_identical > 0.99
    status = "PASS" if metrics_ok else "FAIL"
    print(f"  {status}: metrics sanity")

    # --- Test 9: Sparsity Gate ---
    print("\n[TEST 9] Sparsity Gate")
    validator.check_sparsity(model)

    # --- Test 10: Loss Function ---
    print("\n[TEST 10] Loss Function")
    criterion = MyelinSRLoss(perceptual_weight=0.0, use_perceptual=False)
    model.train()
    model.zero_grad()
    sr_out = model(torch.randn(2, 3, 32, 32))
    hr_out = torch.randn(2, 3, 64, 64)
    total_loss, loss_dict = criterion(sr_out, hr_out)
    print(f"  L1 loss: {loss_dict['l1']:.4f}")
    print(f"  Total loss: {loss_dict['total']:.4f}")
    print(f"  Loss requires grad: {total_loss.requires_grad}")
    status = "PASS" if total_loss.requires_grad else "FAIL"
    print(f"  {status}: loss is differentiable")

    # --- Final Report ---
    validator.print_report()

    print("\n" + "=" * 60)
    print("  ALL TESTS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
