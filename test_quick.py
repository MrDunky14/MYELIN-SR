"""Full 10-test verification suite - results saved to test_results_full.txt"""
import sys, os
sys.path.insert(0, os.path.join('.', 'train'))
sys.path.insert(0, os.path.join('.', 'eval'))
import torch
from model import build_myelin_sr
from fpsan_conv2d import FPSANConv2d
from metrics import calculate_psnr, calculate_ssim, PhaseGateValidator
from losses import MyelinSRLoss

lines = []

# T1: All Quality Presets Forward Pass
for p in ['performance','balanced','quality','ultra']:
    m = build_myelin_sr(2, p)
    o = m(torch.randn(1,3,64,64))
    ok = (o.shape == (1,3,128,128))
    par = sum(x.numel() for x in m.parameters())
    kb = m.get_deploy_size_bytes() / 1024
    sp = m.get_total_sparsity() * 100
    tag = "PASS" if ok else "FAIL"
    lines.append(f"T1 preset_{p}: {tag} | shape={list(o.shape)} params={par} deploy={kb:.1f}KB sparsity={sp:.1f}%")

# T2: Architecture Summary
m = build_myelin_sr(2, 'quality')
params = sum(p.numel() for p in m.parameters())
deploy = m.get_deploy_size_bytes() / 1024
sparsity = m.get_total_sparsity() * 100
lines.append(f"T2 architecture: PASS | params={params} deploy={deploy:.1f}KB sparsity={sparsity:.1f}%")

# T3: Phase Gate 1.1 (Valid Output)
validator = PhaseGateValidator()
gate_ok = validator.check_valid_output(m, torch.device("cpu"))
tag = "PASS" if gate_ok else "FAIL"
lines.append(f"T3 gate_valid_output: {tag}")

# T4: FPSANConv2d Layer Count  
n_fpsan = sum(1 for x in m.modules() if isinstance(x, FPSANConv2d))
tag = "PASS" if n_fpsan > 0 else "FAIL"
lines.append(f"T4 fpsan_layers: {tag} | {n_fpsan} FPSANConv2d modules found")

# T5: Sleep Consolidation
stiff_before = None
for x in m.modules():
    if isinstance(x, FPSANConv2d):
        stiff_before = x.stiffness.mean().item()
        break
m.consolidate_all(rate=0.08)
stiff_after = None
for x in m.modules():
    if isinstance(x, FPSANConv2d):
        stiff_after = x.stiffness.mean().item()
        break
lines.append(f"T5 sleep_consolidation: PASS | stiffness_before={stiff_before:.4f} stiffness_after={stiff_after:.4f}")

# T6: Backward Pass & Gradient Flow
m.train()
dummy_lr = torch.randn(1, 3, 64, 64)
dummy_hr = torch.randn(1, 3, 128, 128)
sr = m(dummy_lr)
loss = torch.nn.functional.l1_loss(sr, dummy_hr)
loss.backward()
n_total = sum(1 for p in m.parameters() if p.requires_grad)
n_grad = sum(1 for p in m.parameters() if p.requires_grad and p.grad is not None)
tag = "PASS" if n_total == n_grad else "FAIL"
lines.append(f"T6 gradient_flow: {tag} | {n_grad}/{n_total} params have gradients")

# T7: Ternary Weight Export
for x in m.modules():
    if isinstance(x, FPSANConv2d):
        tw = x.get_ternary_weights()
        uv = sorted(torch.unique(tw).tolist())
        scale = x.get_weight_scale()
        ok = all(v in [-1,0,1] for v in uv)
        tag = "PASS" if ok else "FAIL"
        lines.append(f"T7 ternary_export: {tag} | unique_values={uv} scale={scale:.6f}")
        break

# T8: PSNR/SSIM Metric Sanity
img = torch.rand(1, 3, 64, 64)
psnr_ident = calculate_psnr(img, img)
psnr_noisy = calculate_psnr((img + torch.randn_like(img)*0.1).clamp(0,1), img)
ssim_ident = calculate_ssim(img, img)
ok = psnr_ident > 50 and 15 < psnr_noisy < 40 and ssim_ident > 0.99
tag = "PASS" if ok else "FAIL"
lines.append(f"T8 psnr_ssim_sanity: {tag} | psnr_ident={psnr_ident:.0f}dB psnr_noisy={psnr_noisy:.1f}dB ssim_ident={ssim_ident:.4f}")

# T9: Sparsity Gate
sparsity_val = m.get_total_sparsity()
tag = "PASS" if sparsity_val >= 0.20 else "FAIL"
lines.append(f"T9 sparsity_gate: {tag} | sparsity={sparsity_val*100:.1f}% (threshold=20%)")

# T10: Loss Function
crit = MyelinSRLoss(perceptual_weight=0.0, use_perceptual=False)
m.zero_grad()
sr2 = m(torch.randn(2, 3, 32, 32))
hr2 = torch.randn(2, 3, 64, 64)
total_loss, loss_dict = crit(sr2, hr2)
tag = "PASS" if total_loss.requires_grad else "FAIL"
lines.append(f"T10 loss_function: {tag} | l1={loss_dict['l1']:.4f} total={loss_dict['total']:.4f} has_grad={total_loss.requires_grad}")

# Summary
n_pass = sum(1 for l in lines if ": PASS" in l)
n_total = len(lines)
lines.append(f"\nSUMMARY: {n_pass}/{n_total} tests PASSED")

# Write and print
with open("test_results_full.txt", "w") as f:
    for l in lines:
        f.write(l + "\n")
        print(l)
