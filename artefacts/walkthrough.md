# MYELIN-SR Phase 1: Implementation Walkthrough

## What Was Built

8 production files implementing the MYELIN-SR spatial super-resolution pipeline:

| File | Purpose | Lines |
|---|---|---|
| [fpsan_conv2d.py](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/train/fpsan_conv2d.py) | [FPSANConv2d](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/train/fpsan_conv2d.py#33-236) — ternary QAT conv with Astrocytic Stiffness | ~220 |
| [model.py](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/train/model.py) | [MyelinSRNet](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/train/model.py#217-360) — SR-MoE, TextureEncoder, FP16 head | ~290 |
| [dataset.py](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/train/dataset.py) | DIV2K/Flickr2K loader with augmentation | ~180 |
| [losses.py](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/train/losses.py) | L1 + VGG perceptual loss | ~90 |
| [train.py](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/train/train.py) | Training loop with Homeostatic LR, Sleep, Phase Gates | ~270 |
| [metrics.py](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/eval/metrics.py) | PSNR/SSIM + 6-gate PhaseGateValidator | ~260 |
| [benchmark.py](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/eval/benchmark.py) | Comparison vs Bicubic/EDSR/SwinIR baselines | ~180 |
| [verify_phase1.py](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/verify_phase1.py) | Verification suite | ~100 |

## Verification Results

### pytest — 26/26 PASSED (6.19s)

```
tests/test_architecture.py   — 8 tests  PASSED
tests/test_metrics_loss.py   — 7 tests  PASSED  
tests/test_pipeline.py       — 11 tests PASSED
=============== 26 passed in 6.19s ==============
```

Coverage:
- All 4 quality presets: correct output shapes, no NaN/Inf
- Exact parameter counts per preset hard-coded and verified
- 21 FPSANConv2d layers confirmed
- 62/62 params receive gradients (full backprop flow)
- Ternary export produces exactly {-1, 0, 1}
- PSNR/SSIM sanity: 100dB identical, 25dB noisy, SSIM=1.0
- Loss forward+backward differentiable
- Full train cycle (forward → loss → backward → step) completes
- Eval mode with batch sizes 1/2/4/8 all pass
- Deploy size under 200KB for all presets
- PhaseGate 1.1 (valid output) and 1.6 (sparsity) both pass


## Key Architecture Stats (Quality Preset)

| Metric | Value |
|---|---|
| Total parameters | 39,488 |
| Deploy size (ternary packed) | **37.7 KB** |
| Ternary sparsity | 24.8% (expected to increase with training) |
| FPSANConv2d layers | 21 |
| Quality presets | performance / balanced / quality / ultra |
| Upscale factor | 2x (64x64 -> 128x128 verified) |

> For comparison: EDSR-baseline is **1.5 MB**, SwinIR-light is **900 KB**. Our quality preset is **37.7 KB** — 40x smaller.

## Novel Features Verified Working

1. **Ternary QAT forward pass** — weights quantize to exact {-1, 0, 1} during forward, gradients flow through FP32 master weights via STE
2. **Astrocytic Stiffness** — [consolidate_all()](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/train/model.py#304-312) runs across all 21 layers
3. **Spike-Routed MoE** — 4 experts with spiking router (soft routing in train, hard in eval)
4. **Texture State Encoder** — 8-neuron combinatorial encoding
5. **FP16 reconstruction head** — PixelShuffle with higher precision for color fidelity
6. **Phase Gate Validator** — 6 quality gates (valid output, beats bicubic, beats Lanczos, matches EDSR, ternary quality, sparsity)

## Next Steps

1. **Export script** (`train/export_model.py`) — TorchScript + ternary binary for C++ inference
2. **Train on Kaggle T4** — Upload [train/](file:///k:/MyProjects/MY_DLSS/core_dump/FP-SAN/src/colab_fpsan_proof.py#108-116) and [eval/](file:///k:/MyProjects/MY_DLSS/core_dump/FP-SAN/src/colab_fpsan_proof.py#128-134) dirs, download DIV2K, run [train.py](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/train/train.py)
3. **Benchmark** — Run [eval/benchmark.py](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/eval/benchmark.py) after training to hit Phase Gates
