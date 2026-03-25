# MYELIN-SR: Next-Gen Super Resolution Engine

## Phase 0: Research & Planning
- [x] Full codebase analysis — identified 17 issues across 6 categories
- [x] Deep dive core_dump — found 6 real FP-SAN innovations
- [x] Architecture plan v1 (standard CNN) → v2 (FP-SAN native)
- [x] Feasibility assessment & architectural questions for Krishna
- [x] Finalize architecture decisions based on Krishna's answers

## Phase 1: Foundation — FP-SAN Spatial Super-Resolution
- [x] [train/fpsan_conv2d.py](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/train/fpsan_conv2d.py) — FPSANConv2d (2D conv version of FPSANLinear)
- [x] [train/model.py](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/train/model.py) — MYELIN-SRNet architecture (SR-MoE, TextureEncoder, FP16 head)
- [x] [train/dataset.py](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/train/dataset.py) — DIV2K/Flickr2K data loader
- [x] [train/train.py](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/train/train.py) — Training loop (Homeostatic LR + Astrocytic Stiffness + Sleep)
- [x] [train/losses.py](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/train/losses.py) — L1 + VGG Perceptual losses
- [x] [eval/metrics.py](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/eval/metrics.py) — PSNR/SSIM + PhaseGateValidator (6 quality gates)
- [x] [eval/benchmark.py](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/eval/benchmark.py) — Comparison against Bicubic/EDSR/SwinIR baselines
- [x] [verify_phase1.py](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/verify_phase1.py) — Verification suite (7/7 tests PASS)
- [x] [train/export_model.py](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/train/export_model.py) — Ternary binary (2-bit packed) + TorchScript export
- [x] [train/kaggle_train.py](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/train/kaggle_train.py) — Full Kaggle T4 training notebook (10 cells)
- [ ] Train on Kaggle T4 — target PSNR >= 33.0 dB on Set14 (x2)
- [ ] A/B comparison: MYELIN-SR vs old NSSCortex vs bicubic/Lanczos

## Phase 2: Temporal — MYELIN-Flow + Temporal Fusion
- [ ] `train/flow_model.py` — MYELIN-Flow architecture
- [ ] `train/temporal_model.py` — Temporal fusion module
- [ ] `train/train_temporal.py` — Joint flow+SR training on REDS/Vimeo-90K
- [ ] Temporal video test on Cyberpunk frames

## Phase 3: GPU Compute — Real-Time HLSL
- [ ] `src/myelin_compute.hlsl` — Multiplication-free GPU compute shader
- [ ] [src/dx12_compute_sandbox.cpp](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/src/dx12_compute_sandbox.cpp) — Working DX12 pipeline
- [ ] `tools/export_hlsl_weights.py` — Weight export for GPU
- [ ] Benchmark: target <2ms/frame on GTX 1650

## Phase 4: Game Integration
- [ ] ReShade add-on or DLL injection
- [ ] Motion vector + depth buffer integration
- [ ] Live game testing
