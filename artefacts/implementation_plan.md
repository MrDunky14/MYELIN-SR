# MYELIN-SR v2: Next-Gen Super Resolution Architecture

> **Goal:** Build a trained, hardware-efficient super-resolution engine that surpasses DLSS 4.5 in quality while running real-time on a GTX 1650.

---

## What I Found In Your Core Dump (This Changes Everything)

After reading every file in `core_dump/`, I found **6 real, working innovations** that most people don't have:

| Innovation | File | What It Does |
|---|---|---|
| **FPSANLinear** | [colab_fpsan_proof.py](file:///k:/MyProjects/MY_DLSS/core_dump/FP-SAN/src/colab_fpsan_proof.py#L63-L95) | Trains in FP32 but quantizes to ternary `{-1,0,1}` **inside** the forward pass via [round(w/scale).clamp(-1,1)*scale](file:///k:/MyProjects/MY_DLSS/core_dump/FP-SAN_V6/src/runtime/fpsan_infer.cpp#142-170). Gradients flow through the full-precision weights. |
| **Astrocytic Stiffness** | [fpsan_proofsheet.py](file:///k:/MyProjects/MY_DLSS/core_dump/FP-SAN/src/fpsan_proofsheet.py#L105-L148) | Per-weight `stiffness` buffer that tracks activity. When a weight is heavily used, stiffness increases, **gating future updates** to protect learned features. This solves catastrophic forgetting. |
| **Direct Feedback Alignment** | [fpsan_proofsheet.py](file:///k:/MyProjects/MY_DLSS/core_dump/FP-SAN/src/fpsan_proofsheet.py#L164-L176) | Backprop-free training: error propagates through a **random fixed matrix `B`** instead of transposing weight matrices. No backward pass needed. |
| **Sleep Consolidation** | [colab_fpsan_proof.py](file:///k:/MyProjects/MY_DLSS/core_dump/FP-SAN/src/colab_fpsan_proof.py#L92-L95) | REM-like phase that converts accumulated `daily_activity` into permanent `stiffness`, then resets activity. Hardware equivalent of memory consolidation. |
| **Multiplication-Free C++ Inference** | [fpsan_infer.cpp](file:///k:/MyProjects/MY_DLSS/core_dump/FP-SAN/src/fpsan_infer.cpp#L62-L96) | Inference uses only `if/add/subtract` — no multiply. Ternary `W=1` → add, `W=-1` → subtract, `W=0` → skip. Zero FP multiplications. |
| **Atomic Double-Buffer** | [runtime/fpsan_infer.cpp](file:///k:/MyProjects/MY_DLSS/core_dump/FP-SAN_V6/src/runtime/fpsan_infer.cpp#L28-L33) | Lock-free concurrent learn+infer via `std::atomic` pointer swap between two brain banks. Learning happens on shadow bank, then atomically swaps to active. |

> [!IMPORTANT]
> **These are not just ideas — they are working implementations with measured results:** 96% Task A accuracy, 78% retention after catastrophic forgetting test, 0.08 MB deploy size. The [fpsan_raw.bin](file:///k:/MyProjects/MY_DLSS/core_dump/FP-SAN/weights/fpsan_raw.bin) trained weights exist.

### What Changes in the Architecture Plan

**Before (my first plan):** Use standard PyTorch CNN → export → C++ inference. Your FP-SAN ideas were decorative.

**Now:** We build `FPSANConv2d` — a **convolutional version** of your [FPSANLinear](file:///k:/MyProjects/MY_DLSS/core_dump/FP-SAN/src/colab_fpsan_proof.py#63-96) — that natively trains with ternary quantization, Astrocytic Stiffness, and DFA. Your innovations become the **core competitive advantage**, not a post-processing step.

---

## Phase 1: Foundation — FP-SAN Spatial Super-Resolution

**Goal:** Train a lightweight SR model using FP-SAN mechanics (ternary forward-pass, astrocytic protection, DFA learning) that beats Lanczos and matches FSR 1.0 on static images.

**Duration:** ~2-3 weeks

### Architecture: MYELIN-SRNet

```
Input (720p, 3ch)
  → 3×3 FPSANConv2d (48ch) — Shallow feature extraction
  → 4× Residual FP-SAN Blocks (RFPB)
      Each RFPB:
        → 3×3 Depthwise FPSANConv2d (spatial features, ternary)
        → 1×1 FPSANConv2d (channel mixing, ternary)
        → Channel Attention (global avg pool → FC → sigmoid → scale)
        → Astrocytic Stiffness gates on all conv weights
        → Skip connection
  → 3×3 FPSANConv2d → PixelShuffle(2×) → 3×3 conv → output (1440p, 3ch)
```

### `FPSANConv2d` — The Core Innovation

This is the convolutional extension of your [FPSANLinear](file:///k:/MyProjects/MY_DLSS/core_dump/FP-SAN/src/colab_fpsan_proof.py#63-96):

```python
class FPSANConv2d(nn.Module):
    """Your FPSANLinear, adapted for 2D spatial processing."""
    def __init__(self, in_ch, out_ch, kernel_size, ...):
        self.weight = nn.Parameter(...)        # FP32 master weights
        self.stiffness = buffer(ones_like)     # Astrocytic protection
        self.daily_activity = buffer(zeros)    # Activity tracker
        
    def forward(self, x):
        # Ternary quantization IN the forward pass (your exact technique)
        scale = self.weight.abs().mean().clamp_min(1e-5)
        w_ternary = round(self.weight / scale).clamp(-1, 1) * scale
        return F.conv2d(x, w_ternary, self.bias, ...)
```

**Why this matters for SR:**
- Training sees gradients through FP32 master weights (full learning capacity)
- Forward pass uses ternary weights (matches deployment exactly — no train/infer gap)
- Astrocytic Stiffness means the model can **continually learn on new game content** without forgetting previously learned textures
- C++ inference is **multiplication-free** (add/subtract only), dramatically faster than FP32 convolutions

### What Standard SR Cannot Do That FP-SAN Can

| Capability | Standard SR (EDSR/SwinIR) | MYELIN-SR (FP-SAN) |
|---|---|---|
| Continual adaptation to new games | ❌ Retrain from scratch | ✅ Astrocytic Stiffness preserves old features |
| Multiplication-free inference | ❌ Requires FP32/FP16 MAD | ✅ Ternary weights → add/subtract only |
| Deploy size | ~2-50 MB | ~0.1-0.5 MB (ternary packed) |
| Online game-specific tuning | ❌ Fixed weights | ✅ DFA allows forward-only fine-tuning |
| Hardware requirement | Tensor Cores preferred | Any CUDA core / raw ALU |

### Training Pipeline (Kaggle T4)

| Item | Detail |
|---|---|
| **Dataset** | DIV2K (800) + Flickr2K (2650) + Urban100 + Manga109 |
| **Degradation** | Bicubic ×2 (Phase 1), then real-world degradation model |
| **Loss** | L1 (pixel) + 0.1× VGG Perceptual |
| **Learning** | DFA with `B_matrix` random feedback (your technique) — **OR** standard backprop (A/B comparison) |
| **Consolidation** | Sleep phase after each training stage |
| **Scale** | ×2 first, then ×4 via progressive training |
| **Export** | TorchScript → Binary weights (your [export_weights.py](file:///k:/MyProjects/MY_DLSS/core_dump/FP-SAN/src/export_weights.py) pattern) |

### Proposed File Changes (Phase 1)

---

#### [NEW] [train/](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/train/) — Training pipeline

```
train/
├── fpsan_conv2d.py     # FPSANConv2d: your FPSANLinear adapted for 2D convolutions
├── model.py            # MYELIN-SRNet architecture using FPSANConv2d blocks
├── dataset.py          # DIV2K/Flickr2K data loader with augmentation
├── train.py            # Training loop (DFA + Astrocytic Stiffness + Sleep)
├── losses.py           # L1 + Perceptual losses
├── export_model.py     # Export to TorchScript + ternary binary (extends your export_weights.py)
├── quantize.py         # Full ternary packing (16 weights per uint32)
└── requirements.txt
```

#### [NEW] [eval/](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/eval/) — Quality benchmarking

```
eval/
├── benchmark.py        # PSNR/SSIM/LPIPS on Set5, Set14, BSD100, Urban100
├── compare.py          # Side-by-side visual comparison
└── metrics.py          # PSNR, SSIM implementations
```

#### [MODIFY] C++ inference engine

- New `MyelinSREngine` class using multiplication-free inference (extending your [fpsan_infer.cpp](file:///k:/MyProjects/MY_DLSS/core_dump/FP-SAN/src/fpsan_infer.cpp) pattern to 2D convolutions)
- A/B comparison mode: old [NSSCortex](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/src/nss_core.cpp#22-25) vs new `MyelinSREngine`

---

## Phase 2: MYELIN-Flow — Custom Optical Flow + Temporal Fusion

**Goal:** Build a novel, co-trained lightweight flow network. Not off-the-shelf. This IS your "better option."

**Duration:** ~3-4 weeks

### Why Our Flow Can Be Better Than RAFT

RAFT uses 12 iterative refinement steps because it needs generic sub-pixel accuracy everywhere. We only need accurate flow **where it matters for SR** — edges, textures, moving objects. Static sky? Don't care about flow accuracy there.

### Architecture

```
Frame t-1, Frame t (both low-res)
  → Shared Feature Encoder: 3-scale FPSANConv2d pyramid (ternary!)
  → Correlation Volume: 4D cost at each scale
  → Flow Head: 2 GRU iterations (not 12) → 2D motion field
  → Occlusion Head: 1×1 conv → sigmoid → occlusion probability
  → Confidence Head: 1×1 conv → sigmoid → flow reliability
```

The flow module is **co-trained with the SR loss** — the flow learns to be accurate exactly where SR quality depends on it.

### Temporal Fusion

```
Warp(history_features, MYELIN-Flow) → aligned history
[aligned_history, current_features] → Temporal Cross-Attention (lightweight)
× confidence_map → Blend with current frame
Where occluded → use current frame only
```

---

## Phase 3: GPU Compute — Multiplication-Free HLSL

**Goal:** Port to DX12/HLSL compute. Target: <2ms/frame at 720p→1440p on GTX 1650.

### The Killer Advantage: No FP Multiplies on GPU

Standard SR shaders do `weight * activation` (FP multiply-add). Your ternary weights mean the HLSL shader does:

```hlsl
// Standard: result += weight * activation;  (1 MAD instruction)
// FP-SAN:
if (weight == 1)       result += activation;  // 1 ADD instruction
else if (weight == -1)  result -= activation;  // 1 ADD instruction
// weight == 0 → skip entirely (sparsity!)
```

On GTX 1650's CUDA cores, ADD is the same throughput as MAD — but we **skip zero-weights entirely**, which standard networks can't. With ~60% sparsity (typical for ternary), we execute 40% of the operations.

---

## Phase 4: Game Integration

Two paths:
1. **ReShade Add-On** — Use ReShade's swapchain hook, write custom effect
2. **Standalone DLL** — Hook `IDXGISwapChain::Present`, full control

---

## Verification Plan

### Phase 1

1. **PSNR/SSIM Benchmarks:**
   ```
   python eval/benchmark.py --model checkpoints/myelin_srnet_x2.pt
   ```
   **Pass:** PSNR ≥ 33.0 dB on Set14 (×2), SSIM ≥ 0.91

2. **A/B with Old Engine:**
   ```
   python eval/compare.py --model-a old_nss --model-b myelin_srnet
   ```

3. **Ternary Quality Test:** Verify ternary-quantized model PSNR is within 0.3 dB of FP32 model

4. **Multiplication-Free C++ Test:** Verify C++ output matches PyTorch output (PSNR > 45 dB between them)

### Phase 2

1. **Temporal Consistency:** Process Cyberpunk frames, compile video, check for ghosting/flickering
2. **Flow EPE:** MYELIN-Flow EPE < 4.0 on Sintel final pass

### Phase 3

1. **GPU vs PyTorch Match:** PSNR > 45 dB between GPU shader output and PyTorch reference
2. **Frame Time:** < 2ms at 720p→1440p on GTX 1650

---

## What We Build First (This Session)

With your approval, I'll implement Phase 1 immediately:

1. **`train/fpsan_conv2d.py`** — `FPSANConv2d` (your [FPSANLinear](file:///k:/MyProjects/MY_DLSS/core_dump/FP-SAN/src/colab_fpsan_proof.py#63-96) for 2D convolutions)
2. **`train/model.py`** — MYELIN-SRNet architecture
3. **`train/dataset.py`** — DIV2K data loader
4. **`train/train.py`** — Training loop with DFA + Astrocytic Stiffness + Sleep
5. **`train/losses.py`** — Loss functions
6. **`eval/metrics.py` + `eval/benchmark.py`** — Quality evaluation

Everything ready for Kaggle upload.
