# FP-SAN / MYELIN-SR — Full Codebase Autopsy

> **Goal:** Identify every architectural flaw, algorithmic mistake, and engineering gap preventing this project from becoming a viable DLSS/FSR replacement.

---

## Executive Summary

Your **vision** is genuinely ambitious — a zero-training, neuromorphic, ternary-weight super-resolution engine that runs on any GPU without Tensor Cores. That's a real research frontier.

But the **current implementation** has **fundamental gaps** at every layer: the core algorithm is mathematically equivalent to a hand-tuned edge-sharpening filter (not a neural network), the GPU pipeline is entirely scaffolding, there are zero quantitative quality metrics, and several claims in the docs don't match the code. Below is the brutally honest breakdown.

---

## 🔴 CRITICAL: Algorithmic Flaws (The Math Is Wrong)

### 1. This Is Not a Neural Network — It's a Handcrafted Filter

The core of [execute_single_pixel()](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/src/nss_core.cpp#26-94) in [nss_core.cpp](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/src/nss_core.cpp#L26-L93):

```
For each pixel in a 3x3 patch:
  if (close to target)        → weight = +1
  if (brighter than avg AND close) → weight = +1
  if (darker than avg AND far)     → weight = -1
  else                             → weight = 0
  
  sum = Σ(weight × pixel_color)
  output = sum × variance_scalar / divisor
```

**This is mathematically equivalent to a variance-adaptive weighted average with edge sharpening.** It has:
- **No learned parameters** — the thresholds (`0.02f`, `0.6f`, `1.5f`, `0.25f`, `4.0f`) are all hardcoded magic numbers
- **No training loop** — not even a single gradient update
- **No latent representation** — there's no hidden layer, no feature extraction
- **No multi-scale analysis** — only a single 3×3 window, so it can't reason about structures larger than 3 pixels

> [!CAUTION]
> A 3×3 static geometric kernel with hardcoded thresholds cannot outperform Lanczos, let alone DLSS. The whitepaper claims to "match and exceed offline-trained interpolation filters" — but mathematically, this IS a simpler version of those filters.

### 2. Variance Scalar Creates Catastrophic Artifacts

```cpp
float cluster_scalar = 1.0f + (local_variance * 4.0f);  // Line 84
```

This **amplifies noise in high-variance regions**. A dark scene with a few bright specks (e.g., stars, particles, UI text on dark backgrounds) will have extreme `local_variance`, causing the scalar to blow up pixel values. The [clamp(0,1)](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/src/nss_core.cpp#17-21) at line 89-91 catches the overflow but **clips it to white**, destroying all highlight detail.

### 3. Inhibitory Weights Create Negative Colors

```cpp
raw_ternary_sum_r -= low_res_input[i].r * 0.25f;  // Line 77
```

Subtracting `0.25 × pixel_value` from the sum can drive `raw_ternary_sum` negative. After dividing by `biological_divisor` (which floors at `0.2f`), the negative-over-small-positive can amplify to large negative values. The final `max(0.0f, ...)` clamp catches this but **crushes the entire channel to black** — a silent data destruction.

### 4. The "Biological Divisor" Is Numerically Unstable

```cpp
float biological_divisor = (float)active_synapses - (inhibitory_synapses * 0.25f);
if (biological_divisor < 0.2f) biological_divisor = 0.2f;  // Lines 85-86
```

With a 3×3 patch most cases will have `active_synapses ∈ {1..5}`. The divisor swings wildly between `0.2` and `5.0`, causing **enormous brightness variation** between adjacent output pixels. This is visible as a "popcorn" or "sparkle" artifact pattern.

---

## 🔴 CRITICAL: Temporal Pipeline Flaws

### 5. No Motion Vectors — Temporal Accumulation Without Motion Compensation = Blur

The manifesto explicitly states:
> *"Since this offline execution lacks Native Engine Motion Vectors (Optical Flow)..."*

Without motion vectors, your temporal buffer at [nss_core.cpp:169](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/src/nss_core.cpp#L169) accumulates pixels at **fixed screen positions**. Any camera movement or object movement will average the old and new content together, producing:
- **Ghosting** on moving objects
- **Smearing** during camera pans
- **Double-images** at disocclusion boundaries

The `voltage_divergence > 0.2f` threshold at line 229 catches *extreme* changes but misses subtle motion (slow pans, gradual animations), which is exactly where DLSS excels and where your engine will visibly fail.

### 6. The "8-Frame Exponential Lag" Is Not Exponential

```cpp
temporal_weight[out_idx] = min(8.0f, temporal_weight[out_idx] + 1.0f);  // Line 233
blend_alpha = 1.0f / temporal_weight[out_idx];  // Line 236
```

This is **linear accumulation** (weight grows by 1 each frame), not exponential. The blend factor sequence is `1/1, 1/2, 1/3, 1/4...1/8`. This means frame 7 contributes only 1/56th as much as frame 1 — meaning old data persists for far too long, creating visible ghosting trails even in static scenes with minor lighting changes.

### 7. Neighborhood Variance Clamping Operates on Wrong Space

The min/max bounding box at lines 208-214 clamps history to the current frame's 3×3 patch color range. But this patch is from the **low-resolution input**, while the temporal buffer stores **upscaled output**. These color ranges won't match properly when `cluster_scalar` amplifies the upscaled values beyond the raw input range.

---

## 🟡 Major: GPU Pipeline Is Non-Functional

### 8. The DX12 Sandbox Is 100% Commented Out

[dx12_compute_sandbox.cpp](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/src/dx12_compute_sandbox.cpp) — every single DX12 API call is a comment. There is no:
- Device creation
- Command queue/list
- Resource allocation
- Pipeline state
- Fence synchronization
- Swapchain integration

**This file does nothing.** It cannot be compiled against DX12 headers.

### 9. HLSL Shader Has Critical Bugs

In [nss_compute.hlsl](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/src/nss_compute.hlsl):

- **Line 58:** `uint2 lowResCoord = texCoord / 2;` — This hardcodes a 2× scale factor. The C++ code supports arbitrary `scale_factor` but the shader is locked to 2×.

- **Lines 112-115:** The sub-pixel offset calculation uses `texCoord % 2`, which only produces offsets `{0,1}` in each axis. For non-2× scales (e.g. 1.5× or 4×), this is completely wrong.

- **Lines 88-95:** The shader reads the 3×3 neighborhood from the **output-resolution** coordinate space (`texCoord ± 1`), but the input texture is low-resolution. This means it's sampling the wrong pixel neighborhood — the patch coordinates don't match the C++ logic at all.

- **The `PackedTernaryWeights` and `ClusterScalars` buffers** (lines 31-32) are declared but **never read** in the shader body. The shader ignores the pre-computed payload and recomputes everything from scratch (which contradicts the "93% bandwidth reduction" claim).

### 10. No Integration Exists

There is no code anywhere that:
- Hooks into a game engine's render pipeline
- Captures game frames (no `IDXGISwapChain::Present` hook, no ReShade integration)
- Provides motion vector input
- Provides depth buffer input
- Manages frame pacing or synchronization

---

## 🟡 Major: Quality Validation Is Absent

### 11. Zero Quantitative Metrics

The [Benchmarks.md](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/docs/Benchmarks.md) claims "SSIM/PSNR Parity" but provides **no actual numbers**. There are:
- No PSNR measurements
- No SSIM measurements  
- No LPIPS measurements
- No comparison against bilinear, bicubic, Lanczos, FSR, or DLSS
- No reference ground-truth images

Without metrics, the claim that this approach produces quality "matching" DLSS is unverifiable.

### 12. Tests Directory Is Empty

The `tests/` directory has zero files. There are no unit tests, no regression tests, no automated quality benchmarks.

### 13. The Stress Test Doesn't Test the Algorithm

[generate_stress_test.py](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/generate_stress_test.py) generates test patterns but there's no script that:
1. Runs the upscaler on the pattern
2. Compares output against a reference
3. Reports pass/fail

---

## 🟡 Major: Documentation vs. Reality Gaps

### 14. Claims Not Matched By Code

| Documentation Claim | Code Reality |
|---|---|
| "1.58-bit ternary weights packed into uint32" | Weights are computed on-the-fly from hardcoded thresholds. No packing anywhere in working code. |
| "Predictive Lateral Synapses for motion" | No lateral connectivity exists in the code |
| "Depth-Layered Parallax Neurogenesis" | No depth buffer support anywhere |
| "Background Pattern Learning" | No learning or memory of patterns |
| "Exponential Homeostasis" | Only appears in docs, not in NSS code |
| "93% bandwidth reduction" | The HLSL shader never reads the packed buffer |
| "~0.4ms GPU execution" | GPU pipeline is non-functional, cannot be measured |
| "Event-driven sparsity (5-10% of pixels)" | Every pixel is processed in both C++ and HLSL |

---

## 🟠 Engineering Issues

### 15. Performance Bottlenecks in C++ Core

- **O(n²) per frame**: For a 2× upscale of 1080p→2160p, the inner loop executes `3840 × 2160 × 9 = ~74M` iterations per frame. On a single CPU thread without SIMD, this is inherently ~250ms/frame (matching your benchmarks), which is **far too slow for real-time**.

- **No SIMD vectorization**: The inner loop could use SSE/AVX to process 4/8 pixels in parallel, easily gaining 4-8× speedup.

- **No multithreading**: The frame loop is single-threaded. Splitting rows across threads would give near-linear speedup.

### 16. `using namespace std` in Header

[nss_core.hpp:8](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/include/nss_core.hpp#L8) — `using namespace std;` in a header file pollutes the namespace of every file that includes it. This will cause name collisions in any non-trivial project.

### 17. No Error Handling for Scale Factor Edge Cases

- `scale_factor = 0` → output dimensions are 0×0 → silent empty output
- `scale_factor < 0` → integer underflow / crash
- `scale_factor = 1.0` → expensive identity operation (every pixel recomputed for no upscale)

---

## What Would It Actually Take to Compete With DLSS?

To put this in perspective, here's what DLSS 3.5 actually does under the hood:

| Component | DLSS | Your Engine (current state) |
|---|---|---|
| Input data | Color + Depth + Motion Vectors + Exposure | Color only |
| Spatial model | Multi-scale CNN (trained on millions of frames) | 3×3 weighted average with edge sharpening |
| Temporal model | Motion-compensated history + confidence maps | Fixed-position accumulation buffer |
| Training | Offline supervised training (ground truth = native resolution) | None |
| GPU execution | Tensor Core accelerated, 0.5-2ms | Non-functional |
| Quality validation | Extensive PSNR/SSIM/LPIPS benchmarks | None |

---

## Questions for You

Before I propose a path forward, I need to understand your intent:

1. **What is your actual quality target?** Do you want to beat bilinear interpolation? Match Lanczos? Match FSR 1.0? Match DLSS Quality mode? Each of these is a fundamentally different difficulty level.

2. **Are you committed to the "no training" constraint?** The ternary-weight approach is interesting, but without any form of optimization (even non-gradient methods like evolutionary search or Bayesian optimization over your thresholds), the magic numbers will always be suboptimal. Would you consider a lightweight offline tuning phase?

3. **What GPU do you actually have access to?** This determines whether we should target DX12 compute, Vulkan compute, or even CUDA directly.

4. **Do you want to integrate with real games first, or nail the quality on static images first?** These are very different engineering paths. My recommendation: images first, temporal second, game integration last.

5. **Would you accept using optical flow estimation (like Farneback or RAFT-lite) for motion vectors in the offline pipeline?** This would immediately fix the temporal ghosting problem and is what FSR 2.0 does when game-provided MVs aren't available.

6. **How do you feel about the "biological" framing?** The neuroscience terminology (synapses, voltage, endocrine, cortex) in the code doesn't change the math — a weighted average is a weighted average regardless of what you call the variables. Would you be open to stripping the metaphor and working directly with the signal processing primitives? This would make the code dramatically easier to debug and optimize.
