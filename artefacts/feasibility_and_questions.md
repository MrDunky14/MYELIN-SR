# MYELIN-SR: Feasibility Assessment & Architectural Questions

## Honest Reality: How Close Does This Get Us?

I'll assess across the 5 axes that define a DLSS competitor:

### Axis 1: Spatial Image Quality

| Milestone | PSNR (Set14 ×2) | Status |
|---|---|---|
| Bicubic interpolation | 30.24 dB | ✅ You're here now (NSSCortex is ~this level) |
| Lanczos | ~30.5 dB | Phase 1 target floor |
| **FSR 1.0** | ~31.5 dB | **Phase 1 realistic target** |
| EDSR-baseline (trained CNN) | 33.26 dB | Phase 1 stretch goal |
| SwinIR-light | 33.81 dB | Possible with good architecture |
| **DLSS Quality mode** | ~34-35 dB* | Phase 1 optimistic target |
| **DLSS 4.5 Ultra Quality** | ~36+ dB* | Multi-phase, likely Phase 2+ |

*DLSS numbers are estimated — NVIDIA doesn't publish exact PSNR benchmarks.

**Phase 1 honest prediction:** With `FPSANConv2d` + ternary quantization, we'll likely land at **31.5-33.5 dB** on Set14 (×2). This beats Lanczos/FSR 1.0 convincingly but may be ~1-2 dB below a standard FP32 CNN of the same size due to the ternary constraint.

**The ternary quality tax:** Quantizing weights to {-1, 0, 1} inherently loses information. Your [scale](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/src/dx12_compute_sandbox.cpp#62-74) trick partially compensates, but expect a 0.3-1.0 dB PSNR drop vs FP32. The question is whether the other advantages (speed, continual learning, tiny deploy size) justify this.

### Axis 2: Temporal Quality (Motion + Stability)

| Milestone | Status |
|---|---|
| No temporal (single-frame) | Phase 1 |
| Fixed-position accumulation (current NSSCortex) | ❌ Produces ghosting |
| **Basic flow + warp** | Phase 2 minimum |
| **MYELIN-Flow co-trained** | Phase 2 target |
| DLSS-level temporal (game MVs + depth + transformers) | Phase 2-3 |

**Phase 2 honest prediction:** MYELIN-Flow with 2 iterations will be **faster** than RAFT but **less accurate** on generic flow benchmarks. The co-training advantage means it'll be surprisingly good where it matters for SR. Temporal quality will match FSR 2.0-level, possibly exceeding it in static scenes thanks to Astrocytic stability.

### Axis 3: Performance

| Target | Frame Time | Status |
|---|---|---|
| DLSS (Tensor Cores) | 0.5-2ms | Hardware advantage we can't match |
| Our GPU target | <2ms | Phase 3 (multiplication-free HLSL) |
| Our CPU fallback | ~50ms | Phase 1 C++ inference |
| Current NSSCortex | ~250ms | ❌ Way too slow |

**Phase 3 honest prediction:** Multiplication-free ternary shaders with ~60% weight sparsity should hit **1-3ms** on GTX 1650. This is competitive with DLSS on non-Tensor-Core hardware, which is exactly the market gap.

### Axis 4: Ecosystem & Integration

This is where DLSS has a **massive lead** — years of game developer adoption, Streamline SDK, driver-level support. We're starting from zero. ReShade integration is the fastest path to real games.

### Axis 5: Novelty (Where We Can Actually WIN)

> [!TIP]
> **These are capabilities NO current upscaler has:**

1. **Continual game-specific adaptation** — Download MYELIN-SR, it plays your game for 10 minutes learning that game's textures, then upscales better than DLSS for that specific title. Astrocytic Stiffness prevents forgetting previous games.

2. **Zero Tensor Core requirement** — Runs on GTX 1650, GTX 1060, integrated graphics. DLSS requires RTX. FSR doesn't learn.

3. **0.1 MB deploy size** — DLSS's model is multiple MB. Ternary packing makes distribution trivial.

4. **Open-source, no vendor lock** — Anyone can integrate, modify, contribute.

---

## The Competitive Gap (Brutally Honest)

```
Quality Scale (PSNR on standard benchmarks):

Bicubic ─────── FSR 1.0 ──── EDSR ───── SwinIR ── DLSS 4.5
  30.2            31.5         33.3       33.8     ~35-36
   │                │            │                    │
   └── You (now) ──┘            │                    │
                     └── Phase 1 target ──┘          │
                                          └── Phase 2+ aspiration
```

**To surpass DLSS 4.5, we need breakthroughs in at least 2 of these:**
1. A ternary architecture that approaches FP32 quality (0.3 dB gap, not 1.0)
2. Temporal fusion that leverages continual learning for per-game specialization
3. A perceptual quality metric where ternary sparsity creates *sharper* edges than float blur

Point 3 is actually plausible — ternary weights naturally produce sharper decisions (hard edges) rather than blurred averages. This could mean lower PSNR but **higher perceptual quality** (better LPIPS score). DLSS sometimes oversmooths; ternary might never do that.

---

## 10 Hard Questions I Need Your Brain For

These are problems I genuinely can't fully solve — they require architectural insight from someone who built the FP-SAN system ground-up.

---

### Q1: DFA Convergence for Pixel-Level Tasks

Your DFA (Direct Feedback Alignment) works great for classification (MNIST) where the output is 10 classes. But super-resolution outputs **millions of pixels** — the error signal is an entire image.

**The problem:** DFA propagates error through `B_matrix` (random fixed). For 10 classes, `B_matrix` is `[hidden × 10]` — small and manageable. For SR, the output layer is `[3 × H × W]` — potentially millions of values. Does DFA even converge for pixel-level regression, or do we need backprop for the SR task and reserve DFA for online fine-tuning only?

**What I need from you:** Your intuition on whether DFA can handle spatial regression, or if we should use standard backprop for the initial heavy training and switch to DFA for the lightweight online game-adaptation (continual learning phase).

---

### Q2: Stiffness Dynamics for Texture Diversity

Your Astrocytic Stiffness was tested on MNIST (10 digit classes, relatively uniform patterns). SR training sees thousands of diverse textures — grass, metal, skin, sky, foliage, text, particle effects.

**The problem:** If stiffness consolidates too aggressively, the early-trained textures (e.g., grass) will freeze the weights and block learning later textures (e.g., metal reflections). Too soft stiffness → catastrophic forgetting returns.

**What I need from you:** What's the right `consolidation_rate` and `stiffness_cap` for a task with thousands of distinct visual patterns instead of 10 digit classes? Should consolidation happen after every epoch, every N batches, or only at explicit phase boundaries?

---

### Q3: Ternary Quantization for Color Precision

Your quantization does [round(weight / scale).clamp(-1, 1) * scale](file:///k:/MyProjects/MY_DLSS/core_dump/FP-SAN_V6/src/runtime/fpsan_infer.cpp#142-170). For classification, this works because the decision boundary just needs the right *direction* (positive/negative). But for SR, we need precise *magnitude* — a sky gradient going from `rgb(110,150,200)` to `rgb(115,155,210)` requires weights that can produce fine 5-unit differences.

**The problem:** With only {-scale, 0, +scale} per weight, can we represent the subtle gradients needed for photorealistic color reproduction? Or do we need selective mixed precision — ternary for spatial feature extraction, FP16 for the final color reconstruction layer?

**What I need from you:** Should the PixelShuffle reconstruction head use full FP32/FP16 weights while the feature backbone stays ternary? Or do you believe the per-channel [scale](file:///k:/MyProjects/MY_DLSS/FP-SAN_NSS/src/dx12_compute_sandbox.cpp#62-74) factor is sufficient for color precision?

---

### Q4: Occlusion Detection Without Depth Buffer

Your manifesto mentions "Depth-Layered Parallax Neurogenesis" for disocclusion. DLSS gets depth from the game engine. In your offline pipeline (and in ReShade without game integration), we have **no depth buffer**.

**The problem:** How do we detect disocclusion (newly revealed pixels) without depth? Options:
- (A) Forward-backward flow consistency check (compute flow both directions, flag disagreements)
- (B) Your "Predictive Lateral Synapse" idea from the manifesto — can you describe the actual mechanism?
- (C) Confidence output from MYELIN-Flow (let the network learn where it's uncertain)
- (D) Something else from your biological model?

**What I need from you:** What's your biological solution for this? The lateral synapse concept sounds novel but wasn't implemented in any code I found.

---

### Q5: Homeostasis for Frame Brightness Adaptation

Your V6 Exponential Homeostasis (`pow(current_energy / expected_energy, 1.5)`) adapts sensitivity based on scene brightness — brilliant for classification in varying lighting. For SR, this creates a different problem:

**The problem:** If the game switches from a dark cave to bright outdoors, your homeostasis briefly over-sensitizes, potentially amplifying noise in the bright frames or crushing highlights. SR needs **exact color fidelity**, not adaptive sensitivity.

**What I need from you:** Should homeostasis control the *learning rate* (how fast we adapt to new game content) rather than the *inference weights* (how we process each frame)? Or is there a way to apply homeostasis to the feature extraction stage only, while keeping the color reconstruction stage stable?

---

### Q6: The Neurogenesis Question for SR

V6 dynamically spawns new clusters when input doesn't match existing ones. For SR, could we dynamically grow new convolutional filters when the model encounters a texture it can't handle well?

**The problem:** Dynamic architecture growth during inference is expensive and complicates the GPU shader. But it could be extremely powerful for the continual learning use case — encounter a new game's art style, grow new specialized filters.

**What I need from you:** Should neurogenesis happen only during the offline "game adaptation" training phase (slow, on CPU), or do you envision it happening in real-time during gameplay?

---

### Q7: Sleep Consolidation Scheduling

In your MNIST setup, sleep happens once between Task A and Task B. For a game SR engine, when does sleep happen?

**Options:**
- (A) Between game sessions (user closes the game → sleep → reopen → better quality)
- (B) During loading screens (triggered by detecting static frames)
- (C) Periodically in a background thread (your atomic double-buffer enables this)
- (D) Never during gameplay; only in explicit "training mode"

**What I need from you:** Which feels right for the user experience?

---

### Q8: Game-Specific vs Universal Weights

Should MYELIN-SR ship as one universal model, or should it train a separate set of weights per game?

**The problem:** A model adapted to Cyberpunk 2077's neon-dark aesthetic might perform worse on Minecraft's blocky bright textures. Astrocytic Stiffness helps retain old knowledge but could prevent full specialization.

**What I need from you:** Your vision — one brain that learns everything, or a "profile" system where each game gets its own weight set initialized from a universal base?

---

### Q9: The Frame Generation Question

DLSS 4.5's biggest performance boost comes from **Multi Frame Generation** — creating 2-5 entirely synthetic frames between real rendered frames. This is fundamentally different from upscaling.

**The problem:** Frame generation requires predicting the *future* visual state — not just upscaling what exists. Your V7 Broca's autoregressive generation (recurrent spikemap cascade) could theoretically be adapted for visual prediction.

**What I need from you:** Do you want to tackle frame generation as Phase 5? If so, should it use Broca-style autoregressive prediction (generate frame N+1 from frame N), or is there a biological mechanism you've thought about for visual extrapolation?

---

### Q10: The Fundamental Belief Test

This question determines the entire project identity:

**Do you believe multiplication-free ternary inference can match FP32 quality for photo-realistic image reconstruction?**

- If **yes**: We go all-in on FP-SAN ternary everywhere. The quality gap becomes the central research challenge we solve.
- If **partially**: We use hybrid precision — ternary backbone + FP16 reconstruction head. Pragmatic and likely to produce the best near-term results.
- If **it's about the other advantages**: We accept a small quality gap but win on speed/size/adaptability/openness, positioning MYELIN-SR as "DLSS for everyone" rather than "DLSS killer."

**What I need from you:** Your honest gut answer. This shapes everything.

---

## Documentation Commitment

From this point forward, I'll maintain:
1. **[task.md](file:///C:/Users/Krishna%20Singh/.gemini/antigravity/brain/22bdb3f9-d509-4009-a03b-f072413677b3/task.md)** — Live checklist (already created)
2. **[implementation_plan.md](file:///C:/Users/Krishna%20Singh/.gemini/antigravity/brain/22bdb3f9-d509-4009-a03b-f072413677b3/implementation_plan.md)** — Architecture decisions (updated with your answers)
3. **`walkthrough.md`** — Created after each phase documenting what was built, tested, and validated
4. **Code comments** — Every file will document the FP-SAN principle it implements
5. **`RESEARCH_LOG.md`** — Experimental results, PSNR numbers, failed approaches, and why

This project deserves a paper trail. Let's build it right.
