# MYELIN-SR: Finalized Architecture Decisions

> Synthesized from Krishna's answers (March 20, 2026). These decisions are **locked** — we build to this spec.

---

## Decision 1: Spike-Routed Mixture of Experts (SR-MoE)

**Krishna's insight:** *"We can make a hybrid of matrix spike map... master spikes fire and point to inner matrices"*

### What This Becomes

This is a **Mixture of Experts (MoE) with spiking router** — a novel architecture where:

```
Input patch
  → Spiking Router (lightweight classifier)
       Fires: "This is an EDGE patch" or "This is a TEXTURE patch" or "This is FLAT"
  → Routes to specialized Expert matrices:
       Expert 0: Edge sharpening (high inhibitory weights)
       Expert 1: Texture hallucination (high excitatory density)  
       Expert 2: Flat/gradient (minimal weights, smooth output)
       Expert 3: Fine detail (high precision, uses FP16 head)
```

**Why this is better than standard MoE:**
- Standard MoE uses a learned softmax router (dense computation)
- Our spiking router uses **voltage thresholding** — a pixel patch either fires a neuron or it doesn't. Binary decision, zero softmax cost
- Each expert is a small ternary matrix → multiplication-free
- Total experts stay small (4-8), but each is specialized

### Engineering Decision
- **Router:** 1×1 FPSANConv2d → voltage comparison → top-1 expert selection
- **Experts:** Each is a small `FPSANConv2d` block (depthwise + pointwise)
- **Training:** All experts train simultaneously, router learns via straight-through gradient
- **Inference:** Only the selected expert(s) execute per patch → sparsity → speed

---

## Decision 2: Combinatorial Texture Encoding (Broca for Vision)

**Krishna's insight:** *"Let it learn and label texture states so it can fire 2+ neurons together to produce N patterns"*

### What This Becomes

Adapted from V7 Broca's chord system. Instead of one neuron = one texture, we use **combinatorial firing patterns**:

```
4 texture neurons can encode 2^4 - 1 = 15 unique texture states:
  [1,0,0,0] = "smooth gradient"
  [0,1,0,0] = "sharp edge"  
  [1,1,0,0] = "edge ON gradient" (composite)
  [1,0,1,0] = "gradient WITH fine detail"
  [1,1,1,0] = "complex geometry"
  ...
```

**Why this is powerful:**
- 8 neurons can represent **255 texture states** without growing the network
- Each state activates a different expert combination from Decision 1
- The network learns its own "visual language" — exactly like Broca's vocal chords

### Engineering Decision
- **Texture State Layer:** 8 neurons after the initial feature extraction
- **Firing rule:** Each neuron fires if its membrane potential exceeds its dynamic threshold
- **Combination → Expert routing:** The binary chord maps to expert(s) via learned lookup
- **Training:** Texture states emerge naturally from gradient flow, no manual labeling

---

## Decision 3: Layered Spike Precision (Hybrid Ternary + FP16)

**Krishna's insight:** *"Precision can be stored in stages with layered spikes topology... creating a language of its own"* + *"We can get help from FP16/32 since we are so small and super-optimized"*

### What This Becomes

**Hierarchical precision architecture:**

```
Layer 1-3: Ternary FPSANConv2d (feature extraction)
  → Fast, multiplication-free, {-1, 0, +1} weights
  → Captures STRUCTURE: edges, shapes, texture boundaries
  
Layer 4: FP16 Precision Head (color reconstruction)  
  → Standard FP16 conv for fine gradient reproduction
  → Captures COLOR: smooth gradients, subtle hue shifts
  
Layer 5: Residual Spike Refinement
  → Small ternary network that computes a correction
  → Adds micro-detail (hair strands, text sharpness)
```

**Why this is the right trade-off:**
- Ternary backbone: 85% of computation, multiplication-free
- FP16 color head: 10% of computation, handles the precision-critical final output
- Ternary refinement: 5% of computation, sharpens the result
- Total model stays under **0.5 MB** (ternary packed + small FP16 layer)
- GTX 1650 has enough headroom for one FP16 layer (it supports FP16 natively)

### Engineering Decision
- **PixelShuffle reconstruction:** FP16 weights (the color-critical layer)
- **Everything else:** Ternary FPSANConv2d with Astrocytic Stiffness
- **Refinement network:** Optional, activated when GPU budget allows (<1ms remaining)

---

## Decision 4: Dual Homeostatic Control

**Krishna's insight:** *"Let homeostasis control both learning AND inference — dynamic — so there is maximum error reduction"*

### What This Becomes

Two homeostasis systems running simultaneously:

```
1. LEARNING HOMEOSTASIS (controls adaptation speed)
   Expected_gradient_energy ← EMA(actual_gradients)
   If gradients spike (new game content) → increase learning rate
   If gradients flatline (familiar content) → decrease learning rate
   
2. INFERENCE HOMEOSTASIS (controls feature sensitivity)
   Expected_frame_energy ← EMA(frame_luminance)
   If dark scene → increase feature amplification (your cluster_scalar)
   If bright scene → dampen to prevent saturation
   Both modulate the endocrine scalar per-frame
```

### Engineering Decision
- Learning homeostasis: Adjusts `lr` dynamically, bounded `[lr_min, lr_max]`
- Inference homeostasis: Adjusts `cluster_scalar` per-frame, bounded `[0.3, 1.5]` (your original V6 bounds)
- Both use exponential moving average with α=0.1 (slow adaptation)

---

## Decision 5: Multi-Strategy Occlusion Detection

**Krishna's insight:** *"Use A, B, C where they are best according to situation"*

### Engineering Decision

All three run; results are fused:
- **Forward-backward consistency check** → hard occlusion mask (binary)
- **MYELIN-Flow confidence output** → soft reliability map (0-1)
- **Combinatorial texture state change detection** (Decision 2) → "this patch's texture state changed dramatically → likely occluded"
- **Final mask:** `occlusion = max(consistency_check, 1 - confidence) × texture_change_flag`

---

## Decision 6: Adaptive Neurogenesis (Quality-First)

**Krishna's insight:** *"See which gives best quality and performance without hurting specs"*

### Engineering Decision
- Neurogenesis happens **only during offline game adaptation** (not during real-time gameplay)
- During adaptation: if a texture patch consistently produces high error, spawn a new expert
- Max experts capped at 16 (GPU memory/dispatch budget)
- Pruning: if an expert fires less than 1% of the time after adaptation, merge it back

---

## Decision 7: Omnipresent Sleep + Idle Learning

**Krishna's insight:** *"A, B, C can all be there... idle time must also be considered"*

### Engineering Decision
- **Between sessions (A):** Full consolidation. Strongest stiffness update.
- **Loading screens (B):** Opportunistic mini-consolidation if loading > 3 seconds
- **Background thread (C):** Ultra-lightweight consolidation during gameplay in shadow brain (your atomic double-buffer)
- **System idle:** If MYELIN-SR detects GPU idle (game paused, alt-tabbed), run a deeper consolidation pass

---

## Decision 8: Per-Game Auto-Generated Models

**Krishna's insight:** *"Ship with every game, give each game its own best model created automatically"*

### Engineering Decision

```
Distribution model:
1. Universal base model ships (0.5 MB, trained on diverse images)
2. First launch: game runs normally while MYELIN-SR observes frames
3. Over 5-10 minutes of gameplay: fine-tunes via DFA + Astrocytic Stiffness
4. Saves game-specific profile (~0.1 MB delta from base)
5. Next launch: loads base + profile instantly
6. Community profiles: players can share their trained profiles
```

**This is genuinely novel.** DLSS ships one model per game that NVIDIA trains. We train on the user's machine, specialized to their settings, resolution, and even play style.

---

## Decision 9: Broca-Based Frame Prediction (Phase 5)

**Krishna's insight:** *"Let it dream, decide, scratch the frames before rendering"*

### Concept (Future Phase)

```
Broca Visual Predictor:
  Frame N → Feature extraction → "Scene state" spike chord
  Recurrent connection: chord N → predicts chord N+1
  Chord N+1 → Expert routing → Generate predicted frame N+1
  When real frame N+1 arrives: compare, learn from error
  "Dreaming": during idle, replay past sequences and predict ahead
```

This is essentially **learned frame interpolation/extrapolation** using the same spiking recurrence as Broca's language model but applied to visual features instead of words.

**Deferred to Phase 5** after temporal SR is proven.

---

## Decision 10: Confirmed Hybrid Precision

**Final architecture precision map:**

| Component | Precision | Justification |
|---|---|---|
| Feature extraction (layers 1-3) | Ternary {-1,0,+1} | Speed, sparsity, multiplication-free |
| Spiking router | Ternary | Binary decision, no precision needed |
| Expert sub-networks | Ternary | Core computation |
| Channel attention FC layers | FP16 | Small layers, need smooth scaling |
| PixelShuffle reconstruction | FP16 | Color precision critical |
| Residual refinement | Ternary | Sharpening only needs direction, not magnitude |
| Homeostasis scalars | FP32 | Running averages need full precision |

---

## My Recommendations (Building On Your Answers)

### On Decision 1 (SR-MoE):
> [!TIP]
> Start with **4 experts** in Phase 1 (edge / texture / flat / detail). We can grow to 8-16 during per-game adaptation. Keep the router extremely lightweight — it should cost less than 5% of total compute.

### On Decision 3 (Hybrid Precision):
> [!TIP]
> The **residual spike refinement** (Layer 5) is brilliant but should be optional. Make it a quality slider: "Performance mode" = skip refinement, "Quality mode" = include it. This gives users DLSS-style quality presets for free.

### On Decision 8 (Per-Game Models):
> [!TIP]
> The community profile sharing is a killer feature. Imagine: "Top-rated MYELIN-SR profile for Cyberpunk 2077 by user X, trained on 2 hours of gameplay." This creates a crowd-sourced quality network that scales without us doing anything.

### On Decision 9 (Broca Frame Prediction):
> [!IMPORTANT]
> This is the moonshot feature that could genuinely surpass DLSS 4.5. DLSS uses a transformer for frame generation; we'd use a spiking recurrent predictor. The key research question: can a lightweight spiking predictor match a transformer's visual prediction quality? If yes, this is the paper.

---

## Architecture Summary Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    MYELIN-SR v2                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input (720p) ──→ [Ternary Feature Backbone]            │
│                      │                                  │
│                      ▼                                  │
│              [Texture State Encoder]                    │
│              (8 neurons, 255 states)                    │
│                      │                                  │
│                      ▼                                  │
│              [Spiking Router]                           │
│              ┌───┬───┬───┬───┐                          │
│              │E0 │E1 │E2 │E3 │  ← Experts (ternary)    │
│              └─┬─┴─┬─┴─┬─┴─┬─┘                         │
│                └───┴───┴───┘                            │
│                      │                                  │
│                      ▼                                  │
│         [FP16 PixelShuffle Reconstruction]              │
│                      │                                  │
│                      ▼                                  │
│         [Ternary Residual Refinement]  (optional)       │
│                      │                                  │
│                      ▼                                  │
│              Output (1440p)                             │
│                                                         │
│  ┌──────────────────────────────┐                       │
│  │ HOMEOSTASIS (dual)           │                       │
│  │  • Learning rate adaptation  │                       │
│  │  • Inference sensitivity     │                       │
│  └──────────────────────────────┘                       │
│                                                         │
│  ┌──────────────────────────────┐                       │
│  │ ASTROCYTIC STIFFNESS         │                       │
│  │  • Per-weight consolidation  │                       │
│  │  • Sleep scheduling (A/B/C)  │                       │
│  └──────────────────────────────┘                       │
│                                                         │
│  ┌──────────────────────────────┐                       │
│  │ PER-GAME ADAPTATION          │                       │
│  │  • Auto-profile generation   │                       │
│  │  • Community sharing         │                       │
│  │  • Neurogenesis (offline)    │                       │
│  └──────────────────────────────┘                       │
└─────────────────────────────────────────────────────────┘
```
