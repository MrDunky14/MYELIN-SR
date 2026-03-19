# FP-SAN Neuromorphic Super Sampling (NSS)
## Master Architecture Manifesto

### The Motivation: Justice for Gamers
Standard Super Resolution and Frame Generation (NVIDIA DLSS, AMD FSR, Intel XeSS) rely on monopolized hardware architectures (Tensor Cores) and dense mathematical arrays (FP16/FP32). They actively gatekeep high-performance gaming behind premium silicon. 

FP-SAN (Forward-Propagating Sparse Agentic Network) shatters this monopoly. By using Biological Physics instead of Brute-Force Math, NSS scales to ancient, low-end hardware natively.

---

### Part 1: Biological Rendering Solutions

#### 1. The Output Problem: Classification vs. Regression
*   **The Problem:** Neural networks natively output discrete values (1 or 0 - e.g., "Is this a dog?"). Graphics require continuous RGB colors.
*   **The FP-SAN Solution:** The **Membrane Voltage Integrator**. Instead of boolean spikes, NSS outputs the continuous biological membrane potential ($V_m$). This enables precise, infinite RGB precision generated purely from binary input events.

#### 2. The Aliasing Problem: Sub-Pixel Jitter
*   **The Problem:** Upscalers need temporal history to construct fine detail (like hair or thin wires) without shimmering noise. 
*   **The FP-SAN Solution:** The **VRAM Memory-Spike Map**. Because the FP-SAN model footprint is so small, we allocate the saved VRAM to a spatial/temporal reservoir. The architecture accumulates sub-threshold voltage directly into the map, natively tracking sub-pixel history.

#### 3. The Ghosting Problem: Transparent Occlusion
*   **The Problem:** DLSS relies entirely on Game-Engine Motion Vectors to track pixels. Motion Vectors fail during transparent or particle effects, creating massive smearing/ghosting.
*   **The FP-SAN Solution:** The **Predictive Lateral Synapse**. Instead of reading vector data, when an edge moves, it propagates current laterally to prime its neighbor. The SNN simulates physical momentum entirely on its own.

#### 4. The Disocclusion Problem: Blind Spots
*   **The Problem:** When an object moves, the pixels hidden behind it must be revealed. DLSS has no history of these pixels and blurs them.
*   **The FP-SAN Solution:** **Depth-Layered Parallax Neurogenesis**. The clusters are mapped into Low/Mid/High structural depth layers. Placed in conjunction with **Background Pattern Learning**, the system caches asset data to instantly hallucinate missing architectural patterns.

---

### Part 2: Extreme Hardware Optimizations

#### 1. The PCI-e Bus Chokehold
Copying a frame from the GPU VRAM to the CPU RAM, calculating it, and sending it back limits hardware to 2-3 FPS. 
**Solution:** The FP-SAN math logic is mapped precisely into an **HLSL Compute Shader**. It executes across generic GPU cores physically adjacent to the Render Pass buffer (Zero-Copy Latency).

#### 2. GPU Warp Divergence (The Hardware Paradox)
Executing a Sparse Network (only calculating active pixels) normally causes a GPU block to stall, as threads wait for inactive threads to finish.
**Solution:** The contiguous data structure and **Exponential Homeostasis** mathematically compress the active neurons. Threads operate aggressively on purely active events without dead branching, bypassing Warp Divergence explicitly.

#### 3. 1.58-Bit Ternary Bitwise Compression (The Bandwidth Killer)
Standard networks read 512+ bits for a simple 4x4 matrix. 
**Solution:** FP-SAN uses 1.58-bit ternary math `[-1, 0, 1]`. The architecture packs 16 individual weights into a single 32-bit `uint32_t` integer. The GPU shader unpacks these utilizing bit-masking in hardware circuitry:
```hlsl
uint extractedBits = (packedWeights >> shift) & 3; 
```
This brutally compresses VRAM payload bandwidth to **6%** of standard artificial networks.
