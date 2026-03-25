<div align="center">
  <h1>🧠 MYELIN-SR v2</h1>
  <p><b>Multiplication-Free AI Super Resolution Engine for DirectX 12</b></p>
  <img src="https://img.shields.io/badge/API-DirectX_12-blue.svg">
  <img src="https://img.shields.io/badge/Language-HLSL_5.1%2B-blueviolet.svg">
  <img src="https://img.shields.io/badge/Platform-Windows_11-lightgrey.svg">
  <img src="https://img.shields.io/badge/License-MIT-green.svg">
</div>

<br/>

MYELIN-SR is a real-time neural upscaler designed for entry-level gaming GPUs (e.g., GTX 1650). It achieves AI-driven upscaling without relying on tensor cores (DLSS) or matrix math units (XeSS). 

Instead, it pioneers a **multiplication-free ternary architecture** (`weights = {-1, 0, 1}`). By dropping all heavy floating-point multipliers, the network evaluates entirely via ultra-fast integer addition and subtraction, executing seamlessly on any standard GPU ALU via HLSL compute shaders.

---

## ⚡ Architecture & Optimizations

*   **Multiplication-Free Ternary Convolution:** Neural weights are strictly quantized to `+1, 0, -1`. The HLSL shader bypasses `fma` instructions entirely, running purely on ADD/SUB/CSEL operations.
*   **Extreme VRAM Packing (2-bit Weights):** Ternary weights are packed 16-to-1 inside standard `uint32` registers at export time. Total weight size drops from 8 MB to **514 KB** (a 93% bandwidth reduction).
*   **Zero-Barrier Dispatch Pipelining:** Adopts a "PS5-style" dispatch approach. The C++ engine pre-builds 55 configurations upfront and submits the entire neural pass in one go to the `D3D12` Compute Queue with **zero mid-frame resource barriers**. The hardware scheduler pipelines dependencies instantly.
*   **SM 6.2 FP16 (`min16float`) Intermediates:** Halves memory bandwidth and doubles register occupancy during inference execution natively on modern hardware via DXC offline pre-compilation (`.cso`).
*   **Async Compute Ready:** Weights and biases fit inside incredibly small 11 KB buffers, making the dispatch highly viable for overlapping on Async Compute queues alongside standard graphics workloads.

---

## 📊 Benchmarks (NVIDIA GeForce GTX 1650)

MYELIN-SR supports two unified binary exports: **Quality** (37-Layers, 48-channels) and **Performance** (29-Layers, 32-channels).

| Mode | Resolution Scaling | Frame Time | Equivalent FPS |
| :--- | :--- | :--- | :--- |
| **Performance** | 540p ➔ 1080p | **11.3 ms** | **88 FPS** | 
| **Performance** | 720p ➔ 1440p | **20.1 ms** | **49 FPS** | 
| **Quality** | 540p ➔ 1080p | **14.4 ms** | **69 FPS** | 
| **Quality** | 720p ➔ 1440p | **25.6 ms** | **38 FPS** | 

*Note: Benchmarks reflect raw compute execution time over 20 warmup iterations with zero inter-layer UAV barriers.*

### Versus The Rivals
*   **vs DLSS:** Brings deep learning super resolution to hardware without Tensor Cores (GTX 10/16 series, AMD RX 500/5000 series).
*   **vs FSR:** Replaces hand-tuned spatial heuristics with a genuine data-driven neural network.
*   **vs XeSS (DP4a):** Outperforms FP32/INT8 precision matrix multiplication limits using 2-bit quantization, unlocking 80+ FPS AI upscaling natively on older hardware ALU cores.

---

## 🚀 Quick Start & Integration

### 1. Build the Engine
Requires **CMake 3.20+** and the **Windows 10/11 SDK** (for D3D12 & DXC).

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

### 1.1 Build the ReShade Addon (No SDK Preinstalled)
You can build `myelin_sr.addon64` even if you do not have the ReShade SDK locally.

```bash
mkdir build && cd build
cmake .. -DBUILD_RESHADE_ADDON=ON -DRESHADE_FETCH_SDK=ON
cmake --build . --config Release --target myelin_sr_addon
```

If you already downloaded ReShade SDK manually, point CMake to it:

```bash
cmake .. -DBUILD_RESHADE_ADDON=ON -DRESHADE_FETCH_SDK=OFF -DRESHADE_SDK_DIR=C:/path/to/reshade
```

Output DLL location:
`build/Release/myelin_sr.addon64`

### 2. Standalone Benchmark Sandbox
Run the compiled `dx12_benchmark.exe` to trigger a simulated 20-frame headless run. 
Specify the preset model:

```bash
# Maximize graphical fidelity
dx12_benchmark.exe quality

# Maximize framerate
dx12_benchmark.exe performance
```

### 3. How to Use in Games (End-User Guide)
MYELIN-SR can be injected into almost any DirectX 12 game using our custom **ReShade Addon**. 

**Disclaimer:** Use in single-player games only. Online games with anti-cheat (EAC, BattlEye) may flag ReShade addons.

**Installation Steps:**
1. Download and install [ReShade](https://reshade.me/) for your target game. **IMPORTANT:** You must check the box during installation that enables **Addons**.
2. Download the latest `myelin_sr.addon64` and the `outputs/` folder from the Releases page.
3. Open your game's installation directory (where the game `.exe` lives).
4. Drop `myelin_sr.addon64` into the same folder as the `.exe`.
5. Drop `myelin_engine_v2_quality.bin` and `myelin_compute.cso` into the exact same folder.
6. Launch the game!
7. Lower your in-game rendering resolution (e.g., set the game to 1280x720 while your monitor is at 2560x1440).
8. Press the `Home` key to open the ReShade overlay. Navigate to the **Addons** tab to verify MYELIN-SR is active and upscaling your frame just before it hits your monitor!

---

## 🛠️ Exporting the PyTorch Model

Train your Spatial and Temporal models via PyTorch (see the `train/` directory), then pack them via the custom exporter.

```bash
python tools/export_hlsl_weights.py
```

This merges the FP32 PyTorch checkpoints into dual `.bin` files (`myelin_engine_v2_quality.bin` and `myelin_engine_v2_performance.bin`) packed precisely for HLSL structure buffers.
