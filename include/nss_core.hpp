#ifndef NSS_CORE_HPP
#define NSS_CORE_HPP

#include <vector>
#include <cstdint>
#include <iostream>

using namespace std;

// The dimensions of our sliding upscaling window.
// e.g., mapping a 3x3 low-res patch to a 2x2 high-res patch (which represents a 2x Upscale)
const int LOW_RES_WINDOW = 9; // 3x3 pixels flattened
const int HIGH_RES_WINDOW = 4; // 2x2 upscaled pixels flattened

// Defines an RGB color as continuous floats (0.0 to 1.0) for interpolation
struct RGBFloat {
    float r, g, b;
};

// The Neuromorphic Output Pixel (Membrane Voltage Integrator)
struct NSS_Neuron {
    // 1.58-bit Memory Compression payload (Up to 16 weights compressed into one uint32_t buffer)
    // Minimizes PCIe VRAM bottlenecks by 93% vs FP32 Dense Nets.
    uint32_t packed_ternary_payload;
    
    // Continuous precision scalar
    float cluster_scalar;

    // The biological voltage which will map directly to RGB intensity
    float membrane_potential;
    
    // Homeostasis / Leak control
    float base_leak_rate;
    float base_spike_threshold;
    uint64_t last_spike_time;

    // Temporal Sub-Pixel Jitter Accumulator (The 'memoryspike_map' concept)
    float temporal_accumulation; 
};

// The Spike-to-RGB Decoder
class NSSCortex {
private:
    vector<NSS_Neuron> clusters;
    float expected_ambient_energy;

    // Helper functions
    void calculate_homeostasis(const vector<RGBFloat>& low_res_input, float& homeostasis_ratio);

public:
    NSSCortex();

    // The core Sparse Matrix Execution loop (Patch based)
    vector<RGBFloat> execute_upscale_cycle(const vector<RGBFloat>& low_res_input);
    
    // NEW: Full Frame execution mapping (Real Image Testing)
    vector<RGBFloat> execute_full_frame_upscale(const vector<RGBFloat>& low_res_image, int in_w, int in_h, int& out_w, int& out_h);
    
    // Configures the V2 layer for texture hallucination based on user requests
    void inject_texture_micro_gradients();
};

#endif // NSS_CORE_HPP
