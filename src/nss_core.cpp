#include "nss_core.hpp"
#include <cmath>
#include <algorithm>
#include <iomanip>

NSSCortex::NSSCortex() {
    expected_ambient_energy = 5.0f;
    // Initialize clusters corresponding to the High-Res output pixels (2x2 = 4)
    clusters.resize(HIGH_RES_WINDOW);
    for (int c = 0; c < HIGH_RES_WINDOW; c++) {
        // The ternary array is no longer static. It natively generates itself 
        // during runtime based completely on the Biological Voltage contrast!
        clusters[c].packed_ternary_payload = 0; 

        clusters[c].cluster_scalar = 1.0f;
        clusters[c].membrane_potential = 0.0f;
        clusters[c].temporal_accumulation = 0.0f;
        clusters[c].base_leak_rate = 0.1f;
        clusters[c].base_spike_threshold = 0.8f;
        clusters[c].last_spike_time = 0;
    }
}

void NSSCortex::calculate_homeostasis(const vector<RGBFloat>& low_res_input, float& homeostasis_ratio) {
    float current_frame_energy = 0.0f;
    
    // We proxy "Luminosity" by calculating the mean RGB intensity
    for (const auto& px : low_res_input) {
        float luminance = (0.299f * px.r) + (0.587f * px.g) + (0.114f * px.b);
        if (luminance > 0.1f) current_frame_energy += 1.0f; // Active pixel event
    }

    expected_ambient_energy = (0.8f * expected_ambient_energy) + (0.2f * current_frame_energy);
    
    homeostasis_ratio = 1.0f;
    if (expected_ambient_energy > 0.1f) {
        homeostasis_ratio = current_frame_energy / expected_ambient_energy;
        homeostasis_ratio = pow(homeostasis_ratio, 1.5f);
    }

    // Bound the pupil dilation
    homeostasis_ratio = max(0.3f, min(1.5f, homeostasis_ratio));
}

vector<RGBFloat> NSSCortex::execute_upscale_cycle(const vector<RGBFloat>& low_res_input) {
    if (low_res_input.size() != LOW_RES_WINDOW) {
        throw invalid_argument("Input patch size must match LOW_RES_WINDOW.");
    }
    
    vector<RGBFloat> high_res_output(HIGH_RES_WINDOW);

    // DYNAMIC BIOLOGICAL STIMULUS (Extracting true membrane voltage contrast)
    float cluster_avg_voltage = 0.0f;
    for (int i = 0; i < LOW_RES_WINDOW; i++) {
        cluster_avg_voltage += (low_res_input[i].r + low_res_input[i].g + low_res_input[i].b) / 3.0f;
    }
    cluster_avg_voltage /= LOW_RES_WINDOW;
    
    // Observer Manager (Dynamic Local Variance tracking to auto-tune scalars)
    float local_variance = 0.0f;
    for (int i = 0; i < LOW_RES_WINDOW; i++) {
        float v = (low_res_input[i].r + low_res_input[i].g + low_res_input[i].b) / 3.0f;
        local_variance += abs(v - cluster_avg_voltage);
    }
    local_variance /= LOW_RES_WINDOW; 

    // Simulation of endocrine ambient lighting checks
    float homeostasis_ratio = 1.0f; 

    for (int c = 0; c < HIGH_RES_WINDOW; c++) {
        // --- DYNAMIC 1.58-BIT VOLTAGE ADAPTATION ---
        uint32_t dynamic_packed_payload = 0;
        int out_x = c % 2;
        int out_y = c / 2;
        float target_x = out_x + 0.5f; 
        float target_y = out_y + 0.5f;

        for (int i = 0; i < LOW_RES_WINDOW; i++) {
            int in_x = i % 3;
            int in_y = i / 3;
            float dist_sq = (target_x - in_x)*(target_x - in_x) + (target_y - in_y)*(target_y - in_y);
            
            float pixel_val = (low_res_input[i].r + low_res_input[i].g + low_res_input[i].b) / 3.0f;
            float voltage_diff = pixel_val - cluster_avg_voltage;
            
            int weight = 0;
            // The Frame Literally Wires Itself: Synapses adapt based on exact structural contrast arrays!
            if (dist_sq < 0.6f) weight = 1; // Direct Spatial Binding (Preserve structural map)
            else if (voltage_diff > 0.03f && dist_sq < 1.1f) weight = 1; // Natively identifies Highlight Edge Expansion
            else if (voltage_diff < -0.03f && dist_sq > 0.5f) weight = -1; // Natively identifies Shadow/Outline Inhibition
            else weight = 0; 
            
            uint32_t bit_val = (weight == 0) ? 1 : ((weight == 1) ? 2 : 3);
            dynamic_packed_payload |= (bit_val << (i * 2));
        }

        // The Engine extracts the dynamically generated bitwise tensor directly
        clusters[c].packed_ternary_payload = dynamic_packed_payload; 
        
        // Observer Manager auto-escalates multipliers on complex textures (Cinematic Gradients)
        clusters[c].cluster_scalar = 1.0f + (local_variance * 4.0f);

        float dynamic_threshold = clusters[c].base_spike_threshold * homeostasis_ratio;
        float dynamic_leak = clusters[c].base_leak_rate * homeostasis_ratio;
        // 2. Continuous State Extraction (The Decoder)
        int active_synapses = 0;
        int inhibitory_synapses = 0; // NEW: Track negative weights to build the Normalizer
        float raw_ternary_sum_r = 0.0f, raw_ternary_sum_g = 0.0f, raw_ternary_sum_b = 0.0f;

        for (int i = 0; i < LOW_RES_WINDOW; i++) {
            // Event threshold (sparsity check)
            float pixel_intensity = (low_res_input[i].r + low_res_input[i].g + low_res_input[i].b) / 3.0f;
            
            if (pixel_intensity > 0.05f) {
                // Decay the temporal memory (resolving ghosting)
                clusters[c].membrane_potential *= (1.0f - dynamic_leak);

                // --- 1.58-BIT BITWISE UNPACKING ---
                uint32_t shift = i * 2;
                uint32_t extracted_bits = (clusters[c].packed_ternary_payload >> shift) & 3;

                // 1.58-Bit Ternary Math Execution (Rapid int + float multiplexing)
                if (extracted_bits == 2) { // Weight is +1 (Excitatory)
                    raw_ternary_sum_r += low_res_input[i].r;
                    raw_ternary_sum_g += low_res_input[i].g;
                    raw_ternary_sum_b += low_res_input[i].b;
                    active_synapses++; 
                } else if (extracted_bits == 3) { // Weight is -1 (Inhibitory Edge)
                    raw_ternary_sum_r -= low_res_input[i].r * 0.25f; // Gentle sharpening
                    raw_ternary_sum_g -= low_res_input[i].g * 0.25f;
                    raw_ternary_sum_b -= low_res_input[i].b * 0.25f;
                    inhibitory_synapses++; // Track for Normalizer
                }
            }
        }

        if (active_synapses > 0) {
            // Aggregate membrane voltage dynamically for texture generation
            float avg_intensity = (raw_ternary_sum_r + raw_ternary_sum_g + raw_ternary_sum_b) / 3.0f;
            clusters[c].membrane_potential += avg_intensity;
            
            // Distribute memory structurally into VRAM jitter map
            clusters[c].temporal_accumulation += (avg_intensity * clusters[c].cluster_scalar) / active_synapses;
            
            // PHYSICAL NORMALIZER: 
            // Calculate the true mathematical denominator. We must normalize the sum of active weights.
            // If there are 5 Excitatory (+1) and 4 Inhibitory (-0.25) connections:
            // Net structural multiplier = 5 - (4 * 0.25) = 4.0f
            float biological_divisor = (float)active_synapses - (inhibitory_synapses * 0.25f);
            
            // Fail-safe to prevent zero-division or blackout on extreme contrast patches
            if (biological_divisor < 0.1f) biological_divisor = max(1.0f, (float)active_synapses);

            // Reconstruct continuous RGB using the hybrid scalar and physical biological divisor
            high_res_output[c].r = max(0.0f, min(1.0f, (raw_ternary_sum_r * clusters[c].cluster_scalar) / biological_divisor));
            high_res_output[c].g = max(0.0f, min(1.0f, (raw_ternary_sum_g * clusters[c].cluster_scalar) / biological_divisor));
            high_res_output[c].b = max(0.0f, min(1.0f, (raw_ternary_sum_b * clusters[c].cluster_scalar) / biological_divisor));
        } else {
            // Background is entirely skipping execution (Zero Compute Cost)
            high_res_output[c] = {0.0f, 0.0f, 0.0f};
        }
    }

    return high_res_output;
}

vector<RGBFloat> NSSCortex::execute_full_frame_upscale(const vector<RGBFloat>& low_res_image, int in_w, int in_h, int& out_w, int& out_h) {
    // 2x Temporal/Spatial Upscaling
    out_w = in_w * 2;
    out_h = in_h * 2;
    vector<RGBFloat> upscaled_image(out_w * out_h, {0.0f, 0.0f, 0.0f});

    // In a full implementation, we allocate the Endocrine ratio globally.
    float homeostasis_ratio;
    // Fast mock-homeostasis sampling the center of the image
    vector<RGBFloat> sample = {low_res_image[(in_h/2)*in_w + (in_w/2)]};
    calculate_homeostasis(sample, homeostasis_ratio);

    // Sliding Window 2x2 iteration mapping to 4x4 output
    for (int y = 0; y < in_h - 1; y++) {
        for (int x = 0; x < in_w - 1; x++) {
            
            // Extract the 3x3 low-res localized patch (handling image boundaries)
            vector<RGBFloat> patch(9);
            for(int dy = -1; dy <= 1; dy++) {
                for(int dx = -1; dx <= 1; dx++) {
                    int px = max(0, min(in_w - 1, x + dx));
                    int py = max(0, min(in_h - 1, y + dy));
                    patch[(dy+1)*3 + (dx+1)] = low_res_image[py * in_w + px];
                }
            }

            // Execute Spike-to-RGB Decoder on the patch
            vector<RGBFloat> high_res_patch = execute_upscale_cycle(patch);

            // Write back to the exact 2x2 upscaled coordinate space
            for(int hy=0; hy<2; hy++) {
                for(int hx=0; hx<2; hx++) {
                    int out_idx = ((y * 2) + hy) * out_w + ((x * 2) + hx);
                    // Direct 1:1 overwrite via sliding window (no distortion)
                    upscaled_image[out_idx] = high_res_patch[hy * 2 + hx];
                }
            }
        }
    }
    return upscaled_image;
}
