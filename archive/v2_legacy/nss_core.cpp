// MYELIN-SR v2: Zero-Barrier Ternary Reconstruction Engine
// Copyright (C) 2026 Krishna Singh
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

#include "../include/nss_core.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <stdexcept>
#include <string>

// Correct Include Paths
#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

template <typename T>
T clamp_local(T val, T min_val, T max_val) {
    return std::max(min_val, std::min(val, max_val));
}

NSSCortex::NSSCortex() {
    // Initialization phase complete. Dynamic weights handle real-time spatial wiring.
}

RGBFloat NSSCortex::execute_single_pixel(const vector<RGBFloat>& low_res_input, float target_x_offset, float target_y_offset) {
    if (low_res_input.size() != LOW_RES_WINDOW) {
        throw invalid_argument("Input patch size must match LOW_RES_WINDOW.");
    }

    // DYNAMIC BIOLOGICAL STIMULUS
    float cluster_avg_voltage = 0.0f;
    for (int i = 0; i < LOW_RES_WINDOW; i++) {
        cluster_avg_voltage += (low_res_input[i].r + low_res_input[i].g + low_res_input[i].b) / 3.0f;
    }
    cluster_avg_voltage /= LOW_RES_WINDOW;
    
    float local_variance = 0.0f;
    for (int i = 0; i < LOW_RES_WINDOW; i++) {
        float v = (low_res_input[i].r + low_res_input[i].g + low_res_input[i].b) / 3.0f;
        local_variance += abs(v - cluster_avg_voltage);
    }
    local_variance /= LOW_RES_WINDOW; 

    // Target geometric center mapped relative to the 3x3 array (0.0 to 2.0 space)
    float target_x = target_x_offset;
    float target_y = target_y_offset;

    int active_synapses = 0;
    int inhibitory_synapses = 0;
    float raw_ternary_sum_r = 0.0f, raw_ternary_sum_g = 0.0f, raw_ternary_sum_b = 0.0f;

    for (int i = 0; i < LOW_RES_WINDOW; i++) {
        int in_x = i % 3;
        int in_y = i / 3;
        float dist_sq = (target_x - in_x)*(target_x - in_x) + (target_y - in_y)*(target_y - in_y);
        
        float pixel_val = (low_res_input[i].r + low_res_input[i].g + low_res_input[i].b) / 3.0f;
        float voltage_diff = pixel_val - cluster_avg_voltage;
        
        int weight = 0;
        // Native Edge Evaluator Logic (Resolution Agnostic mapping)
        if (dist_sq < 0.6f) {
            weight = 1; 
        } else {
            if (voltage_diff > 0.02f && dist_sq < 1.5f) weight = 1; 
            else if (voltage_diff < -0.02f && dist_sq >= 0.5f) weight = -1; 
            else weight = 0; 
        }
        
        if (weight == 1) { 
            raw_ternary_sum_r += low_res_input[i].r;
            raw_ternary_sum_g += low_res_input[i].g;
            raw_ternary_sum_b += low_res_input[i].b;
            active_synapses++; 
        } else if (weight == -1) { 
            raw_ternary_sum_r -= low_res_input[i].r * 0.25f;
            raw_ternary_sum_g -= low_res_input[i].g * 0.25f;
            raw_ternary_sum_b -= low_res_input[i].b * 0.25f;
            inhibitory_synapses++; 
        }
    }

    float cluster_scalar = 1.0f + (local_variance * 4.0f);
    float biological_divisor = (float)active_synapses - (inhibitory_synapses * 0.25f);
    if (biological_divisor < 0.2f) biological_divisor = 0.2f;

    RGBFloat pixel;
    pixel.r = max(0.0f, min(1.0f, (raw_ternary_sum_r * cluster_scalar) / biological_divisor));
    pixel.g = max(0.0f, min(1.0f, (raw_ternary_sum_g * cluster_scalar) / biological_divisor));
    pixel.b = max(0.0f, min(1.0f, (raw_ternary_sum_b * cluster_scalar) / biological_divisor));
    return pixel;
}

void NSSCortex::execute_full_frame_upscale(const char* input_path, const char* output_path, float scale_factor) {
    int in_w, in_h, in_channels;
    unsigned char* img_data = stbi_load(input_path, &in_w, &in_h, &in_channels, 3);

    if (!img_data) {
        throw runtime_error("Failed to load map.");
    }

    // GATHER SCALING MATHEMATICS: Infinitely arbitrary float scaling!
    int out_w = (int)((float)in_w * scale_factor);
    int out_h = (int)((float)in_h * scale_factor);
    
    vector<RGBFloat> upscaled_image(out_w * out_h, {0.0f, 0.0f, 0.0f});
    vector<RGBFloat> patch(LOW_RES_WINDOW);

    // Render loop maps mathematical output back into input space geometry
    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            
            float exact_in_x = (float)x / scale_factor;
            float exact_in_y = (float)y / scale_factor;
            
            int in_x_floor = (int)exact_in_x;
            int in_y_floor = (int)exact_in_y;
            
            float frac_x = exact_in_x - in_x_floor;
            float frac_y = exact_in_y - in_y_floor;

            for (int py = 0; py < 3; py++) {
                for (int px = 0; px < 3; px++) {
                    int sample_x = clamp_local(in_x_floor - 1 + px, 0, in_w - 1);
                    int sample_y = clamp_local(in_y_floor - 1 + py, 0, in_h - 1);
                    
                    int idx = (sample_y * in_w + sample_x) * 3; // Enforced 3-channel bounds
                    patch[py * 3 + px] = {
                        img_data[idx] / 255.0f,
                        img_data[idx+1] / 255.0f,
                        img_data[idx+2] / 255.0f
                    };
                }
            }

            float target_x_offset = frac_x + 0.5f; 
            float target_y_offset = frac_y + 0.5f;

            RGBFloat generated_pixel = execute_single_pixel(patch, target_x_offset, target_y_offset);
            upscaled_image[y * out_w + x] = generated_pixel;
        }
    }

    vector<unsigned char> output_data(out_w * out_h * 3);
    for (int i = 0; i < out_w * out_h; ++i) {
        output_data[i * 3 + 0] = static_cast<unsigned char>(clamp_local(upscaled_image[i].r * 255.0f, 0.0f, 255.0f));
        output_data[i * 3 + 1] = static_cast<unsigned char>(clamp_local(upscaled_image[i].g * 255.0f, 0.0f, 255.0f));
        output_data[i * 3 + 2] = static_cast<unsigned char>(clamp_local(upscaled_image[i].b * 255.0f, 0.0f, 255.0f));
    }

    stbi_write_png(output_path, out_w, out_h, 3, output_data.data(), out_w * 3);
    stbi_image_free(img_data);
}

void NSSCortex::execute_temporal_sequence(const vector<string>& input_frames, const string& output_dir, float scale_factor) {
    if (input_frames.empty()) return;

    int in_w, in_h, in_channels;
    // Probe the first frame for dimensions
    unsigned char* probe = stbi_load(input_frames[0].c_str(), &in_w, &in_h, &in_channels, 3);
    if (!probe) throw runtime_error("Failed to load first sequence frame.");
    stbi_image_free(probe);

    int out_w = (int)((float)in_w * scale_factor);
    int out_h = (int)((float)in_h * scale_factor);

    // LONG TERM MEMORY ARRAY (The Temporal Buffer)
    vector<RGBFloat> temporal_map(out_w * out_h, {0.0f, 0.0f, 0.0f});
    vector<float> temporal_weight(out_w * out_h, 0.0f); // Tracks historical confidence

    for (size_t f = 0; f < input_frames.size(); f++) {
        unsigned char* img_data = stbi_load(input_frames[f].c_str(), &in_w, &in_h, &in_channels, 3);
        if (!img_data) {
            cout << "Warning: Failed to load " << input_frames[f] << endl;
            continue;
        }

        vector<RGBFloat> patch(LOW_RES_WINDOW);
        
        // Loop over the gathered target resolution
        for (int y = 0; y < out_h; y++) {
            for (int x = 0; x < out_w; x++) {
                
                float exact_in_x = (float)x / scale_factor;
                float exact_in_y = (float)y / scale_factor;
                int in_x_floor = (int)exact_in_x;
                int in_y_floor = (int)exact_in_y;
                float frac_x = exact_in_x - in_x_floor;
                float frac_y = exact_in_y - in_y_floor;

                for (int py = 0; py < 3; py++) {
                    for (int px = 0; px < 3; px++) {
                        int sample_x = clamp_local(in_x_floor - 1 + px, 0, in_w - 1);
                        int sample_y = clamp_local(in_y_floor - 1 + py, 0, in_h - 1);
                        int idx = (sample_y * in_w + sample_x) * 3; // Enforced 3-channel bounds
                        patch[py * 3 + px] = { img_data[idx] / 255.0f, img_data[idx+1] / 255.0f, img_data[idx+2] / 255.0f };
                    }
                }

                RGBFloat current_frame_pixel = execute_single_pixel(patch, frac_x + 0.5f, frac_y + 0.5f);
                int out_idx = y * out_w + x;
                RGBFloat history_pixel = temporal_map[out_idx];

                // NEIGHBORHOOD VARIANCE CLAMPING (TAAU Ghosting Eradicator)
                // Since this offline execution lacks Native Engine Motion Vectors (Optical Flow),
                // we calculate the mathematical Bounding Box of the surrounding 3x3 reality structure.
                float min_r = 1.0f, min_g = 1.0f, min_b = 1.0f;
                float max_r = 0.0f, max_g = 0.0f, max_b = 0.0f;
                for (int i=0; i<9; i++) {
                    min_r = min(min_r, patch[i].r); max_r = max(max_r, patch[i].r);
                    min_g = min(min_g, patch[i].g); max_g = max(max_g, patch[i].g);
                    min_b = min(min_b, patch[i].b); max_b = max(max_b, patch[i].b);
                }

                // If the historical ghost trails outside the current geometric physics bounds, 
                // it is immediately clamped securely inside the current geometric boundaries.
                RGBFloat clamped_history;
                clamped_history.r = clamp_local(history_pixel.r, min_r, max_r);
                clamped_history.g = clamp_local(history_pixel.g, min_g, max_g);
                clamped_history.b = clamp_local(history_pixel.b, min_b, max_b);

                // SHORT-TERM ALERTNESS (Disocclusion Evaluator)
                float voltage_divergence = abs(current_frame_pixel.r - history_pixel.r) + 
                                           abs(current_frame_pixel.g - history_pixel.g) + 
                                           abs(current_frame_pixel.b - history_pixel.b);

                // Critical Velocity Reached: Flush temporal memory to baseline to avert extreme smearing
                if (voltage_divergence > 0.2f || temporal_weight[out_idx] == 0.0f) {
                    temporal_weight[out_idx] = 1.0f; 
                } else {
                    // Maximum 8-Frame Exponential Lag: Prevents the buffer from freezing time permanently
                    temporal_weight[out_idx] = min(8.0f, temporal_weight[out_idx] + 1.0f);
                }

                float blend_alpha = 1.0f / temporal_weight[out_idx];
                
                temporal_map[out_idx].r = (current_frame_pixel.r * blend_alpha) + (clamped_history.r * (1.0f - blend_alpha));
                temporal_map[out_idx].g = (current_frame_pixel.g * blend_alpha) + (clamped_history.g * (1.0f - blend_alpha));
                temporal_map[out_idx].b = (current_frame_pixel.b * blend_alpha) + (clamped_history.b * (1.0f - blend_alpha));
            }
        }

        // Render the temporally stabilized frame to disk
        vector<unsigned char> output_data(out_w * out_h * 3);
        for (int i = 0; i < out_w * out_h; ++i) {
            output_data[i * 3 + 0] = static_cast<unsigned char>(clamp_local(temporal_map[i].r * 255.0f, 0.0f, 255.0f));
            output_data[i * 3 + 1] = static_cast<unsigned char>(clamp_local(temporal_map[i].g * 255.0f, 0.0f, 255.0f));
            output_data[i * 3 + 2] = static_cast<unsigned char>(clamp_local(temporal_map[i].b * 255.0f, 0.0f, 255.0f));
        }

        // String formatting explicitly avoiding C++20 std::format for wide MSVC compatibility
        string out_file = output_dir + "/frame_" + to_string(f) + ".png";
        stbi_write_png(out_file.c_str(), out_w, out_h, 3, output_data.data(), out_w * 3);
        stbi_image_free(img_data);
        
        cout << "[Temporal Sequencer] Stitched Frame " << f+1 << "/" << input_frames.size() << " | VRAM Alertness Flushed." << endl;
    }
}
