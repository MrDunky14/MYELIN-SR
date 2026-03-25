#ifndef NSS_CORE_HPP
#define NSS_CORE_HPP

#include <vector>
#include <cstdint>
#include <iostream>

using namespace std;

// The dimensions of our sliding upscaling window.
const int LOW_RES_WINDOW = 9; // 3x3 pixels mapped for feature extraction

// Defines an RGB color as continuous floats (0.0 to 1.0) for interpolation
struct RGBFloat {
    float r, g, b;
};

class NSSCortex {
public:
    NSSCortex();
    
    // Dynamic single-pixel neural extraction (Gather Model)
    RGBFloat execute_single_pixel(const vector<RGBFloat>& low_res_input, float target_x_offset, float target_y_offset);

    // Processes a full image dynamically at an exact mathematical float scale multiplier
    void execute_full_frame_upscale(const char* input_path, const char* output_path, float scale_factor);
    
    // TEMPORAL EXECUTION: Feeds entire directories of frames sequentially into a single Long-Term Memory state.
    void execute_temporal_sequence(const vector<string>& input_frames, const string& output_dir, float scale_factor);
};

#endif // NSS_CORE_HPP
