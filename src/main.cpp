#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include "../include/nss_core.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <string>

using namespace std;
using namespace std::chrono;

int main(int argc, char** argv) {
    cout << "===========================================================" << endl;
    cout << " FP-SAN NSS | Full-Frame Biological Upscaler Test" << endl;
    cout << "===========================================================\n" << endl;

    if (argc < 2) {
        cout << "Usage: nss_engine.exe [path_to_720p_image.png]" << endl;
        cout << "(Testing fallback: generating noisy mock patch for benchmarking...)" << endl;
        // Proceed with a generic memory scale test if no image uploaded
    }

    string input_file = (argc >= 2) ? argv[1] : "data/test_720p.png";
    int in_w, in_h, channels;

    // We can load a real image, or if missing, allocate a dummy 1280x720 memory block for pure speed testing
    unsigned char* img_data = stbi_load(input_file.c_str(), &in_w, &in_h, &channels, 3);
    
    vector<RGBFloat> low_res_image;
    if (img_data) {
        cout << "[IO] Loaded " << input_file << " (" << in_w << "x" << in_h << ")" << endl;
        low_res_image.resize(in_w * in_h);
        for(int i = 0; i < in_w * in_h; i++) {
            low_res_image[i] = {
                img_data[i*3] / 255.0f,
                img_data[i*3+1] / 255.0f,
                img_data[i*3+2] / 255.0f
            };
        }
        stbi_image_free(img_data);
    } else {
        cout << "[IO] Failed to load actual image. Allocating theoretical 1280x720 VRAM state..." << endl;
        in_w = 1280; in_h = 720;
        low_res_image.resize(in_w * in_h, {0.1f, 0.2f, 0.5f}); // Dark blue background
        
        // Inject a high-contrast moving white square
        for(int y = 300; y < 400; y++) {
            for(int x = 600; x < 700; x++) {
                low_res_image[y * in_w + x] = {1.0f, 1.0f, 1.0f};
            }
        }
    }

    NSSCortex upscaler;
    int out_w, out_h;

    cout << "\n--- EXECUTING NATIVE C++ FULL-FRAME DECODING ---" << endl;
    auto start = high_resolution_clock::now();
    
    vector<RGBFloat> high_res_image = upscaler.execute_full_frame_upscale(low_res_image, in_w, in_h, out_w, out_h);
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    cout << "Execution Time (Milliseconds): " << duration.count() << " ms" << endl;
    cout << "Upscaled Resolution: " << out_w << "x" << out_h << " (1440p Target)" << endl;

    // Optional: Write out to disk to visually analyze the Neural Sub-Pixel structure
    vector<unsigned char> out_data(out_w * out_h * 3);
    for(int i=0; i < out_w * out_h; i++) {
        out_data[i*3] = (unsigned char)(max(0.0f, min(1.0f, high_res_image[i].r)) * 255);
        out_data[i*3+1] = (unsigned char)(max(0.0f, min(1.0f, high_res_image[i].g)) * 255);
        out_data[i*3+2] = (unsigned char)(max(0.0f, min(1.0f, high_res_image[i].b)) * 255);
    }

    stbi_write_png("data/output_1440p.png", out_w, out_h, 3, out_data.data(), out_w * 3);
    cout << "[IO] Saved biological rendering output to: data/output_1440p.png" << endl;

    return 0;
}
