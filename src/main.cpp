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
#include <chrono>
#include <string>
#include <filesystem>
#include <algorithm>

using namespace std;
using namespace std::chrono;
namespace fs = std::filesystem;

bool compare_nat(const string& a, const string& b) {
    if (a.find("imgi_") == string::npos || b.find("imgi_") == string::npos) return a < b;
    
    size_t start_a = a.find("imgi_") + 5;
    size_t end_a = a.find('_', start_a);
    int num_a = (end_a != string::npos) ? stoi(a.substr(start_a, end_a - start_a)) : 0;

    size_t start_b = b.find("imgi_") + 5;
    size_t end_b = b.find('_', start_b);
    int num_b = (end_b != string::npos) ? stoi(b.substr(start_b, end_b - start_b)) : 0;

    return num_a < num_b;
}

int main(int argc, char** argv) {
    cout << "===========================================================" << endl;
    cout << " MYELIN-SR | Phase 8 Temporal Accumulation Offline Test" << endl;
    cout << "===========================================================\n" << endl;

    if (argc < 2) {
        cout << "Usage: nss_engine.exe [input_dir_or_file] [output_dir_or_file] [scale_factor]" << endl;
        return 1;
    }

    string input_path = argv[1];
    string output_path = (argc > 2) ? argv[2] : "data/output_scaled.png";
    float scale_factor = (argc > 3) ? stof(argv[3]) : 2.0f; 

    NSSCortex core;
    auto start = chrono::high_resolution_clock::now();

    if (fs::is_directory(input_path)) {
        cout << "\n--- TEMPORAL SEQUENCE MODE (TAAU) ---\n";
        cout << "Accumulator Target: " << scale_factor << "x Upscale\n";
        
        fs::create_directories(output_path); // Ensure output bin exists
        
        vector<string> sequence;
        for (const auto& entry : fs::directory_iterator(input_path)) {
            if (entry.path().extension() == ".png" || entry.path().extension() == ".jpg") {
                sequence.push_back(entry.path().string());
            }
        }
        
        // Natural topological sorting (imgi_2 comes before imgi_10)
        sort(sequence.begin(), sequence.end(), compare_nat);
        
        cout << "Loaded " << sequence.size() << " Sequential Frames. Engaging Long-Term Memory Arrays...\n";
        
        try {
            core.execute_temporal_sequence(sequence, output_path, scale_factor);
        } catch(const exception& e) {
            cerr << "Temporal Execution Fatal: " << e.what() << endl;
            return 1;
        }
    } else {
        cout << "\n--- SINGLE FRAME GATHER (STATIC) ---\n";
        try {
            core.execute_full_frame_upscale(input_path.c_str(), output_path.c_str(), scale_factor);
        } catch(const exception& e) {
            cerr << "Static Execution Fatal: " << e.what() << endl;
            return 1;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << "\nTotal Offline Neural Execution Time: " << duration.count() << " ms\n" << endl;
    return 0;
}
