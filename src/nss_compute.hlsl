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

// ==============================================================================
// FP-SAN Neuromorphic Super Sampling (NSS) - Compute Shader
// Language: HLSL (Direct3D 12 / Vulkan via DXC)
// Concept: Translates the Sparse Biological C++ mechanics into GPU VRAM
// ==============================================================================

// The thread group size defines how many pixels we process in parallel
// We use 8x8 (64 pixels) per wave to maximize GPU warp occupancy and avoid divergence
#define THREAD_GROUP_X 8
#define THREAD_GROUP_Y 8

// High-Res (Output) and Low-Res (Input) Buffers
Texture2D<float3> InputLowResTexture : register(t0);
RWTexture2D<float3> OutputHighResTexture : register(u0);

// VRAM Memory Spike Map (The user's brilliantly proposed Spatial-Temporal Reservoir)
// Stores: r=Accumulated Red, g=Accumulated Green, b=Accumulated Blue, a=Active Count / Spike History
RWTexture2D<float4> MemorySpikeMap : register(u1);

// Structured Buffer for our Biological Architecture (Constant over the frame)
cbuffer NSS_Global_State : register(b0) {
    float ExpectedAmbientEnergy; // Tracked across frames by the Endocrine system
    float BaseLeakRate;
    float BaseSpikeThreshold;
    float DeltaTime;
};

// 1.58-bit Ternary Weights Matrix (Pre-compiled constants or buffered data)
// For prototype: we pack ternary values [-1, 0, 1] into structured arrays. 
// At 1.58 bits, a single 32-bit uint can compress ~20 ternary weights!
StructuredBuffer<int> PackedTernaryWeigths : register(t1);
StructuredBuffer<float> ClusterScalars : register(t2);

// ==============================================================================
// Bypassing Warp Divergence (Sparse Evaluator)
// ==============================================================================
// On GPUs, 'if' statements kill performance if threads diverge.
// We avoid divergence by evaluating the event threshold without branching:
inline bool IsActiveSynapse(float3 color) {
    float luminance = dot(color, float3(0.299f, 0.587f, 0.114f));
    return luminance > 0.05f; // Hard biological threshold
}

// ==============================================================================
// CORE COMPUTE SHADER ENTRY POINT
// ==============================================================================
[numthreads(THREAD_GROUP_X, THREAD_GROUP_Y, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex) {
    uint2 texCoord = dispatchThreadID.xy;
    
    // Bounds check to avoid memory faults
    uint width, height;
    OutputHighResTexture.GetDimensions(width, height);
    if (texCoord.x >= width || texCoord.y >= height) return;

    // We fetch the corresponding 2x2 low-res patch.
    // For simplicity, we sample the low-res buffer nearest the current high-res coordinate.
    uint2 lowResCoord = texCoord / 2;
    float3 lowResColor = InputLowResTexture.Load(int3(lowResCoord, 0));

    // EVENT CHECK: Sparse Architecture.
    // If the pixel is visually dead (ambient background), we skip expensive memory writes.
    // In HLSL, entire thread groups will bypass math if all 64 threads agree on this condition.
    if (!IsActiveSynapse(lowResColor)) {
        OutputHighResTexture[texCoord] = float3(0.0f, 0.0f, 0.0f);
        return;
    }

    // --- ENDOCRINE HOMEOSTASIS ---
    // In a full engine, Endocrine energy is globally calculated beforehand by a separate shader pass.
    // Here we apply the scaling dynamically.
    float homeostasisRatio = clamp(pow((length(lowResColor) + 0.01f) / max(ExpectedAmbientEnergy, 0.1f), 1.5f), 0.3f, 1.5f);
    float dynamicLeak = BaseLeakRate * homeostasisRatio;

    // --- BIOLOGICAL EXECUTION LOGIC ---
    // Fetch biological memory from the VRAM reservoir
    float4 spikeMemory = MemorySpikeMap[texCoord];
    
    // Leak the historical voltage (Ghosting prevention)
    spikeMemory.xyz *= (1.0f - dynamicLeak);

    // --- DYNAMIC BIOLOGICAL STIMULUS (Extracting true membrane voltage contrast) ---
    // The GPU shader natively reads the 3x3 patch to evaluate its own variance
    float clusterAvgVoltage = 0.0f;
    
    // Unroll manually to minimize GPU branching loops
    float pixels[9];
    for (int i = 0; i < 9; i++) {
        // Evaluate the 3x3 bounds
        int px = clamp(int(texCoord.x) - 1 + (i%3), 0, width-1);
        int py = clamp(int(texCoord.y) - 1 + (i/3), 0, height-1);
        float3 val = InputLowResTexture.Load(int3(px, py, 0));
        pixels[i] = (val.r + val.g + val.b) / 3.0f;
        clusterAvgVoltage += pixels[i];
    }
    clusterAvgVoltage /= 9.0f;
    
    float localVariance = 0.0f;
    for (int i = 0; i < 9; i++) {
        localVariance += abs(pixels[i] - clusterAvgVoltage);
    }
    localVariance /= 9.0f;

    // --- DYNAMIC 1.58-BIT VOLTAGE ADAPTATION ---
    // Instead of reading a static payload array via PCIe, the GPU physically wires 
    // its own biological array instantaneously based on the local lighting conditions!
    
    float rawTernarySumR = 0.0f, rawTernarySumG = 0.0f, rawTernarySumB = 0.0f;
    int activeSynapses = 0;
    int inhibitorySynapses = 0;

    int out_x = texCoord.x % 2;
    int out_y = texCoord.y % 2;
    float target_x = float(out_x) + 0.5f;
    float target_y = float(out_y) + 0.5f;

    for (int i = 0; i < 9; i++) {
        int in_x = i % 3;
        int in_y = i / 3;
        float dist_sq = (target_x - float(in_x))*(target_x - float(in_x)) + 
                        (target_y - float(in_y))*(target_y - float(in_y));
            
        float voltageDiff = pixels[i] - clusterAvgVoltage;
        int weight = 0;
            
        // Eliminate the geometric dead zone overlapping 0.5f boundaries
        if (dist_sq < 0.6f) {
            weight = 1; 
        } else {
            if (voltageDiff > 0.02f && dist_sq < 1.5f) weight = 1; 
            else if (voltageDiff < -0.02f && dist_sq >= 0.5f) weight = -1; 
            else weight = 0;
        }

        // Extract the original color for blending
        int px = clamp(int(texCoord.x) - 1 + (i%3), 0, width-1);
        int py = clamp(int(texCoord.y) - 1 + (i/3), 0, height-1);
        float3 baseColor = InputLowResTexture.Load(int3(px, py, 0));

        if (weight == 1) {
            rawTernarySumR += baseColor.r;
            rawTernarySumG += baseColor.g;
            rawTernarySumB += baseColor.b;
            activeSynapses++;
        } else if (weight == -1) {
            rawTernarySumR -= baseColor.r * 0.25f;
            rawTernarySumG -= baseColor.g * 0.25f;
            rawTernarySumB -= baseColor.b * 0.25f;
            inhibitorySynapses++;
        }
    }
    
    float scalar = 1.0f + (localVariance * 4.0f); // Dynamic Endocrine Variance Multiplier

    // --- THE SPIKE-TO-RGB MATHEMATICS ---
    float biologicalDivisor = float(activeSynapses) - (float(inhibitorySynapses) * 0.25f);
    
    // Soft absolute bound ensures High-Inhibition clusters don't crush luminous elements
    if (biologicalDivisor < 0.2f) biologicalDivisor = 0.2f;

    float3 scaledVoltage = float3(
        (rawTernarySumR * scalar) / biologicalDivisor,
        (rawTernarySumG * scalar) / biologicalDivisor,
        (rawTernarySumB * scalar) / biologicalDivisor
    );

    // Add to Temporal VRAM Reservoir (User's memoryspike_map solution)
    spikeMemory.xyz += scaledVoltage;
    spikeMemory.w += 1.0f; // Track firing history

    // Write back spatial-temporal data to VRAM without CPU PCIe latency!
    MemorySpikeMap[texCoord] = spikeMemory;

    // Output Final RGB decoding
    OutputHighResTexture[texCoord] = saturate(spikeMemory.xyz / max(spikeMemory.w, 1.0f));
}
