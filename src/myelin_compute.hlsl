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

// ========================================================================= //
// MYELIN-SR v2: Zero-Barrier Pipeline Engine
//
// PHILOSOPHY: Load once. Dispatch all. Let the GPU run.
// No mid-frame barriers. No shared memory bloat.
// Just fast, lean, pipelined ternary math.
//
// The GPU hardware scheduler handles write-after-write hazards
// between dispatches naturally when they write to the SAME UAV.
// We only need ONE barrier at the very end.
// ========================================================================= //

cbuffer LayerConfig : register(b0)
{
    uint InputWidth;
    uint InputHeight;
    uint LayerType;      // 0=ternary3x3, 1=ternary1x1, 2=FP, 3=PixelShuffle
    uint InChannels;
    uint OutChannels;
    uint Groups;
    uint WeightOffset;   // uint32 index into Weights
    uint ScaleOffset;    // float index into Scales
    uint BiasOffset;     // float index into Biases
    uint Activation;     // 0=none, 1=ReLU, 2=LeakyReLU
};

RWTexture2D<min16float4> OutputTex  : register(u0);
Texture2D<min16float4>   InputTex   : register(t0);
StructuredBuffer<uint>   Weights    : register(t1);
StructuredBuffer<float>  Scales     : register(t2);
StructuredBuffer<float>  Biases     : register(t3);

#define TILE 8
#define PAD 1
#define CDIM (TILE + 2)
groupshared min16float tile[4][CDIM][CDIM];

// ── Ternary 3x3 (multiplication-free) ──
[numthreads(TILE, TILE, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    // Load shared memory tile
    int2 lp = clamp(int2(dtid.xy) - int2(PAD,PAD), int2(0,0), int2(InputWidth-1, InputHeight-1));
    min16float4 t = InputTex[lp];
    tile[0][gtid.y][gtid.x] = t.x;
    tile[1][gtid.y][gtid.x] = t.y;
    tile[2][gtid.y][gtid.x] = t.z;
    tile[3][gtid.y][gtid.x] = t.w;
    GroupMemoryBarrierWithGroupSync();
    
    if (dtid.x >= InputWidth || dtid.y >= InputHeight) return;
    
    min16float4 result = min16float4(0,0,0,0);
    
    // ── TERNARY 3x3 ──
    if (LayerType == 0)
    {
        uint inPG = (Groups > 1) ? 1 : min(InChannels, 4u);
        
        for (uint oc = 0; oc < min(OutChannels, 4u); oc++)
        {
            min16float acc = 0;
            uint fpf = (inPG * 9 + 15) / 16;
            uint wb = WeightOffset + oc * fpf;
            uint wi = 0;
            uint pk = Weights[wb];
            
            for (uint ic = 0; ic < inPG; ic++)
            {
                [unroll] for (int dy = -1; dy <= 1; dy++)
                [unroll] for (int dx = -1; dx <= 1; dx++)
                {
                    min16float v = tile[ic][gtid.y+dy+PAD][gtid.x+dx+PAD];
                    uint bp = wi & 15;
                    if (wi > 0 && bp == 0) pk = Weights[++wb];
                    uint g = (pk >> (bp << 1)) & 3;
                    acc += (g == 1) ? v : ((g == 2) ? -v : (min16float)0);
                    wi++;
                }
            }
            result[oc] = acc * (min16float)Scales[ScaleOffset + oc] + (min16float)Biases[BiasOffset + oc];
            if (Activation == 1) result[oc] = max(result[oc], (min16float)0);
            else if (Activation == 2) result[oc] = result[oc] > 0 ? result[oc] : result[oc] * (min16float)0.1;
        }
    }
    // ── TERNARY 1x1 ──
    else if (LayerType == 1)
    {
        min16float4 inp = InputTex[dtid.xy];
        for (uint oc = 0; oc < min(OutChannels, 4u); oc++)
        {
            min16float acc = 0;
            uint wb = WeightOffset + oc * ((min(InChannels,4u)+15)/16);
            uint pk = Weights[wb];
            for (uint ic = 0; ic < min(InChannels, 4u); ic++)
            {
                uint bp = ic & 15;
                if (ic > 0 && bp == 0) pk = Weights[++wb];
                uint g = (pk >> (bp << 1)) & 3;
                acc += (g == 1) ? inp[ic] : ((g == 2) ? -inp[ic] : (min16float)0);
            }
            result[oc] = acc * (min16float)Scales[ScaleOffset + oc] + (min16float)Biases[BiasOffset + oc];
            if (Activation == 1) result[oc] = max(result[oc], (min16float)0);
        }
    }
    // ── PIXELSHUFFLE ──
    else if (LayerType == 3)
    {
        min16float4 p = InputTex[dtid.xy];
        uint2 ob = dtid.xy * 2;
        OutputTex[ob] = min16float4(p.x,p.x,p.x,1);
        OutputTex[ob+uint2(1,0)] = min16float4(p.y,p.y,p.y,1);
        OutputTex[ob+uint2(0,1)] = min16float4(p.z,p.z,p.z,1);
        OutputTex[ob+uint2(1,1)] = min16float4(p.w,p.w,p.w,1);
        return;
    }
    // ── PASSTHROUGH ──
    else
    {
        result = InputTex[dtid.xy];
    }
    
    OutputTex[dtid.xy] = result;
}
