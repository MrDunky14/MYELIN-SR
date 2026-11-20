// MYELIN-SR v3 Tiled: Cooperative LDS Halo Tiled Ternary Compute Shader (SM 6.0)
// Uses 10x10 groupshared tile per input channel to eliminate redundant global memory reads

cbuffer LayerConfig : register(b0)
{
    uint InW;
    uint InH;
    uint OutW;
    uint OutH;
    uint LayerType;      // 0 = Ternary 3x3, 1 = FP32 3x3, 2 = PixelShuffle 2x + Bilinear Skip
    uint InChannels;
    uint OutChannels;
    uint Activation;     // 0 = None, 1 = ReLU
    uint WeightOffset;   // uint32 index (type 0) or float index (type 1)
    uint ScaleOffset;    // float index
    uint BiasOffset;     // float index
    uint Reserved;
};

StructuredBuffer<float>   InputAct       : register(t0);
StructuredBuffer<uint>    TernaryWeights : register(t1);
StructuredBuffer<float>   FPWeights      : register(t2);
StructuredBuffer<float>   Scales         : register(t3);
StructuredBuffer<float>   Biases         : register(t4);
StructuredBuffer<float>   OrigLR         : register(t5);
RWStructuredBuffer<float> OutputAct      : register(u0);

#define TILE_DIM 8
#define HALO 1
#define SHARED_DIM (TILE_DIM + 2 * HALO) // 10x10 = 100 elements
#define THREADS_PER_GROUP (TILE_DIM * TILE_DIM) // 64 threads

groupshared float s_Tile[SHARED_DIM][SHARED_DIM];

[numthreads(TILE_DIM, TILE_DIM, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint gindex : SV_GroupIndex)
{
    uint x  = dtid.x;
    uint y  = dtid.y;
    uint oc = dtid.z;

    int originX = (int)(dtid.x - gtid.x) - HALO;
    int originY = (int)(dtid.y - gtid.y) - HALO;

    // ── Pass 0: Ternary 3x3 Convolution with Cooperative LDS Tile ──
    if (LayerType == 0)
    {
        uint wordsPerOC = (InChannels * 9u + 15u) / 16u;
        uint wb = WeightOffset + oc * wordsPerOC;
        uint wi = 0u;
        uint pk = TernaryWeights[wb];
        float acc = 0.0f;

        for (uint ic = 0u; ic < InChannels; ++ic)
        {
            uint chBase = ic * InH;

            // Cooperative LDS load: 64 threads load 100 elements safely
            for (uint i = gindex; i < (SHARED_DIM * SHARED_DIM); i += THREADS_PER_GROUP)
            {
                uint ly = i / SHARED_DIM;
                uint lx = i % SHARED_DIM;
                int sy = originY + (int)ly;
                int sx = originX + (int)lx;
                float v = 0.0f;
                if (sy >= 0 && sy < (int)InH && sx >= 0 && sx < (int)InW)
                {
                    v = InputAct[(chBase + (uint)sy) * InW + (uint)sx];
                }
                s_Tile[ly][lx] = v;
            }
            GroupMemoryBarrierWithGroupSync();

            if (x < OutW && y < OutH && oc < OutChannels)
            {
                [unroll]
                for (int ky = 0; ky < 3; ++ky)
                {
                    [unroll]
                    for (int kx = 0; kx < 3; ++kx)
                    {
                        uint bp = wi & 15u;
                        if (wi > 0u && bp == 0u)
                        {
                            pk = TernaryWeights[++wb];
                        }
                        uint code = (pk >> (bp << 1)) & 3u;
                        if (code != 0u)
                        {
                            float v = s_Tile[gtid.y + (uint)ky][gtid.x + (uint)kx];
                            acc += (code == 1u) ? v : -v;
                        }
                        wi++;
                    }
                }
            }
            else
            {
                wi += 9u;
            }
            GroupMemoryBarrierWithGroupSync();
        }

        if (x < OutW && y < OutH && oc < OutChannels)
        {
            float val = acc * Scales[ScaleOffset + oc] + Biases[BiasOffset + oc];
            if (Activation == 1u)
                val = max(val, 0.0f);
            OutputAct[(oc * OutH + y) * OutW + x] = val;
        }
    }
    // ── Pass 1: FP32 3x3 Reconstruction Head ──
    else if (LayerType == 1)
    {
        if (x >= OutW || y >= OutH || oc >= OutChannels) return;

        float acc = 0.0f;
        uint wOcBase = WeightOffset + oc * InChannels * 9u;

        for (uint ic = 0u; ic < InChannels; ++ic)
        {
            uint chBase = ic * InH;
            uint wIcBase = wOcBase + ic * 9u;
            [unroll]
            for (int ky = 0; ky < 3; ++ky)
            {
                int iy = (int)y + ky - 1;
                [unroll]
                for (int kx = 0; kx < 3; ++kx)
                {
                    int ix = (int)x + kx - 1;
                    if (iy >= 0 && iy < (int)InH && ix >= 0 && ix < (int)InW)
                    {
                        float v = InputAct[(chBase + (uint)iy) * InW + (uint)ix];
                        float w = FPWeights[wIcBase + (uint)(ky * 3 + kx)];
                        acc += v * w;
                    }
                }
            }
        }

        float val = acc + Biases[BiasOffset + oc];
        if (Activation == 1u)
            val = max(val, 0.0f);
        OutputAct[(oc * OutH + y) * OutW + x] = val;
    }
    // ── Pass 2: True 12-Ch -> 2x RGB PixelShuffle + Global Bilinear Skip ──
    else if (LayerType == 2)
    {
        if (x >= OutW || y >= OutH || oc >= OutChannels) return;

        uint inX = x >> 1;
        uint inY = y >> 1;
        uint sx  = x & 1u;
        uint sy  = y & 1u;
        uint inCh = oc * 4u + sy * 2u + sx;
        float residual = InputAct[(inCh * InH + inY) * InW + inX];

        float srcX = clamp(((float)x + 0.5f) * 0.5f - 0.5f, 0.0f, (float)(InW - 1u));
        float srcY = clamp(((float)y + 0.5f) * 0.5f - 0.5f, 0.0f, (float)(InH - 1u));

        uint x0 = (uint)floor(srcX);
        uint y0 = (uint)floor(srcY);
        uint x1 = min(x0 + 1u, InW - 1u);
        uint y1 = min(y0 + 1u, InH - 1u);
        float wx = srcX - (float)x0;
        float wy = srcY - (float)y0;

        uint base = oc * InH;
        float p00 = OrigLR[(base + y0) * InW + x0];
        float p10 = OrigLR[(base + y0) * InW + x1];
        float p01 = OrigLR[(base + y1) * InW + x0];
        float p11 = OrigLR[(base + y1) * InW + x1];

        float top = lerp(p00, p10, wx);
        float bot = lerp(p01, p11, wx);
        float bilinear = lerp(top, bot, wy);

        OutputAct[(oc * OutH + y) * OutW + x] = residual + bilinear;
    }
}
