// MYELIN-SR v3.3: OC_TILE=8 Register-Blocked + float16_t2 Vectorized Kernel (SM 6.2)
// Compile flags:
//   -DFORCE_DENSE=0|1  (0 = 2-bit ternary unpack, 1 = dense FP16 HFMA2)

#ifndef FORCE_DENSE
#define FORCE_DENSE 1
#endif

cbuffer LayerConfig : register(b0)
{
    uint InW;
    uint InH;
    uint OutW;
    uint OutH;
    uint LayerType;      // 0 = Backbone 3x3, 1 = Head 3x3, 2 = PixelShuffle 2x + Bilinear Skip
    uint InChannels;
    uint OutChannels;
    uint Activation;     // 0 = None, 1 = ReLU
    uint TernaryOffset;  // uint32 index into TernaryWeights
    uint DenseOffset;    // float16_t element index into DenseWeights
    uint ScaleOffset;    // float index into Scales
    uint BiasOffset;     // float index into Biases
};

ByteAddressBuffer       InputAct       : register(t0);
StructuredBuffer<uint>  TernaryWeights : register(t1);
ByteAddressBuffer       DenseWeights   : register(t2);
StructuredBuffer<float> Scales         : register(t3);
StructuredBuffer<float> Biases         : register(t4);
ByteAddressBuffer       OrigLR         : register(t5);
RWByteAddressBuffer     OutputAct      : register(u0);

#define LOAD_H1(buf, idx)        ((buf).Load<float16_t>((idx) * 2u))
#define STORE_H1(buf, idx, val)  ((buf).Store<float16_t>((idx) * 2u, (val)))
#define STORE_H2(buf, idx, val)  ((buf).Store<float16_t2>((idx) * 2u, (val)))

#define TILE_W 32
#define TILE_H 16
#define SHARED_W 34
#define SHARED_H 18
#define THREADS_PER_GROUP 256
#define OC_TILE 8

// Pad stride to 36 to avoid 32-bank LDS conflicts on 16-bit accesses
groupshared float16_t s_Tile[SHARED_H][36];

[numthreads(16, 16, 1)]
void CSMain(
    uint3 gid    : SV_GroupID,
    uint3 gtid   : SV_GroupThreadID,
    uint3 dtid   : SV_DispatchThreadID,
    uint  gindex : SV_GroupIndex)
{
    // ── Pass 0 & Pass 1: OC_TILE=8 + float16_t2 3x3 Convolution ──
    if (LayerType == 0u || LayerType == 1u)
    {
        bool useTernary = (LayerType == 0u) && (FORCE_DENSE == 0);
        uint ocBase = gid.z * OC_TILE;
        int  baseX  = (int)(gid.x * TILE_W) - 1;
        int  baseY  = (int)(gid.y * TILE_H) - 1;

        float16_t2 acc[OC_TILE];
        [unroll]
        for (uint k = 0u; k < OC_TILE; ++k)
            acc[k] = float16_t2(0.0h, 0.0h);

        uint wordsPerOC = (InChannels * 9u + 15u) / 16u;
        uint tx = gtid.x * 2u;
        uint ty = gtid.y;

        for (uint ic = 0u; ic < InChannels; ++ic)
        {
            // Cooperative 34x18 halo load (ALL 256 threads participate; no early exit)
            uint chBase = ic * InH;
            for (uint idx = gindex; idx < (SHARED_W * SHARED_H); idx += THREADS_PER_GROUP)
            {
                uint ly = idx / SHARED_W;
                uint lx = idx % SHARED_W;
                int gx = baseX + (int)lx;
                int gy = baseY + (int)ly;
                float16_t v = (float16_t)0;
                if (gx >= 0 && gx < (int)InW && gy >= 0 && gy < (int)InH)
                    v = LOAD_H1(InputAct, (chBase + (uint)gy) * InW + (uint)gx);
                s_Tile[ly][lx] = v;
            }
            GroupMemoryBarrierWithGroupSync();

            // Load 3x4 neighborhood ONCE into registers -> 9 float16_t2 vectors reused across 8 OCs
            float16_t2 v[9];
            [unroll]
            for (uint ky = 0u; ky < 3u; ++ky)
            {
                float16_t a0 = s_Tile[ty + ky][tx + 0u];
                float16_t a1 = s_Tile[ty + ky][tx + 1u];
                float16_t a2 = s_Tile[ty + ky][tx + 2u];
                float16_t a3 = s_Tile[ty + ky][tx + 3u];
                v[ky * 3u + 0u] = float16_t2(a0, a1);
                v[ky * 3u + 1u] = float16_t2(a1, a2);
                v[ky * 3u + 2u] = float16_t2(a2, a3);
            }

            if (useTernary)
            {
                uint wiBase = ic * 9u;
                uint wIdx0 = wiBase >> 4;
                uint wIdx1 = (wiBase + 8u) >> 4;

                [unroll]
                for (uint k = 0u; k < OC_TILE; ++k)
                {
                    uint oc = ocBase + k;
                    if (oc < OutChannels)
                    {
                        uint ocWordBase = TernaryOffset + oc * wordsPerOC;
                        uint pk0 = TernaryWeights[ocWordBase + wIdx0];
                        uint pk1 = (wIdx1 != wIdx0) ? TernaryWeights[ocWordBase + wIdx1] : pk0;

                        [unroll]
                        for (uint t = 0u; t < 9u; ++t)
                        {
                            uint wi = wiBase + t;
                            uint word = ((wi >> 4) == wIdx0) ? pk0 : pk1;
                            uint code = (word >> ((wi & 15u) << 1)) & 3u;
                            if (code == 1u)
                                acc[k] += v[t];
                            else if (code == 2u)
                                acc[k] -= v[t];
                        }
                    }
                }
            }
            else
            {
                [unroll]
                for (uint k = 0u; k < OC_TILE; ++k)
                {
                    uint oc = ocBase + k;
                    if (oc < OutChannels)
                    {
                        uint wBase = DenseOffset + (oc * InChannels + ic) * 9u;
                        [unroll]
                        for (uint t = 0u; t < 9u; ++t)
                        {
                            float16_t w = LOAD_H1(DenseWeights, wBase + t);
                            acc[k] += v[t] * w; // Compiles to packed HFMA2
                        }
                    }
                }
            }

            GroupMemoryBarrierWithGroupSync();
        }

        uint x0 = gid.x * TILE_W + gtid.x * 2u;
        uint y  = gid.y * TILE_H + gtid.y;
        if (y >= OutH || x0 >= OutW)
            return;

        [unroll]
        for (uint k = 0u; k < OC_TILE; ++k)
        {
            uint oc = ocBase + k;
            if (oc < OutChannels)
            {
                float2 val = (float2)acc[k];
                if (useTernary)
                    val *= Scales[ScaleOffset + oc];
                val += Biases[BiasOffset + oc];
                if (Activation == 1u)
                    val = max(val, float2(0.0f, 0.0f));

                uint outIdx = (oc * OutH + y) * OutW + x0;
                if (x0 + 1u < OutW && ((outIdx & 1u) == 0u))
                {
                    STORE_H2(OutputAct, outIdx, (float16_t2)val);
                }
                else
                {
                    STORE_H1(OutputAct, outIdx, (float16_t)val.x);
                    if (x0 + 1u < OutW)
                        STORE_H1(OutputAct, outIdx + 1u, (float16_t)val.y);
                }
            }
        }
    }
    // ── Pass 2: True 12-Ch -> 2x RGB PixelShuffle + Global Bilinear Skip ──
    else if (LayerType == 2u)
    {
        uint x  = dtid.x;
        uint y  = dtid.y;
        uint oc = dtid.z;
        if (x >= OutW || y >= OutH || oc >= OutChannels)
            return;

        uint inX = x >> 1;
        uint inY = y >> 1;
        uint sx  = x & 1u;
        uint sy  = y & 1u;
        uint inCh = oc * 4u + sy * 2u + sx;
        float residual = (float)LOAD_H1(InputAct, (inCh * InH + inY) * InW + inX);

        float srcX = clamp(((float)x + 0.5f) * 0.5f - 0.5f, 0.0f, (float)(InW - 1u));
        float srcY = clamp(((float)y + 0.5f) * 0.5f - 0.5f, 0.0f, (float)(InH - 1u));

        uint x0 = (uint)floor(srcX);
        uint y0 = (uint)floor(srcY);
        uint x1 = min(x0 + 1u, InW - 1u);
        uint y1 = min(y0 + 1u, InH - 1u);
        float wx = srcX - (float)x0;
        float wy = srcY - (float)y0;

        uint base = oc * InH;
        float p00 = (float)LOAD_H1(OrigLR, (base + y0) * InW + x0);
        float p10 = (float)LOAD_H1(OrigLR, (base + y0) * InW + x1);
        float p01 = (float)LOAD_H1(OrigLR, (base + y1) * InW + x0);
        float p11 = (float)LOAD_H1(OrigLR, (base + y1) * InW + x1);

        float bilinear = lerp(lerp(p00, p10, wx), lerp(p01, p11, wx), wy);
        STORE_H1(OutputAct, (oc * OutH + y) * OutW + x, (float16_t)(residual + bilinear));
    }
}
