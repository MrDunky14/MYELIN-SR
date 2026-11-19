// MYELIN-SR v3: Multi-Variant Roofline Benchmark Shader
// Compile flags:
//   -DUSE_LDS=0|1      (0 = 8x8 global reads, 1 = 16x16 cooperative 18x18 LDS)
//   -DFORCE_DENSE=0|1  (0 = 2-bit ternary unpack, 1 = dense FMA on same weights)
//   -DUSE_FP16=0|1     (0 = FP32 StructuredBuffer, 1 = FP16 ByteAddressBuffer SM 6.2)

#ifndef USE_LDS
#define USE_LDS 0
#endif

#ifndef FORCE_DENSE
#define FORCE_DENSE 0
#endif

#ifndef USE_FP16
#define USE_FP16 0
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
    uint DenseOffset;    // element index into DenseWeights
    uint ScaleOffset;    // float index into Scales
    uint BiasOffset;     // float index into Biases
};

StructuredBuffer<uint>  TernaryWeights : register(t1);
StructuredBuffer<float> Scales         : register(t3);
StructuredBuffer<float> Biases         : register(t4);

#if USE_FP16
typedef float16_t act_t;
ByteAddressBuffer   InputAct     : register(t0);
ByteAddressBuffer   DenseWeights : register(t2);
ByteAddressBuffer   OrigLR       : register(t5);
RWByteAddressBuffer OutputAct    : register(u0);
#define LOAD_ACT(buf, idx)        ((buf).Load<float16_t>((idx) * 2u))
#define STORE_ACT(buf, idx, val)  ((buf).Store<float16_t>((idx) * 2u, (val)))
#else
typedef float act_t;
StructuredBuffer<float>   InputAct     : register(t0);
StructuredBuffer<float>   DenseWeights : register(t2);
StructuredBuffer<float>   OrigLR       : register(t5);
RWStructuredBuffer<float> OutputAct    : register(u0);
#define LOAD_ACT(buf, idx)        ((buf)[(idx)])
#define STORE_ACT(buf, idx, val)  ((buf)[(idx)] = (val))
#endif

#if USE_LDS
#define TILE_DIM 16
#define HALO 1
#define SHARED_DIM 18
#define THREADS_PER_GROUP 256
groupshared act_t s_Tile[SHARED_DIM][SHARED_DIM];
#else
#define TILE_DIM 8
#endif

[numthreads(TILE_DIM, TILE_DIM, 1)]
void CSMain(
    uint3 gid    : SV_GroupID,
    uint3 gtid   : SV_GroupThreadID,
    uint3 dtid   : SV_DispatchThreadID,
    uint  gindex : SV_GroupIndex)
{
    uint x  = dtid.x;
    uint y  = dtid.y;
    uint oc = dtid.z;

    // ── Pass 0 & Pass 1: 3x3 Convolutions ──
    if (LayerType == 0u || LayerType == 1u)
    {
        bool useTernary = (LayerType == 0u) && (FORCE_DENSE == 0);
        act_t acc = (act_t)0;

#if USE_LDS
        int baseX = (int)(gid.x * TILE_DIM) - HALO;
        int baseY = (int)(gid.y * TILE_DIM) - HALO;

        uint wordsPerOC = (InChannels * 9u + 15u) / 16u;
        uint wb = TernaryOffset + oc * wordsPerOC;
        uint wi = 0u;
        uint pk = useTernary ? TernaryWeights[wb] : 0u;
        uint wOcBase = DenseOffset + oc * InChannels * 9u;

        for (uint ic = 0u; ic < InChannels; ++ic)
        {
            // Cooperative 18x18 halo load (ALL 256 threads participate; no early exit)
            uint chBase = ic * InH;
            for (uint idx = gindex; idx < (SHARED_DIM * SHARED_DIM); idx += THREADS_PER_GROUP)
            {
                uint ly = idx / SHARED_DIM;
                uint lx = idx % SHARED_DIM;
                int gx = baseX + (int)lx;
                int gy = baseY + (int)ly;
                act_t v = (act_t)0;
                if (gx >= 0 && gx < (int)InW && gy >= 0 && gy < (int)InH)
                {
                    v = LOAD_ACT(InputAct, (chBase + (uint)gy) * InW + (uint)gx);
                }
                s_Tile[ly][lx] = v;
            }
            GroupMemoryBarrierWithGroupSync();

            if (useTernary)
            {
                // Warp-uniform weight unpacking: all threads in group share `oc`,
                // so code == 0 skips both LDS fetch and ALU with zero divergence.
                [unroll]
                for (uint ky = 0u; ky < 3u; ++ky)
                {
                    [unroll]
                    for (uint kx = 0u; kx < 3u; ++kx)
                    {
                        uint bp = wi & 15u;
                        if (wi > 0u && bp == 0u)
                        {
                            pk = TernaryWeights[++wb];
                        }
                        uint code = (pk >> (bp << 1)) & 3u;
                        if (code == 1u)
                            acc += s_Tile[gtid.y + ky][gtid.x + kx];
                        else if (code == 2u)
                            acc -= s_Tile[gtid.y + ky][gtid.x + kx];
                        wi++;
                    }
                }
            }
            else
            {
                uint wIcBase = wOcBase + ic * 9u;
                [unroll]
                for (uint ky = 0u; ky < 3u; ++ky)
                {
                    [unroll]
                    for (uint kx = 0u; kx < 3u; ++kx)
                    {
                        act_t w = LOAD_ACT(DenseWeights, wIcBase + ky * 3u + kx);
                        acc += s_Tile[gtid.y + ky][gtid.x + kx] * w;
                    }
                }
            }

            // Prevent fast warps from overwriting s_Tile before slow warps finish reading
            GroupMemoryBarrierWithGroupSync();
        }
#else
        if (x >= OutW || y >= OutH || oc >= OutChannels)
            return;

        if (useTernary)
        {
            uint wordsPerOC = (InChannels * 9u + 15u) / 16u;
            uint wb = TernaryOffset + oc * wordsPerOC;
            uint wi = 0u;
            uint pk = TernaryWeights[wb];

            for (uint ic = 0u; ic < InChannels; ++ic)
            {
                uint chBase = ic * InH;
                [unroll]
                for (int ky = 0; ky < 3; ++ky)
                {
                    int iy = (int)y + ky - 1;
                    [unroll]
                    for (int kx = 0; kx < 3; ++kx)
                    {
                        int ix = (int)x + kx - 1;
                        uint bp = wi & 15u;
                        if (wi > 0u && bp == 0u)
                        {
                            pk = TernaryWeights[++wb];
                        }
                        uint code = (pk >> (bp << 1)) & 3u;
                        if (code != 0u && iy >= 0 && iy < (int)InH && ix >= 0 && ix < (int)InW)
                        {
                            act_t v = LOAD_ACT(InputAct, (chBase + (uint)iy) * InW + (uint)ix);
                            acc += (code == 1u) ? v : -v;
                        }
                        wi++;
                    }
                }
            }
        }
        else
        {
            uint wOcBase = DenseOffset + oc * InChannels * 9u;
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
                            act_t v = LOAD_ACT(InputAct, (chBase + (uint)iy) * InW + (uint)ix);
                            act_t w = LOAD_ACT(DenseWeights, wIcBase + (uint)(ky * 3 + kx));
                            acc += v * w;
                        }
                    }
                }
            }
        }
#endif

        if (x >= OutW || y >= OutH || oc >= OutChannels)
            return;

        float val = (float)acc;
        if (useTernary)
            val *= Scales[ScaleOffset + oc];
        val += Biases[BiasOffset + oc];
        if (Activation == 1u)
            val = max(val, 0.0f);

        STORE_ACT(OutputAct, (oc * OutH + y) * OutW + x, (act_t)val);
    }
    // ── Pass 2: True 12-Ch -> 2x RGB PixelShuffle + Global Bilinear Skip ──
    else if (LayerType == 2u)
    {
        if (x >= OutW || y >= OutH || oc >= OutChannels)
            return;

        uint inX = x >> 1;
        uint inY = y >> 1;
        uint sx  = x & 1u;
        uint sy  = y & 1u;
        uint inCh = oc * 4u + sy * 2u + sx;
        float residual = (float)LOAD_ACT(InputAct, (inCh * InH + inY) * InW + inX);

        // Keep coordinate interpolation in FP32 to avoid 1080p FP16 ULP loss
        float srcX = clamp(((float)x + 0.5f) * 0.5f - 0.5f, 0.0f, (float)(InW - 1u));
        float srcY = clamp(((float)y + 0.5f) * 0.5f - 0.5f, 0.0f, (float)(InH - 1u));

        uint x0 = (uint)floor(srcX);
        uint y0 = (uint)floor(srcY);
        uint x1 = min(x0 + 1u, InW - 1u);
        uint y1 = min(y0 + 1u, InH - 1u);
        float wx = srcX - (float)x0;
        float wy = srcY - (float)y0;

        uint base = oc * InH;
        float p00 = (float)LOAD_ACT(OrigLR, (base + y0) * InW + x0);
        float p10 = (float)LOAD_ACT(OrigLR, (base + y0) * InW + x1);
        float p01 = (float)LOAD_ACT(OrigLR, (base + y1) * InW + x0);
        float p11 = (float)LOAD_ACT(OrigLR, (base + y1) * InW + x1);

        float bilinear = lerp(lerp(p00, p10, wx), lerp(p01, p11, wx), wy);
        STORE_ACT(OutputAct, (oc * OutH + y) * OutW + x, (act_t)(residual + bilinear));
    }
}

