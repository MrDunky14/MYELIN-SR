// MYELIN-SR v3: Numerically Verified Reference Compute Shader (SM 6.0)

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

[numthreads(8, 8, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID)
{
    uint x  = dtid.x;
    uint y  = dtid.y;
    uint oc = dtid.z;

    if (x >= OutW || y >= OutH || oc >= OutChannels)
        return;

    // ── Pass 0: Ternary 3x3 Convolution (Addition-Only, Zero-Padded) ──
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
                        float v = InputAct[(chBase + (uint)iy) * InW + (uint)ix];
                        acc += (code == 1u) ? v : -v;
                    }
                    wi++;
                }
            }
        }

        float val = acc * Scales[ScaleOffset + oc] + Biases[BiasOffset + oc];
        if (Activation == 1u)
            val = max(val, 0.0f);

        OutputAct[(oc * OutH + y) * OutW + x] = val;
    }
    // ── Pass 1: FP32 3x3 Reconstruction Head (Zero-Padded) ──
    else if (LayerType == 1)
    {
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
        // PyTorch PixelShuffle(2) indexing: inCh = c * 4 + sy * 2 + sx
        uint inX = x >> 1;
        uint inY = y >> 1;
        uint sx  = x & 1u;
        uint sy  = y & 1u;
        uint inCh = oc * 4u + sy * 2u + sx;
        float residual = InputAct[(inCh * InH + inY) * InW + inX];

        // Exact PyTorch F.interpolate(..., mode="bilinear", align_corners=False)
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
