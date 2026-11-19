// Baseline FP16 Reference Compute Shader (SM 6.2 min16float)
// Standard dense floating-point convolution baseline for roofline comparison against Ternary

cbuffer LayerConfig : register(b0)
{
    uint InW;
    uint InH;
    uint OutW;
    uint OutH;
    uint LayerType;      // 0 = Dense Conv 3x3 (min16float), 1 = FP32 3x3, 2 = PixelShuffle 2x + Bilinear Skip
    uint InChannels;
    uint OutChannels;
    uint Activation;     // 0 = None, 1 = ReLU
    uint WeightOffset;
    uint ScaleOffset;
    uint BiasOffset;
    uint Reserved;
};

StructuredBuffer<float>   InputAct       : register(t0);
StructuredBuffer<uint>    TernaryWeights : register(t1); // unused in fp16 baseline
StructuredBuffer<float>   FPWeights      : register(t2); // dense float weights
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

    // ── Pass 0: Dense FP16 Multiply-Accumulate 3x3 Convolution ──
    if (LayerType == 0 || LayerType == 1)
    {
        min16float acc = (min16float)0.0;
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
                        min16float v = (min16float)InputAct[(chBase + (uint)iy) * InW + (uint)ix];
                        min16float w = (min16float)FPWeights[wIcBase + (uint)(ky * 3 + kx)];
                        acc += v * w;
                    }
                }
            }
        }

        min16float val = acc + (min16float)Biases[BiasOffset + oc];
        if (Activation == 1u)
            val = max(val, (min16float)0.0);

        OutputAct[(oc * OutH + y) * OutW + x] = (float)val;
    }
    // ── Pass 2: 12-Ch -> 2x RGB PixelShuffle + Global Bilinear Skip ──
    else if (LayerType == 2)
    {
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
