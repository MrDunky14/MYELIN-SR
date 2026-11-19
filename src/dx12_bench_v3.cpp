#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

#define CHECK_HR(hr, msg)                                                      \
    if (FAILED(hr)) {                                                          \
        throw std::runtime_error(std::string(msg) + " failed (HR=0x" +         \
                                 std::to_string((uint32_t)(hr)) + ")");        \
    }

struct LayerDescBench {
    uint32_t type, inCh, outCh, activation;
    uint32_t ternaryOff, denseOff, scaleOff, biasOff;
};

struct GpuLayerConfig {
    uint32_t InW, InH, OutW, OutH;
    uint32_t LayerType, InChannels, OutChannels, Activation;
    uint32_t TernaryOffset, DenseOffset, ScaleOffset, BiasOffset;
};

struct VariantSpec {
    std::string name;
    std::string csoFile;
    uint32_t tileX;
    uint32_t tileY;
    uint32_t ocTile;
    bool useFp16;
    bool useDense;
};

static std::vector<char> ReadAllBytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open file: " + path);
    size_t sz = (size_t)f.tellg();
    f.seekg(0);
    std::vector<char> buf(sz);
    f.read(buf.data(), sz);
    return buf;
}

static float HalfToFloat(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t fBits = 0;
    if (exp == 0) {
        if (mant == 0) {
            fBits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x03FFu;
            fBits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        fBits = sign | 0x7F800000u | (mant << 13);
    } else {
        fBits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &fBits, sizeof(out));
    return out;
}

class RooflineBench {
public:
    void RunAll(const std::string& dir, int warmups, int iters) {
        InitDevice();
        LoadBundle(dir);
        CreateRootSignature();

        std::vector<VariantSpec> variants = {
            { "1. Ternary Naive (FP32)", dir + "/v3_ternary_naive_fp32.cso", 8,  8,  1, false, false },
            { "2. Ternary LDS   (FP32)", dir + "/v3_ternary_lds_fp32.cso",   16, 16, 1, false, false },
            { "3. Dense LDS     (FP32)", dir + "/v3_dense_lds_fp32.cso",     16, 16, 1, false, true  },
            { "4. Ternary LDS   (FP16)", dir + "/v3_ternary_lds_fp16.cso",   16, 16, 1, true,  false },
            { "5. Dense LDS     (FP16)", dir + "/v3_dense_lds_fp16.cso",     16, 16, 1, true,  true  },
            { "6. Ternary OC8-Vec2 (FP16)", dir + "/v3_ternary_oc8_fp16.cso", 32, 16, 8, true, false },
            { "7. Dense OC8-Vec2   (FP16)", dir + "/v3_dense_oc8_fp16.cso",   32, 16, 8, true, true  },
        };

        std::cout << "\n| Variant | Parity | MaxErr | Median (ms) | P95 (ms) | FPS | Eff. BW (GB/s) |\n";
        std::cout << "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n";

        for (const auto& v : variants) {
            if (v.useFp16 && !m_nativeFp16) {
                std::cout << "| " << v.name << " | SKIP (No Native FP16) | - | - | - | - | - |\n";
                continue;
            }
            RunVariant(v, warmups, iters);
        }
    }

private:
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12CommandQueue> m_queue;
    ComPtr<ID3D12CommandAllocator> m_alloc;
    ComPtr<ID3D12GraphicsCommandList> m_cmd;
    ComPtr<ID3D12Fence> m_fence;
    HANDLE m_fenceEvent = nullptr;
    uint64_t m_fenceVal = 1;
    bool m_nativeFp16 = false;

    ComPtr<ID3D12RootSignature> m_rootSig;
    ComPtr<ID3D12Resource> m_ternaryBuf, m_denseFp32Buf, m_denseFp16Buf, m_scaleBuf, m_biasBuf;
    ComPtr<ID3D12Resource> m_origLrFp32, m_origLrFp16;
    ComPtr<ID3D12Resource> m_pingA, m_pingB, m_readbackBuf;
    D3D12_RESOURCE_STATES m_stateA = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES m_stateB = D3D12_RESOURCE_STATE_COMMON;

    ComPtr<ID3D12QueryHeap> m_tsHeap;
    ComPtr<ID3D12Resource> m_tsReadback;

    uint32_t m_inW = 0, m_inH = 0, m_outW = 0, m_outH = 0;
    uint64_t m_ternaryBytes = 0, m_dense32Bytes = 0, m_dense16Bytes = 0;
    std::vector<LayerDescBench> m_layers;
    std::vector<float> m_expectedHr;

    void WaitGPU() {
        uint64_t v = m_fenceVal++;
        CHECK_HR(m_queue->Signal(m_fence.Get(), v), "Fence Signal");
        if (m_fence->GetCompletedValue() < v) {
            CHECK_HR(m_fence->SetEventOnCompletion(v, m_fenceEvent), "Fence Event");
            WaitForSingleObject(m_fenceEvent, INFINITE);
        }
    }

    ComPtr<ID3D12Resource> CreateBuffer(uint64_t bytes, D3D12_HEAP_TYPE heapType,
                                        D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state) {
        if (bytes == 0) bytes = 4;
        D3D12_HEAP_PROPERTIES hp = {}; hp.Type = heapType;
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = bytes; rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; rd.Flags = flags;
        ComPtr<ID3D12Resource> res;
        CHECK_HR(m_device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr,
                                                   IID_PPV_ARGS(&res)), "CreateBuffer");
        return res;
    }

    void Transition(ID3D12Resource* res, D3D12_RESOURCE_STATES& curr, D3D12_RESOURCE_STATES next) {
        if (curr == next) return;
        D3D12_RESOURCE_BARRIER b = {};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition = { res, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, curr, next };
        m_cmd->ResourceBarrier(1, &b);
        curr = next;
    }

    ComPtr<ID3D12Resource> UploadToDefault(const void* data, uint64_t bytes) {
        uint64_t alloc = (bytes == 0) ? 4 : bytes;
        auto staging = CreateBuffer(alloc, D3D12_HEAP_TYPE_UPLOAD,
                                    D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        void* p = nullptr;
        CHECK_HR(staging->Map(0, nullptr, &p), "Stage Map");
        if (bytes > 0) std::memcpy(p, data, bytes);
        else std::memset(p, 0, 4);
        staging->Unmap(0, nullptr);

        auto gpu = CreateBuffer(alloc, D3D12_HEAP_TYPE_DEFAULT,
                                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        CHECK_HR(m_alloc->Reset(), "Up Alloc");
        CHECK_HR(m_cmd->Reset(m_alloc.Get(), nullptr), "Up Cmd");
        m_cmd->CopyBufferRegion(gpu.Get(), 0, staging.Get(), 0, alloc);
        D3D12_RESOURCE_STATES st = D3D12_RESOURCE_STATE_COPY_DEST;
        Transition(gpu.Get(), st, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        CHECK_HR(m_cmd->Close(), "Up Close");
        ID3D12CommandList* lists[] = { m_cmd.Get() };
        m_queue->ExecuteCommandLists(1, lists);
        WaitGPU();
        return gpu;
    }

    void InitDevice() {
        ComPtr<IDXGIFactory4> factory;
        CHECK_HR(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "Factory");

        ComPtr<IDXGIAdapter1> bestAdapter, cand;
        SIZE_T bestVram = 0;
        std::wstring bestName;
        for (UINT i = 0; factory->EnumAdapters1(i, &cand) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 d;
            cand->GetDesc1(&d);
            if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            if (SUCCEEDED(D3D12CreateDevice(cand.Get(), D3D_FEATURE_LEVEL_12_0, _uuidof(ID3D12Device), nullptr))) {
                if (d.DedicatedVideoMemory > bestVram) {
                    bestVram = d.DedicatedVideoMemory;
                    bestAdapter = cand;
                    bestName = d.Description;
                }
            }
        }
        if (!bestAdapter) throw std::runtime_error("No hardware D3D12 GPU found");
        CHECK_HR(D3D12CreateDevice(bestAdapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&m_device)), "Device");

        D3D12_FEATURE_DATA_D3D12_OPTIONS4 opt4 = {};
        if (SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &opt4, sizeof(opt4))))
            m_nativeFp16 = (opt4.Native16BitShaderOpsSupported != FALSE);

        wprintf(L"[GPU] %s (%llu MB VRAM | Native FP16: %s)\n",
                bestName.c_str(), (unsigned long long)(bestVram / (1024 * 1024)),
                m_nativeFp16 ? L"YES" : L"NO");

        D3D12_COMMAND_QUEUE_DESC qd = {}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        CHECK_HR(m_device->CreateCommandQueue(&qd, IID_PPV_ARGS(&m_queue)), "Queue");
        CHECK_HR(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_alloc)), "Alloc");
        CHECK_HR(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_alloc.Get(), nullptr, IID_PPV_ARGS(&m_cmd)), "Cmd");
        CHECK_HR(m_cmd->Close(), "Cmd Close");
        CHECK_HR(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)), "Fence");
        m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    }

    void LoadBundle(const std::string& dir) {
        auto bin = ReadAllBytes(dir + "/myelin_bench_v3.bin");
        if (bin.size() < 20 || std::memcmp(bin.data(), "MBN3", 4) != 0)
            throw std::runtime_error("Invalid MBN3 header");

        const char* p = bin.data() + 4;
        uint32_t ver, numLayers;
        std::memcpy(&ver, p, 4); p += 4;
        std::memcpy(&numLayers, p, 4); p += 4;
        std::memcpy(&m_inW, p, 4); p += 4;
        std::memcpy(&m_inH, p, 4); p += 4;
        m_outW = m_inW * 2; m_outH = m_inH * 2;

        m_layers.resize(numLayers);
        std::memcpy(m_layers.data(), p, numLayers * sizeof(LayerDescBench));
        p += numLayers * sizeof(LayerDescBench);

        uint64_t sz[5];
        std::memcpy(sz, p, sizeof(sz)); p += sizeof(sz);
        m_ternaryBytes = sz[0]; m_dense32Bytes = sz[1]; m_dense16Bytes = sz[2];

        m_ternaryBuf   = UploadToDefault(p, sz[0]); p += sz[0];
        m_denseFp32Buf = UploadToDefault(p, sz[1]); p += sz[1];
        m_denseFp16Buf = UploadToDefault(p, sz[2]); p += sz[2];
        m_scaleBuf     = UploadToDefault(p, sz[3]); p += sz[3];
        m_biasBuf      = UploadToDefault(p, sz[4]); p += sz[4];

        auto lr32 = ReadAllBytes(dir + "/input_lr_fp32.bin");
        auto lr16 = ReadAllBytes(dir + "/input_lr_fp16.bin");
        m_origLrFp32 = UploadToDefault(lr32.data(), lr32.size());
        m_origLrFp16 = UploadToDefault(lr16.data(), lr16.size());

        auto expBytes = ReadAllBytes(dir + "/expected_hr_fp32.bin");
        m_expectedHr.resize(expBytes.size() / sizeof(float));
        std::memcpy(m_expectedHr.data(), expBytes.data(), expBytes.size());

        uint64_t maxElems = (uint64_t)3 * m_outW * m_outH;
        for (const auto& l : m_layers)
            maxElems = std::max(maxElems, (uint64_t)l.outCh * m_inW * m_inH);

        m_pingA = CreateBuffer(maxElems * sizeof(float), D3D12_HEAP_TYPE_DEFAULT,
                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        m_pingB = CreateBuffer(maxElems * sizeof(float), D3D12_HEAP_TYPE_DEFAULT,
                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        m_readbackBuf = CreateBuffer((uint64_t)3 * m_outW * m_outH * sizeof(float),
                                     D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                                     D3D12_RESOURCE_STATE_COPY_DEST);

        uint32_t tsCount = (uint32_t)m_layers.size() + 1;
        D3D12_QUERY_HEAP_DESC qhd = { D3D12_QUERY_HEAP_TYPE_TIMESTAMP, tsCount, 0 };
        CHECK_HR(m_device->CreateQueryHeap(&qhd, IID_PPV_ARGS(&m_tsHeap)), "TS Heap");
        m_tsReadback = CreateBuffer(tsCount * sizeof(uint64_t), D3D12_HEAP_TYPE_READBACK,
                                    D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

        std::cout << "[Workload] " << m_inW << "x" << m_inH << " -> " << m_outW << "x" << m_outH
                  << " | " << numLayers << " layers\n";
    }

    void CreateRootSignature() {
        D3D12_ROOT_PARAMETER params[8] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = { 0, 0, sizeof(GpuLayerConfig) / 4 };
        for (UINT i = 0; i < 6; ++i) {
            params[1 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            params[1 + i].Descriptor = { i, 0 };
        }
        params[7].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[7].Descriptor = { 0, 0 };

        D3D12_ROOT_SIGNATURE_DESC rsd = { 8, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE };
        ComPtr<ID3DBlob> sig;
        CHECK_HR(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, nullptr), "SerSig");
        CHECK_HR(m_device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                               IID_PPV_ARGS(&m_rootSig)), "RootSig");
    }

    uint64_t CalcMinTrafficBytes(const VariantSpec& v) const {
        uint64_t elemBytes = v.useFp16 ? 2 : 4;
        uint64_t bytes = 0;
        for (const auto& l : m_layers) {
            if (l.type == 2) {
                bytes += ((uint64_t)12 * m_inW * m_inH + (uint64_t)3 * m_inW * m_inH
                        + (uint64_t)3 * m_outW * m_outH) * elemBytes;
            } else {
                bytes += ((uint64_t)l.inCh * m_inW * m_inH + (uint64_t)l.outCh * m_inW * m_inH) * elemBytes;
            }
        }
        bytes += v.useDense ? (v.useFp16 ? m_dense16Bytes : m_dense32Bytes)
                            : (m_ternaryBytes + (v.useFp16 ? m_dense16Bytes : m_dense32Bytes) / 4);
        return bytes;
    }

    void RunVariant(const VariantSpec& v, int warmups, int iters) {
        auto cso = ReadAllBytes(v.csoFile);
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = m_rootSig.Get();
        pd.CS = { cso.data(), cso.size() };
        ComPtr<ID3D12PipelineState> pso;
        CHECK_HR(m_device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)), "PSO " + v.name);

        uint64_t freq = 0;
        CHECK_HR(m_queue->GetTimestampFrequency(&freq), "TS Freq");
        uint32_t tsCount = (uint32_t)m_layers.size() + 1;

        std::vector<double> frameMs;
        std::vector<double> layerSumMs(m_layers.size(), 0.0);
        ID3D12Resource* lastOut = nullptr;
        D3D12_RESOURCE_STATES* lastOutState = nullptr;

        for (int it = 0; it < warmups + iters; ++it) {
            bool timed = (it >= warmups);
            CHECK_HR(m_alloc->Reset(), "Alloc");
            CHECK_HR(m_cmd->Reset(m_alloc.Get(), pso.Get()), "Cmd");
            m_cmd->SetComputeRootSignature(m_rootSig.Get());

            ID3D12Resource* origLr = v.useFp16 ? m_origLrFp16.Get() : m_origLrFp32.Get();
            ID3D12Resource* denseW = v.useFp16 ? m_denseFp16Buf.Get() : m_denseFp32Buf.Get();
            m_cmd->SetComputeRootShaderResourceView(2, m_ternaryBuf->GetGPUVirtualAddress());
            m_cmd->SetComputeRootShaderResourceView(3, denseW->GetGPUVirtualAddress());
            m_cmd->SetComputeRootShaderResourceView(4, m_scaleBuf->GetGPUVirtualAddress());
            m_cmd->SetComputeRootShaderResourceView(5, m_biasBuf->GetGPUVirtualAddress());
            m_cmd->SetComputeRootShaderResourceView(6, origLr->GetGPUVirtualAddress());

            ID3D12Resource* currIn = origLr;
            D3D12_RESOURCE_STATES* currInState = nullptr;
            bool writeToA = true;

            if (timed) m_cmd->EndQuery(m_tsHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);

            for (size_t i = 0; i < m_layers.size(); ++i) {
                const auto& l = m_layers[i];
                ID3D12Resource* currOut = writeToA ? m_pingA.Get() : m_pingB.Get();
                D3D12_RESOURCE_STATES& currOutState = writeToA ? m_stateA : m_stateB;

                if (currInState)
                    Transition(currIn, *currInState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                Transition(currOut, currOutState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

                GpuLayerConfig cfg = {};
                cfg.InW = m_inW; cfg.InH = m_inH;
                cfg.OutW = (l.type == 2) ? m_outW : m_inW;
                cfg.OutH = (l.type == 2) ? m_outH : m_inH;
                cfg.LayerType = l.type;
                cfg.InChannels = l.inCh;
                cfg.OutChannels = l.outCh;
                cfg.Activation = l.activation;
                cfg.TernaryOffset = l.ternaryOff;
                cfg.DenseOffset = l.denseOff;
                cfg.ScaleOffset = l.scaleOff;
                cfg.BiasOffset = l.biasOff;

                m_cmd->SetComputeRoot32BitConstants(0, sizeof(GpuLayerConfig) / 4, &cfg, 0);
                m_cmd->SetComputeRootShaderResourceView(1, currIn->GetGPUVirtualAddress());
                m_cmd->SetComputeRootUnorderedAccessView(7, currOut->GetGPUVirtualAddress());
                if (l.type == 2) {
                    // PixelShuffle uses 1 thread per pixel/channel
                    uint32_t psTile = (v.tileY == 16) ? 16 : 8;
                    m_cmd->Dispatch((cfg.OutW + psTile - 1) / psTile,
                                    (cfg.OutH + psTile - 1) / psTile,
                                    cfg.OutChannels);
                } else {
                    m_cmd->Dispatch((cfg.OutW + v.tileX - 1) / v.tileX,
                                    (cfg.OutH + v.tileY - 1) / v.tileY,
                                    (cfg.OutChannels + v.ocTile - 1) / v.ocTile);
                }

                if (timed) m_cmd->EndQuery(m_tsHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, (UINT)(i + 1));

                currIn = currOut;
                currInState = &currOutState;
                writeToA = !writeToA;
            }

            lastOut = currIn;
            lastOutState = currInState;

            if (timed) {
                m_cmd->ResolveQueryData(m_tsHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                                        0, tsCount, m_tsReadback.Get(), 0);
            }

            CHECK_HR(m_cmd->Close(), "Close");
            ID3D12CommandList* lists[] = { m_cmd.Get() };
            m_queue->ExecuteCommandLists(1, lists);
            WaitGPU();

            if (timed) {
                uint64_t* ts = nullptr;
                CHECK_HR(m_tsReadback->Map(0, nullptr, (void**)&ts), "TS Map");
                frameMs.push_back((double)(ts[tsCount - 1] - ts[0]) * 1000.0 / (double)freq);
                for (size_t i = 0; i < m_layers.size(); ++i)
                    layerSumMs[i] += (double)(ts[i + 1] - ts[i]) * 1000.0 / (double)freq;
                m_tsReadback->Unmap(0, nullptr);
            }
        }

        // Readback & numerical parity check against PyTorch oracle
        uint64_t nElems = (uint64_t)3 * m_outW * m_outH;
        uint64_t rbBytes = nElems * (v.useFp16 ? 2 : 4);
        CHECK_HR(m_alloc->Reset(), "RB Alloc");
        CHECK_HR(m_cmd->Reset(m_alloc.Get(), nullptr), "RB Cmd");
        Transition(lastOut, *lastOutState, D3D12_RESOURCE_STATE_COPY_SOURCE);
        m_cmd->CopyBufferRegion(m_readbackBuf.Get(), 0, lastOut, 0, rbBytes);
        CHECK_HR(m_cmd->Close(), "RB Close");
        ID3D12CommandList* lists[] = { m_cmd.Get() };
        m_queue->ExecuteCommandLists(1, lists);
        WaitGPU();

        void* mapped = nullptr;
        CHECK_HR(m_readbackBuf->Map(0, nullptr, &mapped), "RB Map");
        double maxErr = 0.0;
        if (v.useFp16) {
            const uint16_t* gpuH = reinterpret_cast<const uint16_t*>(mapped);
            for (size_t i = 0; i < nElems; ++i)
                maxErr = std::max(maxErr, std::abs((double)HalfToFloat(gpuH[i]) - (double)m_expectedHr[i]));
        } else {
            const float* gpuF = reinterpret_cast<const float*>(mapped);
            for (size_t i = 0; i < nElems; ++i)
                maxErr = std::max(maxErr, std::abs((double)gpuF[i] - (double)m_expectedHr[i]));
        }
        m_readbackBuf->Unmap(0, nullptr);

        double tol = v.useFp16 ? 2e-2 : 1e-4;
        bool pass = (maxErr < tol);

        std::sort(frameMs.begin(), frameMs.end());
        double medMs = frameMs[frameMs.size() / 2];
        double p95Ms = frameMs[(size_t)(frameMs.size() * 0.95)];
        double effBw = ((double)CalcMinTrafficBytes(v) / (medMs / 1000.0)) / 1e9;

        std::cout << "| " << v.name
                  << " | " << (pass ? "PASS" : "FAIL")
                  << " | " << std::scientific << std::setprecision(2) << maxErr
                  << " | " << std::fixed << std::setprecision(3) << medMs
                  << " | " << p95Ms
                  << " | " << std::setprecision(1) << (1000.0 / medMs)
                  << " | " << std::setprecision(1) << effBw << " |\n";

        std::cerr << "  [Per-Layer Avg ms | " << v.name << "] ";
        for (size_t i = 0; i < m_layers.size(); ++i)
            std::cerr << "L" << i << "=" << std::fixed << std::setprecision(3)
                      << (layerSumMs[i] / iters) << "ms ";
        std::cerr << "\n";
    }
};

int main(int argc, char** argv) {
    try {
        std::string dir = (argc > 1) ? argv[1] : "outputs/bench_540p";
        RooflineBench bench;
        bench.RunAll(dir, 15, 50);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }
}

