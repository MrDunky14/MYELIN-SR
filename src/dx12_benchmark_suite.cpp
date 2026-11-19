#ifndef NOMINMAX
#define NOMINMAX
#endif
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

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

using Microsoft::WRL::ComPtr;

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

#define CHECK_HR(hr, msg)                                                      \
    if (FAILED(hr)) {                                                          \
        throw std::runtime_error(std::string(msg) + " failed (HR=0x" +         \
                                 std::to_string((uint32_t)(hr)) + ")");        \
    }

struct LayerDescV3 {
    uint32_t type, inCh, outCh, activation;
    uint32_t wOffset, sOffset, bOffset, reserved;
};

struct GpuLayerConfig {
    uint32_t InW, InH, OutW, OutH;
    uint32_t LayerType, InChannels, OutChannels, Activation;
    uint32_t WeightOffset, ScaleOffset, BiasOffset, Reserved;
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

class BenchmarkEngine {
public:
    void Run(const std::string& dir, const std::string& csoPath, int warmups = 10, int iters = 50) {
        InitDevice();
        LoadBundle(dir + "/myelin_v3.bin", dir + "/input_lr.bin");
        BuildPipeline(csoPath);

        std::cout << "\n=======================================================\n";
        std::cout << " MYELIN-SR v3 Roofline Benchmark Suite (Release)\n";
        std::cout << " Target: " << m_inW << "x" << m_inH << " -> " << m_outW << "x" << m_outH << "\n";
        std::cout << " Shader: " << csoPath << "\n";
        std::cout << "=======================================================\n";

        // Execute warmup passes
        for (int i = 0; i < warmups; ++i) {
            ExecutePass(false, 0);
        }

        // Measure timed iterations using GPU Timestamps
        std::vector<double> gpuTimes;
        gpuTimes.reserve(iters);

        uint64_t freq = 0;
        CHECK_HR(m_queue->GetTimestampFrequency(&freq), "GetTimestampFrequency");

        for (int i = 0; i < iters; ++i) {
            ExecutePass(true, 0);

            uint64_t* ts = nullptr;
            CHECK_HR(m_tsReadback->Map(0, nullptr, (void**)&ts), "TS Map");
            double ms = (double)(ts[1] - ts[0]) * 1000.0 / (double)freq;
            m_tsReadback->Unmap(0, nullptr);
            gpuTimes.push_back(ms);
        }

        std::sort(gpuTimes.begin(), gpuTimes.end());
        double minMs = gpuTimes.front();
        double medMs = gpuTimes[gpuTimes.size() / 2];
        double sum = 0.0;
        for (double d : gpuTimes) sum += d;
        double avgMs = sum / (double)gpuTimes.size();

        // Calculate tensor data movement for rough memory bandwidth estimation
        // Activation memory read + written per pass
        double totalBytes = 0.0;
        for (const auto& l : m_layers) {
            double inBytes = (double)l.inCh * m_inW * m_inH * 4.0;
            double outBytes = (double)l.outCh * ((l.type == 2) ? m_outW : m_inW) * ((l.type == 2) ? m_outH : m_inH) * 4.0;
            totalBytes += (inBytes + outBytes);
        }
        double estBandwidth = (totalBytes / (1024.0 * 1024.0 * 1024.0)) / (avgMs / 1000.0);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << " Iterations:    " << iters << " (after " << warmups << " warmups)\n";
        std::cout << " Min Frame:     " << minMs << " ms (" << (1000.0 / minMs) << " FPS)\n";
        std::cout << " Median Frame:  " << medMs << " ms (" << (1000.0 / medMs) << " FPS)\n";
        std::cout << " Avg Frame:     " << avgMs << " ms (" << (1000.0 / avgMs) << " FPS)\n";
        std::cout << " Bandwidth:     " << estBandwidth << " GB/s (effective activation IO)\n";
        std::cout << "=======================================================\n";

        // Check if golden reference exists for parity evaluation
        std::string refPath = dir + "/expected_hr.bin";
        std::ifstream refFile(refPath, std::ios::binary);
        if (refFile.is_open()) {
            refFile.close();
            auto gpuOut = ReadbackOutput();
            VerifyAndSave(gpuOut, refPath, dir + "/gpu_output.png");
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

    ComPtr<ID3D12RootSignature> m_rootSig;
    ComPtr<ID3D12PipelineState> m_pso;

    ComPtr<ID3D12Resource> m_ternaryBuf, m_fpBuf, m_scaleBuf, m_biasBuf, m_origLrBuf;
    ComPtr<ID3D12Resource> m_pingA, m_pingB, m_readbackBuf;
    D3D12_RESOURCE_STATES m_stateA = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES m_stateB = D3D12_RESOURCE_STATE_COMMON;
    ID3D12Resource* m_finalOutRes = nullptr;
    D3D12_RESOURCE_STATES* m_finalOutState = nullptr;

    ComPtr<ID3D12QueryHeap> m_tsHeap;
    ComPtr<ID3D12Resource> m_tsReadback;

    uint32_t m_inW = 0, m_inH = 0, m_outW = 0, m_outH = 0;
    std::vector<LayerDescV3> m_layers;

    void WaitGPU() {
        uint64_t v = m_fenceVal++;
        CHECK_HR(m_queue->Signal(m_fence.Get(), v), "Fence Signal");
        if (m_fence->GetCompletedValue() < v) {
            CHECK_HR(m_fence->SetEventOnCompletion(v, m_fenceEvent), "Fence Event");
            WaitForSingleObject(m_fenceEvent, INFINITE);
        }
    }

    ComPtr<ID3D12Resource> CreateBuffer(uint64_t bytes, D3D12_HEAP_TYPE heapType,
                                        D3D12_RESOURCE_FLAGS flags,
                                        D3D12_RESOURCE_STATES initState) {
        if (bytes == 0) bytes = 4;
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = heapType;
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = bytes;
        rd.Height = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags = flags;
        ComPtr<ID3D12Resource> res;
        CHECK_HR(m_device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                                   initState, nullptr, IID_PPV_ARGS(&res)),
                 "CreateCommittedResource");
        return res;
    }

    void Transition(ID3D12Resource* res, D3D12_RESOURCE_STATES& curr, D3D12_RESOURCE_STATES next) {
        if (curr == next) return;
        D3D12_RESOURCE_BARRIER b = {};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = res;
        b.Transition.StateBefore = curr;
        b.Transition.StateAfter = next;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        m_cmd->ResourceBarrier(1, &b);
        curr = next;
    }

    ComPtr<ID3D12Resource> UploadToDefault(const void* data, uint64_t bytes) {
        uint64_t allocBytes = (bytes == 0) ? 4 : bytes;
        auto staging = CreateBuffer(allocBytes, D3D12_HEAP_TYPE_UPLOAD,
                                    D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        void* mapped = nullptr;
        CHECK_HR(staging->Map(0, nullptr, &mapped), "Staging Map");
        if (bytes > 0) std::memcpy(mapped, data, bytes);
        else std::memset(mapped, 0, 4);
        staging->Unmap(0, nullptr);

        auto gpuBuf = CreateBuffer(allocBytes, D3D12_HEAP_TYPE_DEFAULT,
                                   D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

        CHECK_HR(m_alloc->Reset(), "Upload Alloc Reset");
        CHECK_HR(m_cmd->Reset(m_alloc.Get(), nullptr), "Upload Cmd Reset");
        m_cmd->CopyBufferRegion(gpuBuf.Get(), 0, staging.Get(), 0, allocBytes);
        D3D12_RESOURCE_STATES st = D3D12_RESOURCE_STATE_COPY_DEST;
        Transition(gpuBuf.Get(), st, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        CHECK_HR(m_cmd->Close(), "Upload Cmd Close");

        ID3D12CommandList* lists[] = { m_cmd.Get() };
        m_queue->ExecuteCommandLists(1, lists);
        WaitGPU();
        return gpuBuf;
    }

    void InitDevice() {
        ComPtr<IDXGIFactory4> factory;
        CHECK_HR(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");

        ComPtr<IDXGIAdapter1> bestAdapter;
        SIZE_T maxVram = 0;
        DXGI_ADAPTER_DESC1 bestDesc = {};

        ComPtr<IDXGIAdapter1> adapter;
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc;
            adapter->GetDesc1(&desc);
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, _uuidof(ID3D12Device), nullptr))) {
                if (desc.DedicatedVideoMemory > maxVram || !bestAdapter) {
                    maxVram = desc.DedicatedVideoMemory;
                    bestAdapter = adapter;
                    bestDesc = desc;
                }
            }
        }
        if (!bestAdapter) throw std::runtime_error("No hardware D3D12 adapter found");
        CHECK_HR(D3D12CreateDevice(bestAdapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&m_device)), "CreateDevice");
        wprintf(L"[DX12] Benchmark GPU: %s (%u MB VRAM)\n", bestDesc.Description,
                (UINT)(bestDesc.DedicatedVideoMemory / (1024 * 1024)));

        D3D12_COMMAND_QUEUE_DESC qd = {};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        CHECK_HR(m_device->CreateCommandQueue(&qd, IID_PPV_ARGS(&m_queue)), "CreateCommandQueue");
        CHECK_HR(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_alloc)), "CreateCommandAllocator");
        CHECK_HR(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_alloc.Get(), nullptr, IID_PPV_ARGS(&m_cmd)), "CreateCommandList");
        CHECK_HR(m_cmd->Close(), "Initial Cmd Close");

        CHECK_HR(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)), "CreateFence");
        m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

        D3D12_QUERY_HEAP_DESC qhd = {};
        qhd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        qhd.Count = 2;
        CHECK_HR(m_device->CreateQueryHeap(&qhd, IID_PPV_ARGS(&m_tsHeap)), "CreateQueryHeap");
        m_tsReadback = CreateBuffer(16, D3D12_HEAP_TYPE_READBACK,
                                    D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    }

    void LoadBundle(const std::string& binPath, const std::string& lrPath) {
        auto bin = ReadAllBytes(binPath);
        if (bin.size() < 20 || std::memcmp(bin.data(), "MYL3", 4) != 0)
            throw std::runtime_error("Invalid MYL3 binary header");

        const char* p = bin.data() + 4;
        uint32_t version, numLayers;
        std::memcpy(&version, p, 4); p += 4;
        std::memcpy(&numLayers, p, 4); p += 4;
        std::memcpy(&m_inW, p, 4); p += 4;
        std::memcpy(&m_inH, p, 4); p += 4;
        m_outW = m_inW * 2;
        m_outH = m_inH * 2;

        m_layers.resize(numLayers);
        std::memcpy(m_layers.data(), p, numLayers * sizeof(LayerDescV3));
        p += numLayers * sizeof(LayerDescV3);

        uint64_t sizes[4];
        std::memcpy(sizes, p, sizeof(sizes)); p += sizeof(sizes);

        m_ternaryBuf = UploadToDefault(p, sizes[0]); p += sizes[0];
        m_fpBuf      = UploadToDefault(p, sizes[1]); p += sizes[1];
        m_scaleBuf   = UploadToDefault(p, sizes[2]); p += sizes[2];
        m_biasBuf    = UploadToDefault(p, sizes[3]); p += sizes[3];

        auto lrBytes = ReadAllBytes(lrPath);
        m_origLrBuf = UploadToDefault(lrBytes.data(), lrBytes.size());

        uint64_t maxElems = (uint64_t)3 * m_outW * m_outH;
        for (const auto& l : m_layers) {
            uint64_t elems = (uint64_t)l.outCh * m_inW * m_inH;
            maxElems = std::max(maxElems, elems);
        }
        uint64_t pingBytes = maxElems * sizeof(float);
        m_pingA = CreateBuffer(pingBytes, D3D12_HEAP_TYPE_DEFAULT,
                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        m_pingB = CreateBuffer(pingBytes, D3D12_HEAP_TYPE_DEFAULT,
                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        m_readbackBuf = CreateBuffer((uint64_t)3 * m_outW * m_outH * sizeof(float),
                                     D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                                     D3D12_RESOURCE_STATE_COPY_DEST);
    }

    void BuildPipeline(const std::string& csoPath) {
        auto cso = ReadAllBytes(csoPath);

        D3D12_ROOT_PARAMETER params[8] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = { 0, 0, sizeof(GpuLayerConfig) / 4 };

        for (UINT i = 0; i < 6; ++i) {
            params[1 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            params[1 + i].Descriptor = { i, 0 };
        }
        params[7].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[7].Descriptor = { 0, 0 };

        D3D12_ROOT_SIGNATURE_DESC rsd = {};
        rsd.NumParameters = 8;
        rsd.pParameters = params;

        ComPtr<ID3DBlob> sig, err;
        HRESULT hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err);
        if (FAILED(hr)) {
            if (err) std::cerr << (const char*)err->GetBufferPointer() << "\n";
            CHECK_HR(hr, "SerializeRootSignature");
        }
        CHECK_HR(m_device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                               IID_PPV_ARGS(&m_rootSig)), "CreateRootSignature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = m_rootSig.Get();
        pd.CS = { cso.data(), cso.size() };
        CHECK_HR(m_device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&m_pso)), "CreateComputePipelineState");
    }

    void ExecutePass(bool timed, int queryIndex) {
        CHECK_HR(m_alloc->Reset(), "Alloc Reset");
        CHECK_HR(m_cmd->Reset(m_alloc.Get(), m_pso.Get()), "Cmd Reset");

        m_cmd->SetComputeRootSignature(m_rootSig.Get());

        m_cmd->SetComputeRootShaderResourceView(2, m_ternaryBuf->GetGPUVirtualAddress());
        m_cmd->SetComputeRootShaderResourceView(3, m_fpBuf->GetGPUVirtualAddress());
        m_cmd->SetComputeRootShaderResourceView(4, m_scaleBuf->GetGPUVirtualAddress());
        m_cmd->SetComputeRootShaderResourceView(5, m_biasBuf->GetGPUVirtualAddress());
        m_cmd->SetComputeRootShaderResourceView(6, m_origLrBuf->GetGPUVirtualAddress());

        if (timed) m_cmd->EndQuery(m_tsHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, queryIndex);

        ID3D12Resource* currIn = m_origLrBuf.Get();
        D3D12_RESOURCE_STATES* currInState = nullptr;
        bool writeToA = true;

        for (size_t i = 0; i < m_layers.size(); ++i) {
            const auto& l = m_layers[i];
            ID3D12Resource* currOut = writeToA ? m_pingA.Get() : m_pingB.Get();
            D3D12_RESOURCE_STATES& currOutState = writeToA ? m_stateA : m_stateB;

            if (currInState)
                Transition(currIn, *currInState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            Transition(currOut, currOutState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

            GpuLayerConfig cfg = {};
            cfg.InW = m_inW;
            cfg.InH = m_inH;
            cfg.OutW = (l.type == 2) ? m_outW : m_inW;
            cfg.OutH = (l.type == 2) ? m_outH : m_inH;
            cfg.LayerType = l.type;
            cfg.InChannels = l.inCh;
            cfg.OutChannels = l.outCh;
            cfg.Activation = l.activation;
            cfg.WeightOffset = l.wOffset;
            cfg.ScaleOffset = l.sOffset;
            cfg.BiasOffset = l.bOffset;

            m_cmd->SetComputeRoot32BitConstants(0, sizeof(GpuLayerConfig) / 4, &cfg, 0);
            m_cmd->SetComputeRootShaderResourceView(1, currIn->GetGPUVirtualAddress());
            m_cmd->SetComputeRootUnorderedAccessView(7, currOut->GetGPUVirtualAddress());

            m_cmd->Dispatch((cfg.OutW + 7) / 8, (cfg.OutH + 7) / 8, cfg.OutChannels);

            currIn = currOut;
            currInState = &currOutState;
            writeToA = !writeToA;
        }

        m_finalOutRes = currIn;
        m_finalOutState = currInState;

        if (timed) {
            m_cmd->EndQuery(m_tsHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, queryIndex + 1);
            m_cmd->ResolveQueryData(m_tsHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, queryIndex, 2,
                                    m_tsReadback.Get(), 0);
        }

        CHECK_HR(m_cmd->Close(), "Cmd Close");
        ID3D12CommandList* lists[] = { m_cmd.Get() };
        m_queue->ExecuteCommandLists(1, lists);
        WaitGPU();
    }

    std::vector<float> ReadbackOutput() {
        uint64_t numFloats = (uint64_t)3 * m_outW * m_outH;
        uint64_t bytes = numFloats * sizeof(float);

        CHECK_HR(m_alloc->Reset(), "RB Alloc Reset");
        CHECK_HR(m_cmd->Reset(m_alloc.Get(), nullptr), "RB Cmd Reset");
        Transition(m_finalOutRes, *m_finalOutState, D3D12_RESOURCE_STATE_COPY_SOURCE);
        m_cmd->CopyBufferRegion(m_readbackBuf.Get(), 0, m_finalOutRes, 0, bytes);
        CHECK_HR(m_cmd->Close(), "RB Cmd Close");

        ID3D12CommandList* lists[] = { m_cmd.Get() };
        m_queue->ExecuteCommandLists(1, lists);
        WaitGPU();

        std::vector<float> out(numFloats);
        void* mapped = nullptr;
        CHECK_HR(m_readbackBuf->Map(0, nullptr, &mapped), "RB Map");
        std::memcpy(out.data(), mapped, bytes);
        m_readbackBuf->Unmap(0, nullptr);
        return out;
    }

    void VerifyAndSave(const std::vector<float>& gpuOut, const std::string& refPath,
                       const std::string& pngPath) {
        auto refBytes = ReadAllBytes(refPath);
        if (refBytes.size() != gpuOut.size() * sizeof(float))
            return;

        const float* ref = reinterpret_cast<const float*>(refBytes.data());
        double maxErr = 0.0, sumSq = 0.0;
        for (size_t i = 0; i < gpuOut.size(); ++i) {
            double d = std::abs((double)gpuOut[i] - (double)ref[i]);
            maxErr = std::max(maxErr, d);
            sumSq += d * d;
        }
        double rmse = std::sqrt(sumSq / (double)gpuOut.size());

        std::cout << "[Parity] Max Abs Error: " << maxErr << " | RMSE: " << rmse << "\n";
        std::cout << "[Parity] Status: " << (maxErr < 1e-4 ? "PASS" : "FAIL") << "\n";

        std::vector<uint8_t> img(m_outW * m_outH * 3);
        for (uint32_t y = 0; y < m_outH; ++y) {
            for (uint32_t x = 0; x < m_outW; ++x) {
                for (uint32_t c = 0; c < 3; ++c) {
                    float v = gpuOut[(c * m_outH + y) * m_outW + x];
                    v = std::clamp(v, 0.0f, 1.0f);
                    img[(y * m_outW + x) * 3 + c] = (uint8_t)(v * 255.0f + 0.5f);
                }
            }
        }
        stbi_write_png(pngPath.c_str(), (int)m_outW, (int)m_outH, 3, img.data(), (int)(m_outW * 3));
        std::cout << "[Output] Saved " << pngPath << "\n";
    }
};

int main(int argc, char** argv) {
    try {
        std::string bundleDir = (argc > 1) ? argv[1] : "outputs/bench_540p";
        std::string csoPath = (argc > 2) ? argv[2] : "outputs/myelin_v3_tiled.cso";
        int warmups = (argc > 3) ? std::stoi(argv[3]) : 10;
        int iters = (argc > 4) ? std::stoi(argv[4]) : 50;

        BenchmarkEngine engine;
        engine.Run(bundleDir, csoPath, warmups, iters);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }
}
