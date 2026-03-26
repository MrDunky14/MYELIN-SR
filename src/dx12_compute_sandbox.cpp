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

/**
 * MYELIN-SR v2: Zero-Barrier Pipeline Engine
 *
 * PS5-style: compile shaders once, submit ALL dispatches in one shot,
 * let the GPU hardware scheduler pipeline everything.
 * ZERO UAV barriers between layers. ONE fence at the end.
 */

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <wrl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <cstdint>

using Microsoft::WRL::ComPtr;

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

#define CHECK_HR(hr, msg) if (FAILED(hr)) { std::cerr << "FATAL: " << msg << " (0x" << std::hex << hr << std::dec << ")" << std::endl; throw std::runtime_error(msg); }

static const char MAGIC[4] = { 'M', 'Y', 'L', 'N' };

struct LayerDesc {
    uint32_t type, in_ch, out_ch, kernel, stride, padding, groups;
    uint32_t w_offset, w_size, s_offset, b_offset;
};

// Matches HLSL cbuffer exactly (10 uint32 = 40 bytes)
struct GpuLayerConfig {
    uint32_t InputWidth;
    uint32_t InputHeight;
    uint32_t LayerType;
    uint32_t InChannels;
    uint32_t OutChannels;
    uint32_t Groups;
    uint32_t WeightOffset;
    uint32_t ScaleOffset;
    uint32_t BiasOffset;
    uint32_t Activation;
};

class MyelinEngine {
public:
    uint32_t m_inputW = 1280, m_inputH = 720;
    uint32_t m_outputW = 2560, m_outputH = 1440;
    
    void Initialize() {
        std::cout << "=== MYELIN-SR v2: Zero-Barrier Pipeline ===" << std::endl;
        
        ComPtr<IDXGIFactory4> factory;
        CHECK_HR(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "Factory");
        
        ComPtr<IDXGIAdapter1> adapter;
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 d;
            adapter->GetDesc1(&d);
            if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr))) {
                char name[128];
                WideCharToMultiByte(CP_UTF8, 0, d.Description, -1, name, 128, NULL, NULL);
                std::cout << "GPU: " << name << " (" << d.DedicatedVideoMemory/(1024*1024) << " MB)" << std::endl;
                break;
            }
        }
        
        CHECK_HR(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_device)), "Device");
        
        D3D12_COMMAND_QUEUE_DESC qd = {}; qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        CHECK_HR(m_device->CreateCommandQueue(&qd, IID_PPV_ARGS(&m_queue)), "Queue");
        CHECK_HR(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&m_alloc)), "Alloc");
        CHECK_HR(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, m_alloc.Get(), nullptr, IID_PPV_ARGS(&m_cmdList)), "CmdList");
        m_cmdList->Close();
        
        CHECK_HR(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)), "Fence");
        m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        m_fenceVal = 1;
        
        D3D12_DESCRIPTOR_HEAP_DESC hd = {};
        hd.NumDescriptors = 64;
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        CHECK_HR(m_device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&m_heap)), "Heap");
        
        std::cout << ">> Pipeline ready" << std::endl;
    }
    
    void LoadWeights(const std::string& path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f.is_open()) throw std::runtime_error("Cannot open " + path);
        
        size_t sz = f.tellg(); f.seekg(0);
        m_bin.resize(sz); f.read(m_bin.data(), sz);
        
        uint32_t numLayers = *(uint32_t*)(m_bin.data() + 8);
        m_layers.resize(numLayers);
        for (uint32_t i = 0; i < numLayers; i++)
            memcpy(&m_layers[i], m_bin.data() + 12 + i * sizeof(LayerDesc), sizeof(LayerDesc));
        
        uint64_t* footer = (uint64_t*)(m_bin.data() + sz - 32);
        uint64_t tOff = footer[0], sOff = footer[1], bOff = footer[2], fpOff = footer[3];
        
        auto MakeBuf = [&](ComPtr<ID3D12Resource>& r, const void* d, size_t s, const char* n) {
            if (!s) return;
            D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_UPLOAD;
            D3D12_RESOURCE_DESC rd = {};
            rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            rd.Width = s; rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
            rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            CHECK_HR(m_device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&r)), n);
            void* p; r->Map(0, nullptr, &p); memcpy(p, d, s); r->Unmap(0, nullptr);
            std::cout << "  [" << n << "] " << s/1024.0f << " KB" << std::endl;
        };
        
        std::cout << "Loading " << numLayers << " layers (" << sz/1024.0f << " KB)..." << std::endl;
        MakeBuf(m_wBuf, m_bin.data()+tOff, sOff-tOff, "Weights");
        MakeBuf(m_sBuf, m_bin.data()+sOff, bOff-sOff, "Scales");
        MakeBuf(m_bBuf, m_bin.data()+bOff, fpOff-bOff, "Biases");
    }
    
    void LoadCompiledShader(const std::string& csoPath) {
        std::ifstream file(csoPath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) throw std::runtime_error("Cannot open " + csoPath);
        
        size_t size = file.tellg();
        file.seekg(0);
        std::vector<char> shaderData(size);
        file.read(shaderData.data(), size);
        
        // Root sig: [0] = constants, [1] = descriptor table
        D3D12_DESCRIPTOR_RANGE ranges[2] = {};
        ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; ranges[0].NumDescriptors = 8;
        ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; ranges[1].NumDescriptors = 4;
        
        D3D12_ROOT_PARAMETER params[2] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants.Num32BitValues = sizeof(GpuLayerConfig) / 4;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[1].DescriptorTable.NumDescriptorRanges = 2;
        params[1].DescriptorTable.pDescriptorRanges = ranges;
        params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        
        D3D12_ROOT_SIGNATURE_DESC sd = {}; sd.NumParameters = 2; sd.pParameters = params;
        ComPtr<ID3DBlob> sig;
        CHECK_HR(D3D12SerializeRootSignature(&sd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, nullptr), "SerializeSig");
        CHECK_HR(m_device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&m_rootSig)), "RootSig");
        
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = m_rootSig.Get();
        pd.CS = { shaderData.data(), shaderData.size() };
        CHECK_HR(m_device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&m_pso)), "PSO");
        
        std::cout << ">> Shader loaded from CSO" << std::endl;
    }
    
    void Benchmark(int iterations = 20) {
        // Pre-build GPU configs for all layers
        std::vector<GpuLayerConfig> configs;
        for (auto& l : m_layers) {
            GpuLayerConfig c = {};
            c.InputWidth = m_inputW;
            c.InputHeight = m_inputH;
            c.InChannels = l.in_ch;
            c.OutChannels = l.out_ch;
            c.Groups = l.groups;
            c.WeightOffset = l.w_offset / 4;
            c.ScaleOffset = l.s_offset / 4;
            c.BiasOffset = l.b_offset / 4;
            
            if (l.type == 0 && l.kernel == 1) c.LayerType = 1;
            else if (l.type == 0) { c.LayerType = 0; c.Activation = 1; }
            else if (l.type == 3) c.LayerType = 3;
            else c.LayerType = 2;
            
            configs.push_back(c);
        }
        
        uint32_t gx = (m_inputW + 7) / 8;
        uint32_t gy = (m_inputH + 7) / 8;
        
        std::cout << "\nBenchmarking " << configs.size() << " layers, "
                  << iterations << " iterations, ZERO mid-frame barriers..." << std::endl;
        
        double totalMs = 0;
        for (int it = 0; it < iterations; it++) {
            CHECK_HR(m_alloc->Reset(), "AR");
            CHECK_HR(m_cmdList->Reset(m_alloc.Get(), m_pso.Get()), "CR");
            
            m_cmdList->SetComputeRootSignature(m_rootSig.Get());
            ID3D12DescriptorHeap* h[] = { m_heap.Get() };
            m_cmdList->SetDescriptorHeaps(1, h);
            
            // ═══════════════════════════════════════════
            // ZERO BARRIERS — Just dispatch everything.
            // Let the GPU pipeline all 55 layers freely.
            // ═══════════════════════════════════════════
            for (size_t i = 0; i < configs.size(); i++) {
                m_cmdList->SetComputeRoot32BitConstants(0, sizeof(GpuLayerConfig)/4, &configs[i], 0);
                m_cmdList->Dispatch(gx, gy, 1);
            }
            
            CHECK_HR(m_cmdList->Close(), "CL");
            
            auto t0 = std::chrono::high_resolution_clock::now();
            ID3D12CommandList* lists[] = { m_cmdList.Get() };
            m_queue->ExecuteCommandLists(1, lists);
            WaitGPU();
            auto t1 = std::chrono::high_resolution_clock::now();
            
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            totalMs += ms;
            if (it == 0) std::cout << "  Warmup: " << ms << " ms" << std::endl;
        }
        
        double avg = totalMs / iterations;
        double proj1080 = avg * 0.5625;  // 540p/720p pixel ratio
        
        std::cout << "\n===========================================" << std::endl;
        std::cout << " MYELIN-SR v2 ZERO-BARRIER BENCHMARK" << std::endl;
        std::cout << "===========================================" << std::endl;
        std::cout << " Resolution:     " << m_inputW << "x" << m_inputH << " => " << m_outputW << "x" << m_outputH << std::endl;
        std::cout << " Layers:         " << configs.size() << std::endl;
        std::cout << " Barriers:       0 (fully pipelined)" << std::endl;
        std::cout << " 720p->1440p:    " << avg << " ms (" << (int)(1000.0/avg) << " FPS)" << std::endl;
        std::cout << " 540p->1080p:    " << proj1080 << " ms (" << (int)(1000.0/proj1080) << " FPS) [projected]" << std::endl;
        std::cout << " Per Layer:      " << avg/configs.size() << " ms" << std::endl;
        std::cout << "===========================================" << std::endl;
        
        if (avg < 16.6) std::cout << ">> 60+ FPS at 720p->1440p!" << std::endl;
        else if (proj1080 < 16.6) std::cout << ">> 60+ FPS at 1080p (projected)!" << std::endl;
    }

private:
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12CommandQueue> m_queue;
    ComPtr<ID3D12CommandAllocator> m_alloc;
    ComPtr<ID3D12GraphicsCommandList> m_cmdList;
    ComPtr<ID3D12Fence> m_fence;
    HANDLE m_fenceEvent;
    uint64_t m_fenceVal;
    ComPtr<ID3D12DescriptorHeap> m_heap;
    ComPtr<ID3D12RootSignature> m_rootSig;
    ComPtr<ID3D12PipelineState> m_pso;
    
    std::vector<char> m_bin;
    std::vector<LayerDesc> m_layers;
    ComPtr<ID3D12Resource> m_wBuf, m_sBuf, m_bBuf;
    
    void WaitGPU() {
        uint64_t v = m_fenceVal++;
        m_queue->Signal(m_fence.Get(), v);
        if (m_fence->GetCompletedValue() < v) {
            m_fence->SetEventOnCompletion(v, m_fenceEvent);
            WaitForSingleObject(m_fenceEvent, INFINITE);
        }
    }
};

int main(int argc, char** argv) {
    try {
        std::string mode = "quality";
        if (argc > 1) {
            std::string arg = argv[1];
            if (arg == "performance" || arg == "quality") {
                mode = arg;
            } else {
                std::cout << "Usage: dx12_benchmark.exe [quality|performance]" << std::endl;
                return 0;
            }
        }
        
        std::string binPath = "outputs/myelin_engine_v2_" + mode + ".bin";
        std::cout << "Using preset: " << mode << " (" << binPath << ")" << std::endl;
        
        MyelinEngine e;
        e.Initialize();
        e.LoadWeights(binPath);
        e.LoadCompiledShader("src/myelin_compute.cso");
        e.Benchmark(20);
    } catch (std::exception& ex) {
        std::cerr << "FATAL: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
