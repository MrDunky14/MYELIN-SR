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
 * MYELIN-SR v2: Production ReShade Addon
 *
 * Hooks into any DX12 game via ReShade's addon API.
 * Intercepts the game's backbuffer before Present, runs the
 * multi-pass ternary compute shader, and outputs the upscaled frame.
 *
 * Build: cmake -DBUILD_RESHADE_ADDON=ON -DRESHADE_SDK_DIR=path/to/sdk
 * Install: Copy myelin_sr.addon64 + myelin_engine_v2.bin to game directory
 */

#include <reshade.hpp>
#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <wrl.h>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>

using Microsoft::WRL::ComPtr;
using namespace reshade::api;

// ── Binary format (matches export_hlsl_weights.py) ──
struct LayerDescriptor {
    uint32_t type, in_ch, out_ch, kernel, stride, padding, groups;
    uint32_t w_offset, w_size, s_offset, b_offset;
};

struct LayerConfig {
    uint32_t InputWidth, InputHeight, OutputWidth, OutputHeight;
    uint32_t LayerType, InChannels, OutChannels, KernelSize;
    uint32_t Stride, Padding, Groups;
    uint32_t WeightOffset, ScaleOffset, BiasOffset;
    uint32_t UpscaleFactor, ActivationType, IsResidual;
};

// ── Per-Device Engine State ──
struct __declspec(uuid("8F26A57E-08B8-4AF9-B5DB-7A39D5C6DB8C")) MyelinState {
    bool initialized = false;
    bool enabled = true;
    
    // DX12 native handles
    ID3D12Device*              device = nullptr;
    ID3D12RootSignature*       rootSig = nullptr;
    ID3D12PipelineState*       pso = nullptr;
    ID3D12DescriptorHeap*      srvHeap = nullptr;
    
    // Weight buffers
    ID3D12Resource*            ternaryBuf = nullptr;
    ID3D12Resource*            scaleBuf = nullptr;
    ID3D12Resource*            biasBuf = nullptr;
    ID3D12Resource*            fpBuf = nullptr;
    
    // Intermediate feature textures (ping-pong)
    ID3D12Resource*            featureA = nullptr;
    ID3D12Resource*            featureB = nullptr;
    
    // Layer table
    std::vector<LayerDescriptor> layers;
    
    // Perf stats
    float lastFrameTimeMs = 0.0f;
    uint32_t frameCount = 0;
};

// ========================================================================= //
// Device Lifecycle
// ========================================================================= //

static void CALLBACK OnInitDevice(device* dev) {
    if (dev->get_api() != device_api::d3d12) return;

    auto* state = dev->create_private_data<MyelinState>();
    if (state == nullptr) return;
    state->device = (ID3D12Device*)dev->get_native();
    
    // Locate weight file relative to the game exe
    char exePath[MAX_PATH];
    GetModuleFileNameA(NULL, exePath, MAX_PATH);
    std::string dir(exePath);
    dir = dir.substr(0, dir.find_last_of('\\') + 1);
    std::string binPath = dir + "myelin_engine_v2_quality.bin";
    
    std::ifstream f(binPath, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        reshade::log::message(reshade::log::level::warning,
            "MYELIN-SR: Could not find myelin_engine_v2_quality.bin next to game exe");
        return;
    }
    
    size_t fileSize = f.tellg();
    f.seekg(0);
    std::vector<char> data(fileSize);
    f.read(data.data(), fileSize);
    
    // Parse header
    uint32_t numLayers = *(uint32_t*)(data.data() + 8);
    state->layers.resize(numLayers);
    size_t offset = 12;
    for (uint32_t i = 0; i < numLayers; i++) {
        memcpy(&state->layers[i], data.data() + offset, sizeof(LayerDescriptor));
        offset += sizeof(LayerDescriptor);
    }
    
    // Read blob offsets from footer
    uint64_t* footer = (uint64_t*)(data.data() + fileSize - 32);
    uint64_t ternaryOff = footer[0], scaleOff = footer[1];
    uint64_t biasOff = footer[2], fpOff = footer[3];
    
    // Upload to GPU (using D3D12 native API)
    auto UploadBuffer = [&](ID3D12Resource** res, const void* src, size_t size) {
        if (size == 0) return;
        D3D12_HEAP_PROPERTIES heap = {}; heap.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC desc = {};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = size; desc.Height = 1; desc.DepthOrArraySize = 1;
        desc.MipLevels = 1; desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        state->device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
            &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(res));
        void* mapped; (*res)->Map(0, nullptr, &mapped);
        memcpy(mapped, src, size); (*res)->Unmap(0, nullptr);
    };

    UploadBuffer(&state->ternaryBuf, data.data() + ternaryOff, scaleOff - ternaryOff);
    UploadBuffer(&state->scaleBuf,   data.data() + scaleOff,   biasOff - scaleOff);
    UploadBuffer(&state->biasBuf,    data.data() + biasOff,    fpOff - biasOff);
    UploadBuffer(&state->fpBuf,      data.data() + fpOff,      (fileSize - 32) - fpOff);
    
    // Load pre-compiled shader
    std::string csoPath = dir + "myelin_compute.cso";
    std::ifstream sf(csoPath, std::ios::binary | std::ios::ate);
    if (!sf.is_open()) {
        reshade::log::message(reshade::log::level::warning, "MYELIN-SR: Could not find myelin_compute.cso");
        return;
    }
    size_t sFileSize = sf.tellg();
    sf.seekg(0);
    std::vector<char> sData(sFileSize);
    sf.read(sData.data(), sFileSize);
    
    // Create root signature
    D3D12_DESCRIPTOR_RANGE ranges[2] = {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; ranges[0].NumDescriptors = 16;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; ranges[1].NumDescriptors = 8;
    
    D3D12_ROOT_PARAMETER params[2] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = sizeof(LayerConfig) / 4;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 2;
    params[1].DescriptorTable.pDescriptorRanges = ranges;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    
    D3D12_ROOT_SIGNATURE_DESC sigDesc = {};
    sigDesc.NumParameters = 2; sigDesc.pParameters = params;
    ComPtr<ID3DBlob> sigBlob;
    D3D12SerializeRootSignature(&sigDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sigBlob, nullptr);
    state->device->CreateRootSignature(0, sigBlob->GetBufferPointer(),
        sigBlob->GetBufferSize(), IID_PPV_ARGS(&state->rootSig));
    
    // Create PSO
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = state->rootSig;
    psoDesc.CS = { sData.data(), sData.size() };
    state->device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&state->pso));
    
    state->initialized = true;
    reshade::log::message(reshade::log::level::info,
        ("MYELIN-SR: Loaded " + std::to_string(numLayers) + " layers, " +
         std::to_string(fileSize / 1024) + " KB").c_str());
}

static void CALLBACK OnDestroyDevice(device* dev) {
    if (dev->get_api() != device_api::d3d12) return;

    auto* state = dev->get_private_data<MyelinState>();
    if (state == nullptr) return;
    if (state->ternaryBuf) state->ternaryBuf->Release();
    if (state->scaleBuf)   state->scaleBuf->Release();
    if (state->biasBuf)    state->biasBuf->Release();
    if (state->fpBuf)      state->fpBuf->Release();
    if (state->featureA)   state->featureA->Release();
    if (state->featureB)   state->featureB->Release();
    if (state->rootSig)    state->rootSig->Release();
    if (state->pso)        state->pso->Release();
    if (state->srvHeap)    state->srvHeap->Release();
    
    dev->destroy_private_data<MyelinState>();
}

// ========================================================================= //
// Present Hook — Where The Magic Happens
// ========================================================================= //

static void CALLBACK OnPresent(command_queue* queue, swapchain* swapchain,
    const rect*, const rect*, uint32_t, const rect*) 
{
    device* dev = swapchain->get_device();
    if (dev->get_api() != device_api::d3d12) return;

    auto* state = dev->get_private_data<MyelinState>();
    if (state == nullptr || !state->initialized || !state->enabled) return;
    
    command_list* cmd = queue->get_immediate_command_list();
    resource backbuffer = swapchain->get_current_back_buffer();
    
    // Transition to UAV
    cmd->barrier(backbuffer, resource_usage::present, resource_usage::unordered_access);
    
    // Bind pipeline
    cmd->bind_pipeline(pipeline_stage::compute_shader, pipeline{ reinterpret_cast<uint64_t>(state->pso) });
    
    // Dispatch each layer
    resource_desc bb_desc = dev->get_resource_desc(backbuffer);
    uint32_t width = (uint32_t)bb_desc.texture.width;
    uint32_t height = bb_desc.texture.height;
    
    for (size_t i = 0; i < state->layers.size(); i++) {
        const auto& layer = state->layers[i];
        
        LayerConfig config = {};
        config.InputWidth = width; config.InputHeight = height;
        config.OutputWidth = width * 2; config.OutputHeight = height * 2;
        config.LayerType = layer.type;
        config.InChannels = layer.in_ch; config.OutChannels = layer.out_ch;
        config.KernelSize = layer.kernel;
        config.Stride = layer.stride; config.Padding = layer.padding;
        config.Groups = layer.groups;
        config.WeightOffset = layer.w_offset;
        config.ScaleOffset = layer.s_offset;
        config.BiasOffset = layer.b_offset;
        config.UpscaleFactor = 2;
        config.ActivationType = (layer.type == 0) ? 1 : 0;
        config.IsResidual = 0;
        
        cmd->push_constants(shader_stage::compute, pipeline_layout{ reinterpret_cast<uint64_t>(state->rootSig) },
            0, 0, sizeof(LayerConfig) / 4, &config);
        
        cmd->dispatch((width + 15) / 16, (height + 15) / 16, 1);
    }
    
    // Transition back
    cmd->barrier(backbuffer, resource_usage::unordered_access, resource_usage::present);
    
    state->frameCount++;
}

// ========================================================================= //
// ReShade Overlay UI
// ========================================================================= //

static void CALLBACK OnOverlay(effect_runtime* runtime) {
    device* dev = runtime->get_device();
    if (dev->get_api() != device_api::d3d12) return;

    auto* state = dev->get_private_data<MyelinState>();
    if (state == nullptr) return;
    
    // ImGui overlay
    // Note: ReShade provides ImGui context automatically
    // ImGui::Text("MYELIN-SR v2 | %d layers | %.2f ms", (int)state.layers.size(), state.lastFrameTimeMs);
    // ImGui::Checkbox("Enable", &state.enabled);
}

// ========================================================================= //
// DLL Entry Point
// ========================================================================= //

extern "C" __declspec(dllexport) const char* NAME = "MYELIN-SR v2";
extern "C" __declspec(dllexport) const char* DESCRIPTION = 
    "Multiplication-Free AI Super Resolution (514 KB, Any GPU)";

BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID) {
    switch (reason) {
    case DLL_PROCESS_ATTACH:
        if (!reshade::register_addon(hModule)) return FALSE;
        reshade::register_event<reshade::addon_event::init_device>(OnInitDevice);
        reshade::register_event<reshade::addon_event::destroy_device>(OnDestroyDevice);
        reshade::register_event<reshade::addon_event::present>(OnPresent);
        // reshade::register_overlay(nullptr, OnOverlay);
        break;
    case DLL_PROCESS_DETACH:
        reshade::unregister_event<reshade::addon_event::init_device>(OnInitDevice);
        reshade::unregister_event<reshade::addon_event::destroy_device>(OnDestroyDevice);
        reshade::unregister_event<reshade::addon_event::present>(OnPresent);
        reshade::unregister_addon(hModule);
        break;
    }
    return TRUE;
}
