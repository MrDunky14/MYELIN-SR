cmake_minimum_required(VERSION 3.20)
project(MYELIN-SR VERSION 2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Windows SDK ──
find_package(WindowsSDK QUIET)

# ── DirectX 12 Libraries ──
set(DX12_LIBS d3d12.lib dxgi.lib d3dcompiler.lib user32.lib)

# ── HLSL Shader Compilation ──
set(SHADER_DIR ${CMAKE_SOURCE_DIR}/src)
set(SHADER_OUTPUT_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

# Custom command to compile HLSL to CSO using Shader Model 6.2 + FP16
add_custom_command(
    OUTPUT ${SHADER_OUTPUT_DIR}/myelin_compute.cso
    COMMAND dxc -T cs_6_2 -E CSMain -O3 -enable-16bit-types -HV 2021
            -Fo ${SHADER_OUTPUT_DIR}/myelin_compute.cso
            ${SHADER_DIR}/myelin_compute.hlsl
    DEPENDS ${SHADER_DIR}/myelin_compute.hlsl
    COMMENT "Compiling myelin_compute.hlsl to SM 6.2 CSO with FP16 support"
)

add_custom_target(Shaders ALL DEPENDS ${SHADER_OUTPUT_DIR}/myelin_compute.cso)

# ════════════════════════════════════════
# Target 1: DX12 Benchmark Executable
# ════════════════════════════════════════
add_executable(dx12_benchmark
    src/dx12_compute_sandbox.cpp
)
target_link_libraries(dx12_benchmark PRIVATE ${DX12_LIBS})
target_compile_definitions(dx12_benchmark PRIVATE
    UNICODE _UNICODE WIN32_LEAN_AND_MEAN NOMINMAX
)
add_dependencies(dx12_benchmark Shaders)

# Copy resources to build dir
add_custom_command(TARGET dx12_benchmark POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/outputs
            ${CMAKE_BINARY_DIR}/outputs
    COMMAND ${CMAKE_COMMAND} -E copy
            ${SHADER_DIR}/myelin_compute.hlsl
            ${CMAKE_BINARY_DIR}/src/myelin_compute.hlsl
    COMMENT "Copying weights and shaders to build directory"
)

# ════════════════════════════════════════
# Target 2: ReShade Addon DLL (optional)
# ════════════════════════════════════════
option(BUILD_RESHADE_ADDON "Build ReShade addon DLL" OFF)

if(BUILD_RESHADE_ADDON)
    # ReShade SDK must be available
    set(RESHADE_SDK_DIR "${CMAKE_SOURCE_DIR}/external/reshade" CACHE PATH "Path to ReShade SDK")
    
    if(NOT EXISTS "${RESHADE_SDK_DIR}/include/reshade.hpp")
        message(WARNING "ReShade SDK not found at ${RESHADE_SDK_DIR}. Skipping addon build.")
    else()
        add_library(myelin_sr_addon SHARED
            src/reshade_myelin_addon.cpp
        )
        target_include_directories(myelin_sr_addon PRIVATE ${RESHADE_SDK_DIR}/include)
        target_link_libraries(myelin_sr_addon PRIVATE ${DX12_LIBS})
        set_target_properties(myelin_sr_addon PROPERTIES
            OUTPUT_NAME "myelin_sr"
            SUFFIX ".addon64"
        )
    endif()
endif()

# ════════════════════════════════════════
# Install
# ════════════════════════════════════════
install(TARGETS dx12_benchmark RUNTIME DESTINATION bin)
install(FILES
    outputs/myelin_engine_v2.bin
    src/myelin_compute.hlsl
    DESTINATION bin
)
install(FILES README.md LICENSE DESTINATION .)
