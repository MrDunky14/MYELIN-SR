cmake_minimum_required(VERSION 3.20)
project(MYELIN-SR VERSION 3.3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── DirectX 12 Libraries ──
set(DX12_LIBS d3d12.lib dxgi.lib user32.lib)

include_directories(${CMAKE_SOURCE_DIR}/include)

# ════════════════════════════════════════
# Target 1: Parity Test Oracle
# ════════════════════════════════════════
add_executable(dx12_parity_test
    src/dx12_parity_test.cpp
)
target_link_libraries(dx12_parity_test PRIVATE ${DX12_LIBS})
target_compile_definitions(dx12_parity_test PRIVATE
    UNICODE _UNICODE WIN32_LEAN_AND_MEAN NOMINMAX
)

# ════════════════════════════════════════
# Target 2: 7-Variant Roofline Benchmark
# ════════════════════════════════════════
add_executable(dx12_bench_v3
    src/dx12_bench_v3.cpp
)
target_link_libraries(dx12_bench_v3 PRIVATE ${DX12_LIBS})
target_compile_definitions(dx12_bench_v3 PRIVATE
    UNICODE _UNICODE WIN32_LEAN_AND_MEAN NOMINMAX
)

# ════════════════════════════════════════
# Install
# ════════════════════════════════════════
install(TARGETS dx12_parity_test dx12_bench_v3 RUNTIME DESTINATION bin)
install(FILES README.md LICENSE DESTINATION .)
