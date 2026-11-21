# Standalone operator bench of the CUDA reduced tier (package 2, stage 2a,
# 17 November 2026; integration the same day). Not part of the main build: it
# compiles the operator sources and the bench directly, with the flags of the main
# library (CPU: Release, -march=native, the core's fast-math set, OpenMP; CUDA: -O3,
# sm_90). Three device operators: the integrated one (src/, with the test hooks for
# its source-cell check), the stage-2b candidate (candidate_operator.cu) and, as a
# frozen reference used only here, the operator before package 2 (frozen_20261115/).
#   env -u LD_LIBRARY_PATH cmake -S tests/reduced_cuda_bench -B build_p2_bench -DCMAKE_BUILD_TYPE=Release
#   env -u LD_LIBRARY_PATH cmake --build build_p2_bench -j 16
cmake_minimum_required(VERSION 3.24)
project(xhdfe_reduced_cuda_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 90)
endif()

get_filename_component(XHDFE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
find_package(OpenMP REQUIRED)

add_executable(xhdfe_reduced_cuda_bench
    reduced_cuda_bench.cpp
    candidate_operator.cu
    frozen_20261115/frozen_operator.cu
    "${XHDFE_ROOT}/src/reduced_schur_pcg_cuda.cu")
# Registers and spills of every kernel in the build log (B8.3).
set_source_files_properties(candidate_operator.cu "${XHDFE_ROOT}/src/reduced_schur_pcg_cuda.cu"
    PROPERTIES COMPILE_OPTIONS "-Xptxas=-v")
target_include_directories(xhdfe_reduced_cuda_bench PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}" "${XHDFE_ROOT}/src" "${XHDFE_ROOT}/include")
target_include_directories(xhdfe_reduced_cuda_bench SYSTEM PRIVATE "${XHDFE_ROOT}/r/xhdfe/src/eigen")
target_compile_definitions(xhdfe_reduced_cuda_bench PRIVATE HDFE_USE_CUDA HDFE_USE_OPENMP XHDFE_REDUCED_CUDA_TEST_HOOKS)
target_compile_options(xhdfe_reduced_cuda_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -mtune=native -ffast-math -fno-finite-math-only
        -funroll-loops -ftree-vectorize -fomit-frame-pointer -fno-signed-zeros -fno-trapping-math
        -fassociative-math -freciprocal-math>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr -Xcompiler=-fopenmp>)
target_link_libraries(xhdfe_reduced_cuda_bench PRIVATE OpenMP::OpenMP_CXX)
