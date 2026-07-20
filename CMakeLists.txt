cmake_minimum_required(VERSION 3.16)
project(xhdfe VERSION 2.19.0.20260720 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Production artifacts are Release by default and must never inherit Eigen's
# assertion-enabled ABI merely because a stale build directory once selected
# Debug.  Diagnostic non-Release builds remain possible, but require an
# explicit opt-out and are not shippable.
option(XHDFE_ALLOW_NON_RELEASE_BUILD
       "Allow assertion-enabled diagnostic builds (never use for release artifacts)."
       OFF)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
if(NOT XHDFE_ALLOW_NON_RELEASE_BUILD)
    if(CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING
            "Supported configurations for certifiable artifacts" FORCE)
    elseif(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
        message(FATAL_ERROR
            "xhdfe production builds require CMAKE_BUILD_TYPE=Release. "
            "Use -DXHDFE_ALLOW_NON_RELEASE_BUILD=ON only for local diagnostics.")
    endif()
    add_compile_definitions(NDEBUG)
endif()

include(FetchContent)
include(CheckCXXCompilerFlag)

set(XHDFE_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

set(XHDFE_ENABLE_MARCH_NATIVE_DEFAULT ON)
if(APPLE)
    set(XHDFE_ENABLE_MARCH_NATIVE_DEFAULT OFF)
endif()
option(XHDFE_ENABLE_MARCH_NATIVE "Enable -march=native optimizations when supported by the active target compiler." ${XHDFE_ENABLE_MARCH_NATIVE_DEFAULT})
option(XHDFE_BUILD_PYTHON "Build the Python bindings (pybind11)." ON)
option(XHDFE_ENABLE_CUDA "Enable CUDA backend for GPU acceleration." OFF)
option(XHDFE_ENABLE_METAL "Enable Metal backend for GPU acceleration." OFF)
set(XHDFE_GPU_BACKEND_DEFAULT "cpu" CACHE STRING "Default GPU backend (cpu|cuda|metal).")

function(xhdfe_enable_native_tuning target_name)
    if(NOT XHDFE_ENABLE_MARCH_NATIVE)
        return()
    endif()
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        return()
    endif()

    check_cxx_compiler_flag("-march=native" XHDFE_CXX_HAS_MARCH_NATIVE)
    check_cxx_compiler_flag("-mtune=native" XHDFE_CXX_HAS_MTUNE_NATIVE)

    if(XHDFE_CXX_HAS_MARCH_NATIVE)
        target_compile_options(${target_name} PRIVATE -march=native)
    else()
        message(WARNING "XHDFE_ENABLE_MARCH_NATIVE=ON, but ${CMAKE_CXX_COMPILER_ID} rejects -march=native for the active target; skipping it.")
    endif()

    if(XHDFE_CXX_HAS_MTUNE_NATIVE)
        target_compile_options(${target_name} PRIVATE -mtune=native)
    endif()
endfunction()

if(XHDFE_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "XHDFE_ENABLE_CUDA=ON but no CUDA compiler was found.")
    endif()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
    foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        string(REGEX MATCH "^[0-9]+" arch_num "${arch}")
        if(arch_num STREQUAL "")
            message(FATAL_ERROR "Unsupported CMAKE_CUDA_ARCHITECTURES entry: ${arch}")
        endif()
        if(arch_num LESS 75)
            message(FATAL_ERROR "CUDA arch ${arch} is < 75; minimum supported is T4 class (sm_75).")
        endif()
    endforeach()
endif()
if(XHDFE_ENABLE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "XHDFE_ENABLE_METAL=ON requires an Apple platform.")
    endif()
    enable_language(OBJCXX)
endif()

# Eigen3: pin provenance to the copy tracked inside this repository.  Never
# allow find_package() to resolve an unrelated sibling checkout or host install.
set(XHDFE_EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/r/xhdfe/src/eigen"
    CACHE PATH "In-repository Eigen 3.4.0 source directory")
if(NOT EXISTS "${XHDFE_EIGEN_DIR}/Eigen/Dense")
    message(FATAL_ERROR
        "Vendored Eigen is missing at ${XHDFE_EIGEN_DIR}; refusing an "
        "uncertifiable out-of-repository fallback.")
endif()
message(STATUS "Using pinned in-repository Eigen: ${XHDFE_EIGEN_DIR}")
add_library(Eigen3::Eigen INTERFACE IMPORTED)
target_include_directories(Eigen3::Eigen INTERFACE "${XHDFE_EIGEN_DIR}")

# pybind11 (optional)
if(XHDFE_BUILD_PYTHON)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        if(EXISTS "${XHDFE_THIRD_PARTY_DIR}/pybind11-2.11.1/CMakeLists.txt")
            message(STATUS "pybind11 not found. Using vendored third_party/pybind11-2.11.1.")
            add_subdirectory("${XHDFE_THIRD_PARTY_DIR}/pybind11-2.11.1" "${CMAKE_BINARY_DIR}/_deps/pybind11-build" EXCLUDE_FROM_ALL)
        else()
            message(STATUS "pybind11 not found. Fetching from source...")
            FetchContent_Declare(
                pybind11
                DOWNLOAD_EXTRACT_TIMESTAMP TRUE
                URL https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.tar.gz
            )
            FetchContent_MakeAvailable(pybind11)
        endif()
    endif()
endif()

# OpenMP (optional)
find_package(OpenMP)

# Core library (v11 implementation)
set(XHDFE_SOURCES
    src/hdfe_regressor_v11.cpp
    src/fe_absorption.cpp
    src/schwarz_demean.cpp
    src/akm_kss.cpp
    src/iv.cpp
    src/ols.cpp
)
if(XHDFE_ENABLE_CUDA)
    list(APPEND XHDFE_SOURCES src/fe_absorption_cuda.cu src/akm_kss_cuda.cu)
endif()
if(XHDFE_ENABLE_METAL)
    list(APPEND XHDFE_SOURCES src/fe_absorption_metal.mm)
endif()
add_library(xhdfe ${XHDFE_SOURCES})

target_include_directories(xhdfe PUBLIC include)
target_link_libraries(xhdfe PUBLIC Eigen3::Eigen)
target_compile_definitions(xhdfe PUBLIC HDFE_GPU_BACKEND_DEFAULT="${XHDFE_GPU_BACKEND_DEFAULT}")
if(MSVC)
    target_compile_options(xhdfe PRIVATE /bigobj)
elseif(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(xhdfe PRIVATE -Wa,-mbig-obj)
endif()

if(XHDFE_ENABLE_CUDA)
    target_compile_definitions(xhdfe PUBLIC HDFE_USE_CUDA)
    target_compile_options(xhdfe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr>)
endif()
if(XHDFE_ENABLE_METAL)
    target_compile_definitions(xhdfe PUBLIC HDFE_USE_METAL)
    target_compile_options(xhdfe PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
    target_link_libraries(xhdfe PUBLIC "-framework Metal" "-framework Foundation")
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(xhdfe PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(xhdfe PUBLIC HDFE_USE_OPENMP)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    xhdfe_enable_native_tuning(xhdfe)
    target_compile_options(xhdfe PRIVATE
        -O3
        -ffast-math
        -funroll-loops
        -ftree-vectorize
        -fomit-frame-pointer
        -fno-signed-zeros
        -fno-trapping-math
        -fassociative-math
        -freciprocal-math
    )
endif()

# Python bindings
if(XHDFE_BUILD_PYTHON)
    add_library(py_hdfe_v11 MODULE python/py_hdfe_v11.cpp)
    target_link_libraries(py_hdfe_v11 PRIVATE xhdfe pybind11::module)
    pybind11_extension(py_hdfe_v11)
    if(MSVC)
        target_compile_options(py_hdfe_v11 PRIVATE /bigobj)
    elseif(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(py_hdfe_v11 PRIVATE -Wa,-mbig-obj)
    endif()

    if(OpenMP_CXX_FOUND)
        target_link_libraries(py_hdfe_v11 PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(py_hdfe_v11 PRIVATE HDFE_USE_OPENMP)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        xhdfe_enable_native_tuning(py_hdfe_v11)
        target_compile_options(py_hdfe_v11 PRIVATE
            -O3
            -ffast-math
            -funroll-loops
            -ftree-vectorize
            -fomit-frame-pointer
            -fno-signed-zeros
            -fno-trapping-math
            -fassociative-math
            -freciprocal-math
        )
    endif()
endif()
