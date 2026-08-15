cmake_minimum_required(VERSION 3.16)
project(xhdfe VERSION 2.24.0.20260815 LANGUAGES CXX)

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
include(CTest)

set(XHDFE_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

set(XHDFE_ENABLE_MARCH_NATIVE_DEFAULT ON)
if(APPLE OR WIN32)
    # Prebuilt/native-extension artifacts must not inherit the build host's
    # instruction set. Users can still opt in explicitly for a local-only build.
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

# Eigen3: pin provenance to a copy shipped inside this source tree.  The full
# checkout keeps the R-package mirror as the primary copy; Python sdists carry
# the canonical third_party tree instead of duplicating that large directory.
# Never allow find_package() to resolve an unrelated sibling checkout or host
# install.
set(_XHDFE_EIGEN_R_MIRROR "${CMAKE_CURRENT_SOURCE_DIR}/r/xhdfe/src/eigen")
set(_XHDFE_EIGEN_SDIST
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen-3.4.0")
if(EXISTS "${_XHDFE_EIGEN_R_MIRROR}/Eigen/Dense")
    set(_XHDFE_EIGEN_DEFAULT "${_XHDFE_EIGEN_R_MIRROR}")
elseif(EXISTS "${_XHDFE_EIGEN_SDIST}/Eigen/Dense")
    set(_XHDFE_EIGEN_DEFAULT "${_XHDFE_EIGEN_SDIST}")
else()
    set(_XHDFE_EIGEN_DEFAULT "${_XHDFE_EIGEN_R_MIRROR}")
endif()
set(XHDFE_EIGEN_DIR "${_XHDFE_EIGEN_DEFAULT}"
    CACHE PATH "Source-tree Eigen 3.4.0 directory")
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
    list(APPEND XHDFE_SOURCES
        src/fe_absorption_cuda.cu
        src/akm_kss_cuda.cu
        src/fe_absorption_cuda_certificate.cpp
        src/fe_absorption_cuda_certificate.cu)
    # WP3/P1 — verifier compile contract (plan §7.6): the certificate verifier
    # TU is built with strict FP semantics, independently of whatever the
    # solver TUs use: no fast-math, no contraction (device --fmad=false,
    # host -ffp-contract=off), full-precision div/sqrt, no FTZ. The host-side
    # pass also records its switches into .GCC.command.line so the §15.3 flag
    # gate can prove the contract from the OBJECT, fail-closed.
    # Source-level options append AFTER the target-level fast-math block, and
    # for GCC the last flag wins — so every relaxed flag the target sets is
    # explicitly negated here. The TU's own #error guard proves the negation
    # took effect (it fired when these negations were missing).
    # The host-side negations are passed BARE, not via -Xcompiler: nvcc groups
    # -Xcompiler flags BEFORE the -forward-unknown-to-host-compiler group where
    # the target's -ffast-math lands, which would put the negations first and
    # lose the last-wins battle. Bare unknown flags join the same forwarded
    # group, after the target's, so GCC sees them last. The TU's #error guard
    # is the proof this ordering holds on every future toolchain.
    set_source_files_properties(src/fe_absorption_cuda_certificate.cu
        PROPERTIES COMPILE_OPTIONS
        "--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false;-fno-fast-math;-fno-unsafe-math-optimizations;-fno-associative-math;-fno-reciprocal-math;-fsigned-zeros;-ftrapping-math;-fno-finite-math-only;-ffp-contract=off;-frecord-gcc-switches")
    # NOTE deliberately NOT applied to the existing host certificate
    # (fe_absorption_cuda_certificate.cpp): its arithmetic feeds the public
    # abs_residual/abs_residual_rel fields and must keep byte-identical
    # codegen until the WP6 re-baseline under P2. The strict contract governs
    # the NEW dedicated verifier TUs only.
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

    # Non-finite guards in the fast-math translation units must remain live
    # in the exact production compilation mode. The portable source gate
    # prevents raw std/Eigen predicates from returning. Native builds also run
    # a code-generation probe; cross builds retain the source gate without
    # attempting to execute a foreign binary.
    if(CMAKE_CROSSCOMPILING)
        add_custom_target(xhdfe_nonfinite_guard
            COMMAND ${CMAKE_COMMAND} -P
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_no_raw_isfinite.cmake
            COMMENT "fast-math non-finite source guard"
            VERBATIM)
    else()
        add_executable(xhdfe_ieee_bits_liveness EXCLUDE_FROM_ALL
            tests/ieee_bits_liveness.cpp)
        target_include_directories(xhdfe_ieee_bits_liveness PRIVATE include)
        target_link_libraries(xhdfe_ieee_bits_liveness PRIVATE Eigen3::Eigen)
        xhdfe_enable_native_tuning(xhdfe_ieee_bits_liveness)
        target_compile_options(xhdfe_ieee_bits_liveness PRIVATE
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
        add_custom_target(xhdfe_nonfinite_guard
            COMMAND ${CMAKE_COMMAND} -P
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_no_raw_isfinite.cmake
            COMMAND $<TARGET_FILE:xhdfe_ieee_bits_liveness>
            DEPENDS xhdfe_ieee_bits_liveness
            COMMENT "fast-math non-finite guard liveness gate"
            VERBATIM)
        if(BUILD_TESTING)
            add_test(NAME xhdfe_ieee_bits_liveness
                     COMMAND $<TARGET_FILE:xhdfe_ieee_bits_liveness>)
        endif()
    endif()
    add_dependencies(xhdfe xhdfe_nonfinite_guard)
endif()

if(BUILD_TESTING)
    add_test(NAME xhdfe_no_raw_nonfinite_guards
             COMMAND ${CMAKE_COMMAND} -P
                     ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_no_raw_isfinite.cmake)

    add_executable(xhdfe_fail_closed_entrypoints
                   tests/fail_closed_entrypoints.cpp)
    target_link_libraries(xhdfe_fail_closed_entrypoints PRIVATE xhdfe)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        xhdfe_enable_native_tuning(xhdfe_fail_closed_entrypoints)
        target_compile_options(xhdfe_fail_closed_entrypoints PRIVATE
            -O3
            -ffast-math
            -funroll-loops
            -ftree-vectorize
            -fomit-frame-pointer
            -fno-signed-zeros
            -fno-trapping-math
            -fassociative-math
            -freciprocal-math)
    endif()
    add_test(NAME xhdfe_fail_closed_entrypoints
             COMMAND $<TARGET_FILE:xhdfe_fail_closed_entrypoints>)
endif()

# Python bindings
if(XHDFE_BUILD_PYTHON)
    add_library(py_hdfe_v11 MODULE python/py_hdfe_v11.cpp)
    target_link_libraries(py_hdfe_v11 PRIVATE xhdfe pybind11::module)
    pybind11_extension(py_hdfe_v11)
    if(XHDFE_ENABLE_CUDA)
        # §7.6/R-04 device FMA gate on the linked module (the gate filters to
        # hdfe_cert functions, so the solver kernels' legitimate FMA use is
        # out of scope). Fail-closed: a contraction regression in the
        # verifier breaks the build, not the certification.
        add_custom_command(TARGET py_hdfe_v11 POST_BUILD
            COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_verifier_device_fma.sh
                    $<TARGET_FILE:py_hdfe_v11>
            COMMENT "verifier device FMA gate (§7.6)"
            VERBATIM)
    endif()
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

    if(BUILD_TESTING AND NOT CMAKE_CROSSCOMPILING)
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            add_test(
                NAME xhdfe_audit_20260804_contracts
                COMMAND ${Python3_EXECUTABLE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/tests/audit_20260804_contracts.py
                        --module-dir $<TARGET_FILE_DIR:py_hdfe_v11>)
            set_tests_properties(xhdfe_audit_20260804_contracts PROPERTIES
                ENVIRONMENT "XHDFE_GPU_BACKEND=cpu")
        endif()
    endif()
endif()
