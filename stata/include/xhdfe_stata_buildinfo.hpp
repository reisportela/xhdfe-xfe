#pragma once

#include "stplugin.h"

namespace xhdfe_stata {

// These are caller locals (the SPI uses a leading underscore to select a
// local). Only the explicit buildinfo task writes them.
inline ST_retcode save_buildinfo(const char* role) {
    [[maybe_unused]] constexpr const char* unrecorded = "unrecorded";
#ifdef XHDFE_STATA_BUILD_CPU
    constexpr const char* cpu = XHDFE_STATA_BUILD_CPU;
#else
    constexpr const char* cpu = unrecorded;
#endif
#ifdef HDFE_USE_CUDA
    constexpr const char* cuda = "1";
#ifdef XHDFE_STATA_BUILD_SMS
    constexpr const char* sms = XHDFE_STATA_BUILD_SMS;
#else
    constexpr const char* sms = unrecorded;
#endif
#else
    constexpr const char* cuda = "0";
    constexpr const char* sms = "none";
#endif
#if defined(_OPENMP) && defined(HDFE_USE_OPENMP)
    constexpr const char* openmp = "1";
#else
    constexpr const char* openmp = "0";
#endif
#if defined(__clang__)
    constexpr const char* compiler = "Clang " __clang_version__;
#elif defined(__GNUC__)
    constexpr const char* compiler = "GCC " __VERSION__;
#elif defined(_MSC_FULL_VER)
#define XHDFE_STATA_STRING_VALUE_(n) #n
#define XHDFE_STATA_STRING_VALUE(n) XHDFE_STATA_STRING_VALUE_(n)
    constexpr const char* compiler = "MSVC " XHDFE_STATA_STRING_VALUE(_MSC_FULL_VER);
#undef XHDFE_STATA_STRING_VALUE
#undef XHDFE_STATA_STRING_VALUE_
#elif defined(__VERSION__)
    constexpr const char* compiler = __VERSION__;
#else
    constexpr const char* compiler = unrecorded;
#endif
#ifdef XHDFE_STATA_BUILD_VERSION
    constexpr const char* version = XHDFE_STATA_BUILD_VERSION;
#else
    constexpr const char* version = unrecorded;
#endif

    const struct {
        const char* local;
        const char* value;
    } fields[] = {
        {"_xhdfe_buildinfo_protocol", "1"},
        {"_xhdfe_buildinfo_role", role},
        {"_xhdfe_buildinfo_cpu", cpu},
        {"_xhdfe_buildinfo_cuda", cuda},
        {"_xhdfe_buildinfo_sms", sms},
        {"_xhdfe_buildinfo_openmp", openmp},
        {"_xhdfe_buildinfo_compiler", compiler},
        {"_xhdfe_buildinfo_version", version},
    };
    for (const auto& field : fields) {
        const ST_retcode rc = SF_macro_save(
            const_cast<char*>(field.local), const_cast<char*>(field.value));
        if (rc) return rc;
    }
    return 0;
}

}  // namespace xhdfe_stata
