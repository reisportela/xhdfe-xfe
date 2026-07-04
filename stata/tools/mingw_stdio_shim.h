/* Force-included (via -include) for MinGW-w64 Windows builds only.
 *
 * On some MinGW-w64 toolchains the global C stdio stream macros
 * (stdin/stdout/stderr) are not exposed to C++ translation units even with
 * -std=gnu++17 and <cstdio> included, which breaks std::fprintf(stderr, ...).
 * This shim guarantees they are defined using the canonical MinGW-w64 CRT
 * accessor (__acrt_iob_func), without touching the shared C++ core sources.
 */
#pragma once
#include <stdio.h>

#if defined(_WIN32) && !defined(stderr)
#ifdef __cplusplus
extern "C" {
#endif
FILE *__acrt_iob_func(unsigned index);
#ifdef __cplusplus
}
#endif
#define stdin  (__acrt_iob_func(0))
#define stdout (__acrt_iob_func(1))
#define stderr (__acrt_iob_func(2))
#endif
