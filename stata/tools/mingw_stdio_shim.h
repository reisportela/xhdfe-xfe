/* Force-included (via -include) for MinGW-w64 Windows builds only.
 *
 * On some MinGW-w64 toolchains <cstdio> does not leave the global C stdio
 * stream macros (stdin/stdout/stderr) visible to C++ translation units, which
 * breaks std::fprintf(stderr, ...). We fix this WITHOUT editing the shared C++
 * core: this shim is included first (via -include), so it trips the <cstdio>
 * include guard up front and then (re)defines the stream macros. Because the
 * guard is already tripped, the core's own later `#include <cstdio>` is a
 * no-op and cannot undo these definitions.
 */
#pragma once
#include <cstdio>   /* trip the include guard before the core includes it */
#include <stdio.h>

#if defined(_WIN32)
#ifdef __cplusplus
extern "C" {
#endif
FILE *__acrt_iob_func(unsigned index);
#ifdef __cplusplus
}
#endif
#ifndef stdin
#define stdin  (__acrt_iob_func(0))
#endif
#ifndef stdout
#define stdout (__acrt_iob_func(1))
#endif
#ifndef stderr
#define stderr (__acrt_iob_func(2))
#endif
#endif
