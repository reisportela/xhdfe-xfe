#ifndef HDFE_COMPILER_COMPAT_HPP
#define HDFE_COMPILER_COMPAT_HPP

#if defined(_MSC_VER)
#include <xmmintrin.h>
#endif

namespace hdfe {
namespace detail {

template <int RW, int Locality>
inline void compiler_prefetch(const void* ptr) {
#if defined(_MSC_VER)
    (void)RW;
    (void)Locality;
    _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(ptr, RW, Locality);
#else
    (void)RW;
    (void)Locality;
    (void)ptr;
#endif
}

}  // namespace detail
}  // namespace hdfe

#if defined(_MSC_VER)
#define HDFE_RESTRICT __restrict
#else
#define HDFE_RESTRICT __restrict__
#endif

#define HDFE_PREFETCH(ptr, rw, locality) \
    ::hdfe::detail::compiler_prefetch<(rw), (locality)>((ptr))

#endif  // HDFE_COMPILER_COMPAT_HPP
