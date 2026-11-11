#pragma once

// Scoped x87 precision control (issue #18).
//
// GCC and Clang targeting x86 MinGW, Linux and macOS keep long double as the
// 80-bit x87 type (64-bit significand).  Code that relies on those extra bits
// is only correct when the x87 precision-control field (control word bits
// 8-9) selects 64-bit rounding.  Windows x64 starts every thread with that
// field at 53 bits and treats the control word as nonvolatile (x64 calling
// convention, FPCSR), so inside Stata or python.exe every long double
// operation of a MinGW-built module rounds to 53 bits.  Linux and x86 macOS
// start threads at 64 bits.
//
// ScopedExtendedPrecision raises ONLY the precision-control bits to 64-bit
// rounding for the lifetime of the object and restores the caller's complete
// control word on every exit, normal or by exception.  Exception masks,
// rounding control and MXCSR are never touched, and a thread whose control
// word already selects 64 bits is not written at all.  The control word is
// per thread: OpenMP workers and other threads must each own a scope (no
// state is shared or inherited by this utility).  Keep scopes private to
// numerical code: never call back into the host (Stata, Python, R) while a
// scope that changed the word is alive.
//
// The scope establishes the precision; it does not prove it.  Callers that
// depend on the extra bits keep their own arithmetic confirmation, e.g.
// long_double_significand_active() below, and must not take a silent
// lower-precision fallback when it fails.
//
// MSVC (long double == double) and non-x86 targets get an inert scope.

#include <cmath>
#include <limits>

namespace hdfe::detail::x87 {

#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
#define XHDFE_X87_PRECISION_CONTROL 1

constexpr unsigned short kPrecisionControlMask = 0x0300u;  // bits 8-9
constexpr unsigned short kPrecisionExtended = 0x0300u;     // 11b: 64-bit significand

inline unsigned short control_word() noexcept {
    unsigned short word = 0;
    __asm__ __volatile__("fnstcw %0" : "=m"(word) : : "memory");
    return word;
}

inline void load_control_word(unsigned short word) noexcept {
    __asm__ __volatile__("fldcw %0" : : "m"(word) : "memory");
}

class ScopedExtendedPrecision {
public:
    ScopedExtendedPrecision() noexcept : saved_(control_word()) {
        const unsigned short wanted =
            static_cast<unsigned short>(saved_ | kPrecisionExtended);
        if (wanted != saved_) {
            load_control_word(wanted);
            changed_ = true;
        }
    }
    ~ScopedExtendedPrecision() {
        if (changed_) {
            load_control_word(saved_);
        }
    }
    ScopedExtendedPrecision(const ScopedExtendedPrecision&) = delete;
    ScopedExtendedPrecision& operator=(const ScopedExtendedPrecision&) = delete;

    bool changed() const noexcept { return changed_; }
    unsigned short saved_control_word() const noexcept { return saved_; }

private:
    unsigned short saved_;
    bool changed_ = false;
};

#else

class ScopedExtendedPrecision {
public:
    ScopedExtendedPrecision() noexcept = default;
    ScopedExtendedPrecision(const ScopedExtendedPrecision&) = delete;
    ScopedExtendedPrecision& operator=(const ScopedExtendedPrecision&) = delete;

    bool changed() const noexcept { return false; }
};

#endif

// True when long double arithmetic currently keeps the lowest significand bit
// it advertises.  Separate volatile stores and loads materialize both
// operations even under -ffast-math, so the probe observes the active
// rounding precision instead of trusting numeric_limits.
inline bool long_double_significand_active() noexcept {
    volatile long double one = 1.0L;
    volatile long double unit =
        std::ldexp(1.0L, 1 - std::numeric_limits<long double>::digits);
    volatile long double probe = one + unit;
    volatile long double recovered = probe - one;
    return recovered == unit;
}

}  // namespace hdfe::detail::x87
