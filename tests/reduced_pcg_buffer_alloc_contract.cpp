// The reduced tier's 64-byte buffers on the branch Apple targets take (17 November
// 2026): Buffer::allocate takes its block from posix_memalign and Buffer::release gives it
// back with std::free (src/reduced_schur_pcg.hpp), because the aligned operator new of
// C++17 is not available to the macOS Intel slice at its deployment target. This test is
// built with XHDFE_REDUCED_PCG_POSIX_MEMALIGN, which forces that branch on any platform,
// and checks: buffers of 0, 1 and many elements of several types are allocated, aligned
// to 64 bytes, zeroed, moved (construction and assignment, self-assignment included) and
// released; allocate() on a filled buffer replaces its block; a request whose byte count
// does not fit, and one the allocator cannot serve, throw std::bad_alloc and leave the
// buffer empty; the aligned operator new is never called. Prints BUFFER_ALLOC_PASS.
// The macOS compilation itself is not exercised here.
#ifndef XHDFE_REDUCED_PCG_POSIX_MEMALIGN
#error "build this test with -DXHDFE_REDUCED_PCG_POSIX_MEMALIGN"
#endif
#include "reduced_schur_pcg.hpp"

#ifndef RPCG_BUFFER_POSIX_MEMALIGN
#error "the posix_memalign branch of Buffer was not selected"
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
int aligned_new_calls = 0;
}  // namespace

// Counted, to prove the branch never reaches the C++17 aligned allocation.
void* operator new(std::size_t bytes, std::align_val_t alignment) {
    ++aligned_new_calls;
    void* p = nullptr;
    if (posix_memalign(&p, static_cast<std::size_t>(alignment), bytes != 0 ? bytes : 1) != 0) throw std::bad_alloc();
    return p;
}
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

namespace {

int checks = 0;
void expect(bool ok, const std::string& what) {
    ++checks;
    if (!ok) throw std::runtime_error(what);
}

template <class T>
bool aligned64(const hdfe::reduced_pcg::internal::Buffer<T>& b) {
    return (reinterpret_cast<std::uintptr_t>(b.data()) % 64U) == 0;
}

template <class T>
void exercise(const char* type) {
    using Buffer = hdfe::reduced_pcg::internal::Buffer<T>;
    const std::string tag = std::string(type) + ": ";
    for (std::size_t count : {std::size_t{0}, std::size_t{1}, std::size_t{7}, std::size_t{4096},
                              std::size_t{1} << 20}) {
        const std::string what = tag + std::to_string(count) + " elements: ";
        Buffer a;
        expect(a.data() == nullptr && a.size() == 0, what + "a new buffer is not empty");
        a.allocate(count);
        const std::size_t held = count == 0 ? 1 : count;  // allocate(0) holds one element
        expect(a.data() != nullptr && a.size() == held, what + "allocate");
        expect(aligned64(a), what + "not aligned to 64 bytes");
        for (std::size_t i = 0; i < held; ++i) a[i] = static_cast<T>(i % 97 + 1);
        a.zero();
        for (std::size_t i = 0; i < held; ++i) expect(a[i] == T(0), what + "zero");
        a[held - 1] = T(5);
        T* const block = a.data();
        Buffer b(std::move(a));
        expect(a.data() == nullptr && a.size() == 0, what + "moved-from buffer not empty");
        expect(b.data() == block && b.size() == held && b[held - 1] == T(5), what + "move construction");
        Buffer c;
        c.allocate(3);
        c = std::move(b);  // releases c's own block
        expect(b.data() == nullptr && c.data() == block && c.size() == held, what + "move assignment");
        Buffer& alias = c;
        c = std::move(alias);  // self-assignment keeps the block
        expect(c.data() == block && c.size() == held, what + "self move assignment");
        c.allocate(held + 64);  // a filled buffer replaces its block
        expect(c.size() == held + 64 && aligned64(c), what + "re-allocate");
        c.release();
        expect(c.data() == nullptr && c.size() == 0, what + "release");
        c.release();  // releasing an empty buffer is a no-op
        expect(c.data() == nullptr, what + "second release");
    }
    // A byte count that does not fit size_t (types wider than a byte), then one no
    // allocator can serve.
    Buffer d;
    d.allocate(16);
    bool thrown = false;
    if (sizeof(T) > 1) {
        try {
            d.allocate(std::numeric_limits<std::size_t>::max() / sizeof(T) + 1);
        } catch (const std::bad_alloc&) {
            thrown = true;
        }
        expect(thrown && d.data() == nullptr && d.size() == 0, tag + "overflowing request");
        d.allocate(16);
    }
    thrown = false;
    try {
        d.allocate(std::numeric_limits<std::size_t>::max() / sizeof(T) - 64);
    } catch (const std::bad_alloc&) {
        thrown = true;
    }
    expect(thrown && d.data() == nullptr && d.size() == 0, tag + "unservable request");
    d.allocate(2);
    expect(d.data() != nullptr && aligned64(d), tag + "allocation after a failure");
}

}  // namespace

int main() {
    try {
        exercise<double>("double");
        exercise<int>("int");
        exercise<char>("char");
        exercise<std::uint64_t>("uint64");
        expect(aligned_new_calls == 0, "the aligned operator new was called " + std::to_string(aligned_new_calls) + " times");
    } catch (const std::exception& e) {
        std::printf("BUFFER_ALLOC_FAIL %s\n", e.what());
        return 1;
    }
    std::printf("BUFFER_ALLOC_PASS %d checks (posix_memalign branch, aligned operator new calls %d)\n", checks,
                aligned_new_calls);
    return 0;
}
