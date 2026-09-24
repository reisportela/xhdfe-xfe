#pragma once

#include "hdfe/parallel_work_observer.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>
#if defined(__SHA__) && (defined(__x86_64__) || defined(__i386__)) && !defined(XHDFE_CACHE_SCALAR_SHA)
#include <immintrin.h>
#define XHDFE_CACHE_SHA_INTRINSICS 1
#endif
#ifdef HDFE_USE_OPENMP
#include <omp.h>
#endif

namespace hdfe {
namespace detail {

// SHA-256, FIPS 180-4. Integer-only cache custody, independent of fast-math.
class CacheSha256 {
    std::array<std::uint32_t, 8> state_{{
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19}};
    std::array<std::uint8_t, 64> buffer_{};
    std::uint64_t bytes_ = 0;
    std::size_t used_ = 0;

    static std::uint32_t rotr(std::uint32_t x, unsigned n) {
        return (x >> n) | (x << (32 - n));
    }
    void compress() {
        static constexpr std::uint32_t constants[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        std::uint32_t w[64];
        for (unsigned i = 0; i < 16; ++i) {
            const auto* p = buffer_.data() + 4 * i;
            w[i] = (std::uint32_t(p[0]) << 24) | (std::uint32_t(p[1]) << 16) |
                   (std::uint32_t(p[2]) << 8) | std::uint32_t(p[3]);
        }
        for (unsigned i = 16; i < 64; ++i) {
            const auto a = w[i-15], b = w[i-2];
            w[i] = w[i-16] + (rotr(a,7) ^ rotr(a,18) ^ (a >> 3)) +
                   w[i-7] + (rotr(b,17) ^ rotr(b,19) ^ (b >> 10));
        }
#ifdef XHDFE_CACHE_SHA_INTRINSICS
#ifdef __AVX__
        // SHA uses legacy SSE encoding; clear upper AVX lanes once before it.
        _mm256_zeroupper();
#endif
        // SHA256RNDS2 uses high-to-low A,B,E,F and C,D,G,H lanes.
        const std::array<std::uint32_t,4> first{{state_[5],state_[4],state_[1],state_[0]}};
        const std::array<std::uint32_t,4> second{{state_[7],state_[6],state_[3],state_[2]}};
        __m128i abef = _mm_loadu_si128(reinterpret_cast<const __m128i*>(first.data()));
        __m128i cdgh = _mm_loadu_si128(reinterpret_cast<const __m128i*>(second.data()));
        for (unsigned i = 0; i < 64; i += 4) {
            const __m128i message = _mm_add_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(w+i)),
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(constants+i)));
            cdgh = _mm_sha256rnds2_epu32(cdgh, abef, message);
            abef = _mm_sha256rnds2_epu32(abef, cdgh, _mm_srli_si128(message,8));
        }
        std::array<std::uint32_t,4> out0, out1;
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out0.data()), abef);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out1.data()), cdgh);
        state_[0]+=out0[3]; state_[1]+=out0[2]; state_[2]+=out1[3]; state_[3]+=out1[2];
        state_[4]+=out0[1]; state_[5]+=out0[0]; state_[6]+=out1[1]; state_[7]+=out1[0];
#else
        auto a=state_[0], b=state_[1], c=state_[2], d=state_[3];
        auto e=state_[4], f=state_[5], g=state_[6], h=state_[7];
        for (unsigned i = 0; i < 64; ++i) {
            const auto first = h + (rotr(e,6) ^ rotr(e,11) ^ rotr(e,25)) +
                ((e & f) ^ (~e & g)) + constants[i] + w[i];
            const auto second = (rotr(a,2) ^ rotr(a,13) ^ rotr(a,22)) +
                ((a & b) ^ (a & c) ^ (b & c));
            h=g; g=f; f=e; e=d+first; d=c; c=b; b=a; a=first+second;
        }
        state_[0]+=a; state_[1]+=b; state_[2]+=c; state_[3]+=d;
        state_[4]+=e; state_[5]+=f; state_[6]+=g; state_[7]+=h;
#endif
    }
public:
    void update(const void* data, std::size_t length) {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        bytes_ += length;
        while (length) {
            const auto count = std::min(length, buffer_.size() - used_);
            std::memcpy(buffer_.data() + used_, bytes, count);
            bytes += count; used_ += count; length -= count;
            if (used_ == buffer_.size()) {
                compress();
                used_ = 0;
            }
        }
    }
    void word(std::uint64_t value) {
        std::uint8_t bytes[8];
        for (unsigned i = 0; i < 8; ++i) bytes[i] = static_cast<std::uint8_t>(value >> (8*i));
        update(bytes, sizeof(bytes));
    }
    std::array<std::uint8_t, 32> finish() const {
        CacheSha256 hash = *this;
        const std::uint64_t bits = bytes_ * 8;
        const std::uint8_t marker = 0x80, zero = 0;
        hash.update(&marker, 1);
        while (hash.used_ != 56) hash.update(&zero, 1);
        std::uint8_t length[8];
        for (unsigned i = 0; i < 8; ++i) length[i] = static_cast<std::uint8_t>(bits >> (56-8*i));
        hash.update(length, sizeof(length));
        std::array<std::uint8_t, 32> result{};
        for (unsigned i = 0; i < 8; ++i)
            for (unsigned j = 0; j < 4; ++j)
                result[4*i+j] = static_cast<std::uint8_t>(hash.state_[i] >> (24-8*j));
        return result;
    }
};

// Typed, length-delimited nodes. The fixed block graph does not depend on the
// OpenMP team size; only SHA digests, never raw inputs, enter the cache header.
class CacheDigestBuilder {
    struct Node {
        bool blob;
        std::uint64_t value;
        const std::uint8_t* data;
        std::size_t size;
    };
    std::uint64_t domain_;
    std::vector<Node> nodes_;
public:
    static constexpr std::size_t block_size = 1U << 20;
    explicit CacheDigestBuilder(std::uint64_t domain) : domain_(domain) {}
    void word(std::uint64_t value) { nodes_.push_back({false,value,nullptr,0}); }
    void data(const void* value, std::size_t size) {
        nodes_.push_back({true,0,static_cast<const std::uint8_t*>(value),size});
    }
    std::array<std::uint8_t, 32> finish(int threads = 1, ParallelWorkObserver* observer = nullptr) const {
        struct Job { const std::uint8_t* data; std::size_t size; };
        std::vector<Job> jobs;
        std::vector<std::size_t> counts(nodes_.size(),0);
        for (std::size_t i = 0; i < nodes_.size(); ++i) {
            const auto& node = nodes_[i];
            if (!node.blob) continue;
            counts[i] = node.size ? 1 + (node.size - 1) / block_size : 1;
            for (std::size_t j = 0; j < counts[i]; ++j) {
                const std::size_t offset = j * block_size;
                jobs.push_back({node.data ? node.data + offset : nullptr,
                                std::min(block_size, node.size - offset)});
            }
        }
        std::vector<std::array<std::uint8_t,32>> leaves(jobs.size());
        const int workers = std::max(1, threads);
        if (!jobs.empty()) {
            if (observer) observer->begin_region(workers);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(workers) if(workers > 1)
#endif
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(jobs.size()); ++i) {
                if (observer) observer->observe_work();
                CacheSha256 leaf;
                const std::uint8_t tag = 0x43;
                leaf.update(&tag,1);
                leaf.word(jobs[static_cast<std::size_t>(i)].size);
                leaf.update(jobs[static_cast<std::size_t>(i)].data, jobs[static_cast<std::size_t>(i)].size);
                leaves[static_cast<std::size_t>(i)] = leaf.finish();
            }
            if (observer) observer->end_region();
        }
        CacheSha256 root;
        constexpr char prefix[] = "xhdfe-cache-sha256-blocks-v1";
        root.update(prefix,sizeof(prefix)-1);
        root.word(domain_);
        root.word(nodes_.size());
        std::size_t next = 0;
        for (std::size_t i = 0; i < nodes_.size(); ++i) {
            const auto& node = nodes_[i];
            const std::uint8_t tag = node.blob ? 1 : 0;
            root.update(&tag,1);
            if (!node.blob) {
                root.word(node.value);
            } else {
                root.word(node.size);
                root.word(counts[i]);
                for (std::size_t j = 0; j < counts[i]; ++j) {
                    root.update(leaves[next].data(),leaves[next].size());
                    ++next;
                }
            }
        }
        return root.finish();
    }
};

}  // namespace detail
}  // namespace hdfe
