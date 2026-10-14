#ifndef HDFE_AUDIT_ENVIRONMENT_HPP
#define HDFE_AUDIT_ENVIRONMENT_HPP

#include <array>
#include <cerrno>
#include <cctype>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace hdfe {
namespace detail {

// These switches were introduced with the performance candidate. A snapshot
// belongs to one fit, including its CPU and CUDA certificate work.
struct AuditEnvironment {
    static constexpr std::size_t kDynamicFallbackFirst = 10;
    static constexpr std::size_t kFlags = 13;
    std::array<const char*, kFlags> names{{
        "XHDFE_CERTIFICATE_TEAM", "XHDFE_GI_FUSE_OPERATOR_REGIONS",
        "XHDFE_HUGE_PAGES", "XHDFE_MLSMR_AC_ARENA",
        "XHDFE_MLSMR_FACTOR_PRODUCTS", "XHDFE_MLSMR_LOCKSTEP",
        "XHDFE_MLSMR_LOCKSTEP_CHECK", "XHDFE_MLSMR_LOCKSTEP_SCAN",
        "XHDFE_MLSMR_STATE_TRACE", "XHDFE_PACKED_FUSED",
        "XHDFE_MLSMR_PARALLEL_APPLY", "XHDFE_MLSMR_DISJOINT_DIRECT_APPLY",
        "XHDFE_MLSMR_CSR_DOMAIN"}};
    std::array<bool, kFlags> flags{{true, false, false, true, true,
                                     true, false, false, false, true,
                                     true, false, true}};
    std::array<bool, kFlags> supplied{};
    std::string build = "assembled";
    std::string edge_order = "sorted";
    int packed_block_bytes = 0;

    static std::string value(const char* name) {
        const char* raw = std::getenv(name);
        if (!raw) return {};  // Unset uses the caller's default; empty fails.
        std::string text(raw);
        const std::size_t first = text.find_first_not_of(" \t\n\r\f\v");
        if (first == std::string::npos) {
            throw std::runtime_error(std::string(name) + " has an empty value");
        }
        const std::size_t last = text.find_last_not_of(" \t\n\r\f\v");
        text = text.substr(first, last - first + 1);
        for (char& c : text) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        return text;
    }

    static bool boolean(const char* name, bool fallback) {
        const std::string v = value(name);
        if (v.empty()) return fallback;
        if (v == "1" || v == "on" || v == "true" || v == "yes") return true;
        if (v == "0" || v == "off" || v == "false" || v == "no") return false;
        throw std::runtime_error(std::string(name) + " has invalid value '" + v + "'");
    }

    static std::string choice(const char* name, const char* fallback,
                              const char* first, const char* second,
                              const char* third = nullptr) {
        std::string v = value(name);
        if (v.empty()) return fallback;
        if (v == first || v == second || (third && v == third)) return v;
        throw std::runtime_error(std::string(name) + " has invalid value '" + v + "'");
    }

    static AuditEnvironment read() {
        AuditEnvironment out;
        for (std::size_t i = 0; i < kFlags; ++i) {
            out.supplied[i] = std::getenv(out.names[i]) != nullptr;
            out.flags[i] = boolean(out.names[i], out.flags[i]);
        }
        out.build = choice("XHDFE_MLSMR_BUILD", "assembled",
                           "assembled", "reference", "legacy");
        out.edge_order = choice("XHDFE_MLSMR_EDGE_ORDER", "sorted",
                                "sorted", "hash");
        const std::string v = value("XHDFE_PACKED_BLOCK_BYTES");
        if (!v.empty()) {
            errno = 0;
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(v.c_str(), &end, 10);
            if (errno || end == v.c_str() || *end != '\0' || v[0] == '-' ||
                parsed > static_cast<unsigned long long>(INT_MAX)) {
                throw std::runtime_error(
                    "XHDFE_PACKED_BLOCK_BYTES has invalid value '" + v + "'");
            }
            out.packed_block_bytes = static_cast<int>(parsed);
        }
        return out;
    }

    bool find_flag(const char* name, bool fallback, bool* found) const noexcept {
        for (std::size_t i = 0; i < kFlags; ++i) {
            if (std::strcmp(name, names[i]) == 0) {
                *found = true;
                return i >= kDynamicFallbackFirst && !supplied[i]
                           ? fallback : flags[i];
            }
        }
        *found = false;
        return false;
    }
};

inline const AuditEnvironment*& active_audit_environment() noexcept {
    static thread_local const AuditEnvironment* current = nullptr;
    return current;
}

class ScopedAuditEnvironment {
public:
    ScopedAuditEnvironment() : previous_(active_audit_environment()) {
        if (!previous_) {
            owned_ = AuditEnvironment::read();
            active_audit_environment() = &owned_;
        }
    }
    ~ScopedAuditEnvironment() { active_audit_environment() = previous_; }
    ScopedAuditEnvironment(const ScopedAuditEnvironment&) = delete;
    ScopedAuditEnvironment& operator=(const ScopedAuditEnvironment&) = delete;

private:
    AuditEnvironment owned_;
    const AuditEnvironment* previous_;
};

class ScopedBorrowedAuditEnvironment {
public:
    explicit ScopedBorrowedAuditEnvironment(const AuditEnvironment* inherited)
        : previous_(active_audit_environment()) {
        if (inherited) active_audit_environment() = inherited;
    }
    ~ScopedBorrowedAuditEnvironment() { active_audit_environment() = previous_; }

private:
    const AuditEnvironment* previous_;
};

inline bool audit_flag(const char* name, bool fallback, bool* found) {
    const AuditEnvironment* current = active_audit_environment();
    if (current) return current->find_flag(name, fallback, found);
    static const AuditEnvironment defaults;
    (void)defaults.find_flag(name, fallback, found);
    if (!*found) return fallback;
    const AuditEnvironment snapshot = AuditEnvironment::read();
    const bool result = snapshot.find_flag(name, fallback, found);
    return *found ? result : fallback;
}

inline std::string audit_choice(const char* name) {
    const AuditEnvironment* current = active_audit_environment();
    if (!current) {
        const AuditEnvironment snapshot = AuditEnvironment::read();
        return std::strcmp(name, "XHDFE_MLSMR_BUILD") == 0
                   ? snapshot.build : snapshot.edge_order;
    }
    return std::strcmp(name, "XHDFE_MLSMR_BUILD") == 0
               ? current->build : current->edge_order;
}

inline int audit_packed_block_bytes() {
    const AuditEnvironment* current = active_audit_environment();
    return current ? current->packed_block_bytes
                   : AuditEnvironment::read().packed_block_bytes;
}

}  // namespace detail
}  // namespace hdfe

#endif
