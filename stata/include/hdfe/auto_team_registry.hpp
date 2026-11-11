#ifndef HDFE_AUTO_TEAM_REGISTRY_HPP
#define HDFE_AUTO_TEAM_REGISTRY_HPP

#include <cstdint>

namespace hdfe {
namespace detail {

// Automatic OpenMP teams shared between the fits of one user.
//
// A fit whose team has more than one thread records it in a small per-user
// file: an intent when the fit starts (first team resolution) and a lease
// when its absorption starts (second resolution). An automatic fit is then
// granted
//
//     max(legacy, min(requested, capacity / sessions, capacity - leased))
//
// where `requested` is the team the automatic rule resolves, `legacy` the team
// the 2.30.1 rule (one thread per 500 000 rows) resolves under the same caps,
// `sessions` the live intents and leases including this fit, and `leased` the
// sum of the teams of the other live leases. A grant never exceeds `requested`
// and never falls below `legacy`. An explicit request is never reduced; it is
// only recorded so that automatic fits see it.
//
// Any failure (no usable directory, no lock within a short wait, a malformed
// or foreign file, an unsupported platform) means no coordination: the grant
// is `requested`. XHDFE_AUTO_TEAM_COORDINATION=0 (also off/false/no), read
// once per process, turns the registry off. Supported on Linux and macOS;
// every other platform compiles this unit as a no-op.
//
// File: "xhdfe-teams-<uid>" in XHDFE_AUTO_TEAM_REGISTRY_DIR when set (tests),
// otherwise /dev/shm when it is a directory, otherwise $TMPDIR, otherwise
// /tmp. Created with mode 0600 and opened without following links; ignored
// unless it is a regular file with one link, owned by the calling user and
// without group or other permissions. Every read-modify-write holds an
// exclusive flock(). Native byte order:
//
//   header, 32 bytes: magic "XHDFETR1", u32 version (1), u32 record size
//                     (32), u64 next lease number, u64 reserved (0)
//   record, 32 bytes: AutoTeamRecord below; pid 0 marks a free slot
//
// A record whose process no longer exists, or whose process start time no
// longer matches (Linux: /proc/<pid>/stat field 22, clock ticks since boot),
// is discarded by the next reader. On macOS, and wherever the start time
// cannot be read, liveness is kill(pid, 0) alone: a reused pid can only keep
// a record alive, which may lower a team towards legacy but never raise one.

constexpr std::uint32_t kAutoTeamIntent = 1;
constexpr std::uint32_t kAutoTeamLease = 2;
constexpr std::uint32_t kAutoTeamExplicitFlag = 1;

struct AutoTeamRecord {
    std::int32_t pid;          // 0 = free slot
    std::int32_t team;         // threads granted to this phase of the fit
    std::uint64_t start_time;  // process start time; 0 = unknown
    std::uint64_t lease;       // number unique within the file
    std::uint32_t kind;        // kAutoTeamIntent or kAutoTeamLease
    std::uint32_t flags;       // kAutoTeamExplicitFlag for explicit requests
};
static_assert(sizeof(AutoTeamRecord) == 32,
              "the registry record layout is fixed at 32 bytes");

struct AutoTeamGrant {
    int team = 1;          // team to use
    bool reduced = false;  // true when the registry lowered the team
};

// The grant formula on its own, for an automatic request. Never above
// max(1, requested); never below min(legacy, requested) and 1.
int auto_team_grant(int requested, int legacy, int capacity, int sessions,
                    long long leased) noexcept;

// False when XHDFE_AUTO_TEAM_COORDINATION disables the registry or the
// platform does not support it.
bool auto_team_coordination_enabled() noexcept;

// One fit's registration. Declare it on the stack before the first team
// resolution; the destructor releases the record on every exit path. A
// registration created while another one is alive on the same thread (a fit
// that calls fit() again) acts on the outer record instead of adding one.
class AutoTeamRegistration {
public:
    AutoTeamRegistration() noexcept;
    ~AutoTeamRegistration();

    AutoTeamRegistration(const AutoTeamRegistration&) = delete;
    AutoTeamRegistration& operator=(const AutoTeamRegistration&) = delete;
    AutoTeamRegistration(AutoTeamRegistration&&) = delete;
    AutoTeamRegistration& operator=(AutoTeamRegistration&&) = delete;

    // `capacity` is the number of logical processors the fit may use.
    AutoTeamGrant acquire_intent(int requested, int legacy, int capacity,
                                 bool explicit_request) noexcept;
    AutoTeamGrant acquire_lease(int requested, int legacy, int capacity,
                                bool explicit_request) noexcept;
    void release() noexcept;

private:
    AutoTeamGrant acquire(std::uint32_t kind, int requested, int legacy,
                          int capacity, bool explicit_request) noexcept;

    AutoTeamRegistration* owner_ = nullptr;
    std::uint64_t lease_ = 0;
};

}  // namespace detail
}  // namespace hdfe

#endif  // HDFE_AUTO_TEAM_REGISTRY_HPP
