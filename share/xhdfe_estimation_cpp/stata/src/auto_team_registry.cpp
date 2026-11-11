// Per-user registry of automatic OpenMP teams; see
// include/hdfe/auto_team_registry.hpp for the contract and the file layout.

#include "hdfe/auto_team_registry.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#define XHDFE_AUTO_TEAM_REGISTRY_POSIX 1
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#endif

namespace hdfe {
namespace detail {

int auto_team_grant(int requested, int legacy, int capacity, int sessions,
                    long long leased) noexcept {
    const int wanted = std::max(1, requested);
    const int floor_team = std::max(1, legacy);
    if (floor_team >= wanted) {
        // Nothing to coordinate, and a legacy team above the request must
        // never raise it.
        return wanted;
    }
    const long long processors = std::max(1, capacity);
    const long long share = processors / std::max(1, sessions);
    const long long free_processors = processors - std::max(0LL, leased);
    const long long bounded =
        std::min<long long>(wanted, std::min(share, free_processors));
    return static_cast<int>(std::max<long long>(floor_team, bounded));
}

#if defined(XHDFE_AUTO_TEAM_REGISTRY_POSIX)

namespace {

struct RegistryHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t record_size;
    std::uint64_t next_lease;
    std::uint64_t reserved;
};
static_assert(sizeof(RegistryHeader) == 32,
              "the registry header layout is fixed at 32 bytes");

constexpr char kRegistryMagic[8] = {'X', 'H', 'D', 'F', 'E', 'T', 'R', '1'};
constexpr std::uint32_t kRegistryVersion = 1;
constexpr std::size_t kMaxRecords = 1024;
// Critical sections take microseconds; never wait long for a lock holder
// that may be stopped. A fit gives up after about 20 ms (no coordination);
// a release keeps trying for about 1 s before leaving the lease to the next
// operation of this process.
constexpr int kAcquireLockAttempts = 200;
constexpr int kReleaseLockAttempts = 10000;
constexpr long kLockPauseNanoseconds = 100000;
constexpr long long kMaxRecordedTeam = 1 << 20;

// The outermost registration alive on this thread (see the class comment).
thread_local AutoTeamRegistration* t_outermost = nullptr;

// Leases whose release could not be written. Any later registry operation of
// this process removes them. Lock-free so that it stays usable after fork().
constexpr std::size_t kPendingReleaseSlots = 16;
std::atomic<std::uint64_t> g_pending_release[kPendingReleaseSlots];

void remember_pending_release(std::uint64_t lease) noexcept {
    for (auto& slot : g_pending_release) {
        std::uint64_t expected = 0;
        if (slot.compare_exchange_strong(expected, lease)) {
            return;
        }
    }
}

bool pending_release(std::uint64_t lease) noexcept {
    if (lease == 0) {
        return false;
    }
    for (const auto& slot : g_pending_release) {
        if (slot.load() == lease) {
            return true;
        }
    }
    return false;
}

void forget_pending_release(std::uint64_t lease) noexcept {
    for (auto& slot : g_pending_release) {
        std::uint64_t expected = lease;
        if (slot.compare_exchange_strong(expected, 0)) {
            return;
        }
    }
}

bool blank(char c) noexcept {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

bool equals_ignoring_case(const char* begin, const char* end,
                          const char* word) noexcept {
    for (; begin != end && *word != '\0'; ++begin, ++word) {
        char c = *begin;
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
        if (c != *word) {
            return false;
        }
    }
    return begin == end && *word == '\0';
}

bool switched_off(const char* raw) noexcept {
    if (raw == nullptr) {
        return false;
    }
    const char* begin = raw;
    while (blank(*begin)) {
        ++begin;
    }
    const char* end = begin + std::strlen(begin);
    while (end > begin && blank(end[-1])) {
        --end;
    }
    return equals_ignoring_case(begin, end, "0") ||
           equals_ignoring_case(begin, end, "off") ||
           equals_ignoring_case(begin, end, "false") ||
           equals_ignoring_case(begin, end, "no");
}

std::string usable_directory(const char* raw) {
    if (raw == nullptr || raw[0] != '/') {
        return std::string();
    }
    std::string dir(raw);
    while (dir.size() > 1 && dir.back() == '/') {
        dir.pop_back();
    }
    return dir;
}

std::string registry_path() {
    std::string dir = usable_directory(std::getenv("XHDFE_AUTO_TEAM_REGISTRY_DIR"));
    if (dir.empty()) {
        struct stat st;
        if (::stat("/dev/shm", &st) == 0 && S_ISDIR(st.st_mode)) {
            dir = "/dev/shm";
        } else {
            dir = usable_directory(std::getenv("TMPDIR"));
            if (dir.empty()) {
                dir = "/tmp";
            }
        }
    }
    if (dir != "/") {
        dir += '/';
    }
    return dir + "xhdfe-teams-" +
           std::to_string(static_cast<unsigned long long>(::getuid()));
}

// Chosen once per process.
const std::string& cached_registry_path() {
    static const std::string path = registry_path();
    return path;
}

// Opaque process start time; only equality is used. 0 when unknown.
std::uint64_t process_start_time(pid_t pid) noexcept {
#if defined(__linux__)
    char path[48];
    std::snprintf(path, sizeof path, "/proc/%ld/stat", static_cast<long>(pid));
    const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return 0;
    }
    char buffer[2048];
    ssize_t got = 0;
    do {
        got = ::read(fd, buffer, sizeof buffer - 1);
    } while (got < 0 && errno == EINTR);
    ::close(fd);
    if (got <= 0) {
        return 0;
    }
    buffer[got] = '\0';
    // Field 2, the command name, is in parentheses and may itself hold
    // blanks or parentheses: count the fields from the last ')'. The start
    // time is field 22 (proc(5)).
    const char* cursor = std::strrchr(buffer, ')');
    if (cursor == nullptr) {
        return 0;
    }
    ++cursor;
    int field = 2;
    while (*cursor != '\0') {
        while (*cursor == ' ') {
            ++cursor;
        }
        if (*cursor == '\0') {
            break;
        }
        ++field;
        if (field == 22) {
            std::uint64_t value = 0;
            bool digits = false;
            while (*cursor >= '0' && *cursor <= '9') {
                value = value * 10 + static_cast<std::uint64_t>(*cursor - '0');
                digits = true;
                ++cursor;
            }
            return digits ? value : 0;
        }
        while (*cursor != '\0' && *cursor != ' ') {
            ++cursor;
        }
    }
    return 0;
#else
    // macOS: no start time is read, so liveness is kill(pid, 0) alone. A
    // reused pid can only keep a record alive, which never raises a team.
    (void)pid;
    return 0;
#endif
}

std::atomic<long> g_self_pid{0};
std::atomic<std::uint64_t> g_self_start{0};

// Cached per pid, so a forked child reads its own start time.
std::uint64_t self_start_time(pid_t self) noexcept {
    if (g_self_pid.load() == static_cast<long>(self)) {
        return g_self_start.load();
    }
    const std::uint64_t start = process_start_time(self);
    g_self_start.store(start);
    g_self_pid.store(static_cast<long>(self));
    return start;
}

bool process_exists(pid_t pid) noexcept {
    if (::kill(pid, 0) == 0) {
        return true;
    }
    // ESRCH: no such process. EPERM: the number now belongs to another
    // user's process, so the process of this user that wrote the record is
    // gone.
    return errno != ESRCH && errno != EPERM;
}

bool record_alive(const AutoTeamRecord& record, pid_t self,
                  std::uint64_t self_start) noexcept {
    if (record.pid <= 0) {
        return false;
    }
    if (record.kind != kAutoTeamIntent && record.kind != kAutoTeamLease) {
        return false;
    }
    if (record.pid == static_cast<std::int32_t>(self)) {
        return record.start_time == 0 || self_start == 0 ||
               record.start_time == self_start;
    }
    if (!process_exists(static_cast<pid_t>(record.pid))) {
        return false;
    }
    if (record.start_time == 0) {
        return true;
    }
    const std::uint64_t start =
        process_start_time(static_cast<pid_t>(record.pid));
    return start == 0 || start == record.start_time;
}

bool read_exact(int fd, void* buffer, std::size_t size, off_t offset) noexcept {
    char* cursor = static_cast<char*>(buffer);
    while (size > 0) {
        const ssize_t got = ::pread(fd, cursor, size, offset);
        if (got < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (got == 0) {
            return false;
        }
        cursor += got;
        size -= static_cast<std::size_t>(got);
        offset += got;
    }
    return true;
}

bool write_exact(int fd, const void* buffer, std::size_t size,
                 off_t offset) noexcept {
    const char* cursor = static_cast<const char*>(buffer);
    while (size > 0) {
        const ssize_t put = ::pwrite(fd, cursor, size, offset);
        if (put < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (put == 0) {
            return false;
        }
        cursor += put;
        size -= static_cast<std::size_t>(put);
        offset += put;
    }
    return true;
}

bool lock_exclusive(int fd, int attempts) noexcept {
    for (int attempt = 0; attempt < attempts; ++attempt) {
        if (::flock(fd, LOCK_EX | LOCK_NB) == 0) {
            return true;
        }
        if (errno != EWOULDBLOCK && errno != EAGAIN && errno != EINTR) {
            return false;
        }
        struct timespec delay;
        delay.tv_sec = 0;
        delay.tv_nsec = kLockPauseNanoseconds;
        ::nanosleep(&delay, nullptr);
    }
    return false;
}

// The registry file, opened and locked for one read-modify-write.
class RegistryFile {
public:
    RegistryFile() = default;
    ~RegistryFile() { close(); }
    RegistryFile(const RegistryFile&) = delete;
    RegistryFile& operator=(const RegistryFile&) = delete;

    bool open_locked(const std::string& path, int lock_attempts) noexcept {
        fd_ = ::open(path.c_str(),
                     O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK,
                     S_IRUSR | S_IWUSR);
        if (fd_ < 0) {
            return false;
        }
        struct stat st;
        if (::fstat(fd_, &st) != 0 || !S_ISREG(st.st_mode) ||
            st.st_uid != ::getuid() || (st.st_mode & 077) != 0 ||
            st.st_nlink != 1 || !lock_exclusive(fd_, lock_attempts)) {
            close();
            return false;
        }
        return true;
    }

    void close() noexcept {
        if (fd_ >= 0) {
            ::flock(fd_, LOCK_UN);
            ::close(fd_);
            fd_ = -1;
        }
    }

    // False for a file this version does not own (foreign magic, version or
    // record size, or too large); such a file is never written.
    bool load(std::vector<AutoTeamRecord>& records) {
        records.clear();
        struct stat st;
        if (::fstat(fd_, &st) != 0) {
            return false;
        }
        size_ = st.st_size;
        const off_t header_size = static_cast<off_t>(sizeof(RegistryHeader));
        if (st.st_size < header_size) {
            // A new file, or a header cut short: start an empty registry.
            std::memset(&header_, 0, sizeof header_);
            std::memcpy(header_.magic, kRegistryMagic, sizeof header_.magic);
            header_.version = kRegistryVersion;
            header_.record_size = sizeof(AutoTeamRecord);
            header_.next_lease = 1;
            return true;
        }
        const off_t max_size =
            header_size +
            static_cast<off_t>(kMaxRecords * sizeof(AutoTeamRecord));
        if (st.st_size > max_size ||
            !read_exact(fd_, &header_, sizeof header_, 0) ||
            std::memcmp(header_.magic, kRegistryMagic, sizeof header_.magic) != 0 ||
            header_.version != kRegistryVersion ||
            header_.record_size != sizeof(AutoTeamRecord)) {
            return false;
        }
        // A trailing partial record is ignored here and trimmed by store().
        const std::size_t count =
            static_cast<std::size_t>(st.st_size - header_size) /
            sizeof(AutoTeamRecord);
        records.resize(count);
        if (count > 0 &&
            !read_exact(fd_, records.data(), count * sizeof(AutoTeamRecord),
                        header_size)) {
            return false;
        }
        if (header_.next_lease == 0) {
            header_.next_lease = 1;
        }
        return true;
    }

    std::uint64_t take_lease_number() noexcept {
        const std::uint64_t lease = header_.next_lease;
        header_.next_lease = lease == UINT64_MAX ? 1 : lease + 1;
        return lease;
    }

    // Writes every slot (freed ones as zeros), then trims trailing free
    // slots; a failed trim only leaves zeros or stale records behind.
    bool store(const std::vector<AutoTeamRecord>& records) noexcept {
        const off_t header_size = static_cast<off_t>(sizeof(RegistryHeader));
        if (!write_exact(fd_, &header_, sizeof header_, 0) ||
            (!records.empty() &&
             !write_exact(fd_, records.data(),
                          records.size() * sizeof(AutoTeamRecord),
                          header_size))) {
            return false;
        }
        std::size_t used = records.size();
        while (used > 0 && records[used - 1].pid == 0) {
            --used;
        }
        const off_t wanted =
            header_size + static_cast<off_t>(used * sizeof(AutoTeamRecord));
        const off_t written =
            header_size +
            static_cast<off_t>(records.size() * sizeof(AutoTeamRecord));
        if (std::max(size_, written) > wanted &&
            ::ftruncate(fd_, wanted) == 0) {
            size_ = wanted;
        }
        return true;
    }

private:
    int fd_ = -1;
    off_t size_ = 0;
    RegistryHeader header_{};
};

struct Census {
    int other_sessions = 0;
    long long other_leased = 0;
    std::size_t own = 0;  // index of this registration's record; size() if none
};

// Discards dead or malformed records and this process's pending releases,
// and counts the live records of others.
Census sweep(std::vector<AutoTeamRecord>& records, std::uint64_t own_lease,
             pid_t self, std::uint64_t self_start,
             std::vector<std::uint64_t>& released) {
    Census census;
    census.own = records.size();
    for (std::size_t i = 0; i < records.size(); ++i) {
        AutoTeamRecord& record = records[i];
        if (record.pid == 0) {
            continue;
        }
        const bool same_process = record.pid == static_cast<std::int32_t>(self);
        if (same_process && own_lease != 0 && record.lease == own_lease) {
            census.own = i;
            continue;
        }
        if (same_process && pending_release(record.lease)) {
            released.push_back(record.lease);
            record = AutoTeamRecord{};
            continue;
        }
        if (!record_alive(record, self, self_start)) {
            record = AutoTeamRecord{};
            continue;
        }
        ++census.other_sessions;
        if (record.kind == kAutoTeamLease) {
            census.other_leased += std::min<long long>(
                std::max<long long>(record.team, 0), kMaxRecordedTeam);
        }
    }
    return census;
}

}  // namespace

bool auto_team_coordination_enabled() noexcept {
    static const bool enabled =
        !switched_off(std::getenv("XHDFE_AUTO_TEAM_COORDINATION"));
    return enabled;
}

AutoTeamRegistration::AutoTeamRegistration() noexcept
    : owner_(t_outermost != nullptr ? t_outermost : this) {
    if (owner_ == this) {
        t_outermost = this;
    }
}

AutoTeamRegistration::~AutoTeamRegistration() {
    if (owner_ == this) {
        release();
        t_outermost = nullptr;
    }
}

AutoTeamGrant AutoTeamRegistration::acquire(std::uint32_t kind, int requested,
                                            int legacy, int capacity,
                                            bool explicit_request) noexcept {
    AutoTeamGrant grant;
    grant.team = std::max(1, requested);
    if (owner_ != this) {
        return owner_->acquire(kind, requested, legacy, capacity,
                               explicit_request);
    }
    // A one-thread phase with nothing recorded yet is not recorded.
    if (!auto_team_coordination_enabled() || (lease_ == 0 && grant.team <= 1)) {
        return grant;
    }
    try {
        RegistryFile file;
        std::vector<AutoTeamRecord> records;
        if (!file.open_locked(cached_registry_path(), kAcquireLockAttempts) ||
            !file.load(records)) {
            return grant;
        }
        const pid_t self = ::getpid();
        const std::uint64_t self_start = self_start_time(self);
        std::vector<std::uint64_t> released;
        Census census = sweep(records, lease_, self, self_start, released);

        int team = grant.team;
        if (!explicit_request) {
            team = auto_team_grant(grant.team, legacy, capacity,
                                   census.other_sessions + 1,
                                   census.other_leased);
        }
        if (census.own == records.size()) {
            std::size_t slot = 0;
            while (slot < records.size() && records[slot].pid != 0) {
                ++slot;
            }
            if (slot == records.size()) {
                if (records.size() >= kMaxRecords) {
                    return grant;
                }
                records.push_back(AutoTeamRecord{});
            }
            census.own = slot;
            lease_ = file.take_lease_number();
        }
        AutoTeamRecord& mine = records[census.own];
        mine.pid = static_cast<std::int32_t>(self);
        mine.team = team;
        mine.start_time = self_start;
        mine.lease = lease_;
        mine.kind = kind;
        mine.flags = explicit_request ? kAutoTeamExplicitFlag : 0u;
        if (!file.store(records)) {
            // The record may be partly written: release() still clears it.
            return grant;
        }
        for (const std::uint64_t lease : released) {
            forget_pending_release(lease);
        }
        grant.reduced = team < grant.team;
        grant.team = team;
        return grant;
    } catch (...) {
        return grant;
    }
}

void AutoTeamRegistration::release() noexcept {
    if (owner_ != this || lease_ == 0) {
        return;
    }
    const std::uint64_t lease = lease_;
    lease_ = 0;
    try {
        RegistryFile file;
        std::vector<AutoTeamRecord> records;
        if (file.open_locked(cached_registry_path(), kReleaseLockAttempts) &&
            file.load(records)) {
            const pid_t self = ::getpid();
            std::vector<std::uint64_t> released;
            const Census census = sweep(records, lease, self,
                                        self_start_time(self), released);
            if (census.own < records.size()) {
                records[census.own] = AutoTeamRecord{};
            }
            if (file.store(records)) {
                for (const std::uint64_t done : released) {
                    forget_pending_release(done);
                }
                return;
            }
        }
    } catch (...) {
    }
    remember_pending_release(lease);
}

#else  // no registry on this platform: every grant is the request

bool auto_team_coordination_enabled() noexcept {
    return false;
}

AutoTeamRegistration::AutoTeamRegistration() noexcept : owner_(this) {}

AutoTeamRegistration::~AutoTeamRegistration() = default;

AutoTeamGrant AutoTeamRegistration::acquire(std::uint32_t, int requested, int,
                                            int, bool) noexcept {
    AutoTeamGrant grant;
    grant.team = std::max(1, requested);
    return grant;
}

void AutoTeamRegistration::release() noexcept {
    if (owner_ == this && lease_ != 0) {
        lease_ = 0;
    }
}

#endif

AutoTeamGrant AutoTeamRegistration::acquire_intent(
    int requested, int legacy, int capacity, bool explicit_request) noexcept {
    return acquire(kAutoTeamIntent, requested, legacy, capacity,
                   explicit_request);
}

AutoTeamGrant AutoTeamRegistration::acquire_lease(
    int requested, int legacy, int capacity, bool explicit_request) noexcept {
    return acquire(kAutoTeamLease, requested, legacy, capacity,
                   explicit_request);
}

}  // namespace detail
}  // namespace hdfe
