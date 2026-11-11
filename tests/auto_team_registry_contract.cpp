// Contract of the per-user team registry (src/auto_team_registry.cpp): the
// grant formula, intent/lease/release, nesting, live, stale and explicit
// records of other processes, files that must be ignored, a held lock, a
// release that could not be written, and the switch. Every file lives in a
// private directory named through XHDFE_AUTO_TEAM_REGISTRY_DIR; /dev/shm is
// never touched.

#include "hdfe/auto_team_registry.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <cerrno>
#include <chrono>
#include <fcntl.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

using hdfe::detail::AutoTeamGrant;
using hdfe::detail::AutoTeamRecord;
using hdfe::detail::AutoTeamRegistration;
using hdfe::detail::auto_team_grant;
using hdfe::detail::kAutoTeamExplicitFlag;
using hdfe::detail::kAutoTeamIntent;
using hdfe::detail::kAutoTeamLease;

int g_failures = 0;
std::string g_dir;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) {
        ++g_failures;
    }
}

struct Header {
    char magic[8];
    std::uint32_t version;
    std::uint32_t record_size;
    std::uint64_t next_lease;
    std::uint64_t reserved;
};
static_assert(sizeof(Header) == 32, "registry header layout");

std::string registry_file() {
    return g_dir + "/xhdfe-teams-" +
           std::to_string(static_cast<unsigned long long>(::getuid()));
}

std::string side_file() {
    return g_dir + "/side-file";
}

// Independent of the unit's parser: sscanf over the fields after the last ')'.
// 0 on macOS, where the unit uses pid-only liveness.
std::uint64_t start_time_of(pid_t pid) {
#if defined(__linux__)
    char path[64];
    std::snprintf(path, sizeof path, "/proc/%ld/stat", static_cast<long>(pid));
    FILE* file = std::fopen(path, "r");
    if (file == nullptr) {
        return 0;
    }
    char buffer[4096];
    const std::size_t got = std::fread(buffer, 1, sizeof buffer - 1, file);
    std::fclose(file);
    buffer[got] = '\0';
    const char* cursor = std::strrchr(buffer, ')');
    if (cursor == nullptr) {
        return 0;
    }
    unsigned long long start = 0;
    // Fields 3 to 21 are skipped; field 22 is the start time (proc(5)).
    if (std::sscanf(cursor + 1,
                    " %*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %*u %*u"
                    " %*d %*d %*d %*d %*d %*d %llu",
                    &start) != 1) {
        return 0;
    }
    return start;
#else
    (void)pid;
    return 0;
#endif
}

std::vector<char> file_bytes(const std::string& path) {
    std::vector<char> bytes;
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        return bytes;
    }
    char buffer[4096];
    std::size_t got = 0;
    while ((got = std::fread(buffer, 1, sizeof buffer, file)) > 0) {
        bytes.insert(bytes.end(), buffer, buffer + got);
    }
    std::fclose(file);
    return bytes;
}

bool exists(const std::string& path) {
    struct stat st;
    return ::lstat(path.c_str(), &st) == 0;
}

std::vector<AutoTeamRecord> live_records() {
    std::vector<AutoTeamRecord> records;
    const std::vector<char> bytes = file_bytes(registry_file());
    for (std::size_t offset = sizeof(Header);
         offset + sizeof(AutoTeamRecord) <= bytes.size();
         offset += sizeof(AutoTeamRecord)) {
        AutoTeamRecord record;
        std::memcpy(&record, bytes.data() + offset, sizeof record);
        if (record.pid != 0) {
            records.push_back(record);
        }
    }
    return records;
}

bool holds_lease(std::uint64_t lease) {
    for (const AutoTeamRecord& record : live_records()) {
        if (record.lease == lease) {
            return true;
        }
    }
    return false;
}

const AutoTeamRecord* own_record(const std::vector<AutoTeamRecord>& records) {
    for (const AutoTeamRecord& record : records) {
        if (record.pid == static_cast<std::int32_t>(::getpid())) {
            return &record;
        }
    }
    return nullptr;
}

void write_file(const std::string& path, const std::vector<char>& bytes,
                mode_t mode) {
    ::unlink(path.c_str());
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) {
        check(false, "setup: create " + path);
        return;
    }
    if (!bytes.empty() &&
        ::write(fd, bytes.data(), bytes.size()) !=
            static_cast<ssize_t>(bytes.size())) {
        check(false, "setup: write " + path);
    }
    ::fchmod(fd, mode);
    ::close(fd);
}

std::vector<char> registry_bytes(const std::vector<AutoTeamRecord>& records,
                                 std::uint32_t version = 1,
                                 const char* magic = "XHDFETR1") {
    Header header;
    std::memset(&header, 0, sizeof header);
    std::memcpy(header.magic, magic, sizeof header.magic);
    header.version = version;
    header.record_size = sizeof(AutoTeamRecord);
    header.next_lease = 1000;
    std::vector<char> bytes(sizeof header + records.size() * sizeof(AutoTeamRecord));
    std::memcpy(bytes.data(), &header, sizeof header);
    if (!records.empty()) {
        std::memcpy(bytes.data() + sizeof header, records.data(),
                    records.size() * sizeof(AutoTeamRecord));
    }
    return bytes;
}

void write_registry(const std::vector<AutoTeamRecord>& records) {
    write_file(registry_file(), registry_bytes(records), 0600);
}

AutoTeamRecord make_record(pid_t pid, std::uint64_t start, std::uint64_t lease,
                           std::uint32_t kind, int team,
                           bool explicit_request = false) {
    AutoTeamRecord record;
    std::memset(&record, 0, sizeof record);
    record.pid = static_cast<std::int32_t>(pid);
    record.team = team;
    record.start_time = start;
    record.lease = lease;
    record.kind = kind;
    record.flags = explicit_request ? kAutoTeamExplicitFlag : 0u;
    return record;
}

// A live process of this user that does nothing until it is killed.
struct Helper {
    pid_t pid = -1;
    std::uint64_t start = 0;
};

Helper spawn_helper() {
    Helper helper;
    std::fflush(nullptr);
    const pid_t pid = ::fork();
    if (pid == 0) {
        for (;;) {
            ::pause();
        }
    }
    if (pid < 0) {
        check(false, "setup: fork");
        return helper;
    }
    helper.pid = pid;
    helper.start = start_time_of(pid);
    return helper;
}

void stop_helper(Helper& helper) {
    if (helper.pid > 0) {
        ::kill(helper.pid, SIGKILL);
        ::waitpid(helper.pid, nullptr, 0);
        helper.pid = -1;
    }
}

std::vector<AutoTeamRecord> intents_of(const std::vector<Helper>& helpers,
                                       std::size_t count) {
    std::vector<AutoTeamRecord> records;
    for (std::size_t i = 0; i < count && i < helpers.size(); ++i) {
        records.push_back(make_record(helpers[i].pid, helpers[i].start, 100 + i,
                                      kAutoTeamIntent, 16));
    }
    return records;
}

// Each value runs in a fresh child, which reads the switch for the first time.
void test_switch() {
    struct Case {
        const char* value;
        bool enabled;
    };
    const Case cases[] = {{"0", false}, {" Off ", false}, {"FALSE", false},
                          {"no", false}, {"1", true}};
    for (const Case& value : cases) {
        std::fflush(nullptr);
        const pid_t pid = ::fork();
        if (pid == 0) {
            ::setenv("XHDFE_AUTO_TEAM_COORDINATION", value.value, 1);
            bool ok = hdfe::detail::auto_team_coordination_enabled() == value.enabled;
            if (!value.enabled) {
                AutoTeamRegistration registration;
                const AutoTeamGrant grant =
                    registration.acquire_lease(16, 1, 48, false);
                ok = ok && grant.team == 16 && !grant.reduced &&
                     !exists(registry_file());
            }
            ::_exit(ok ? 0 : 1);
        }
        int status = 0;
        const bool waited = pid > 0 && ::waitpid(pid, &status, 0) == pid;
        check(waited && WIFEXITED(status) && WEXITSTATUS(status) == 0,
              std::string("switch: XHDFE_AUTO_TEAM_COORDINATION='") + value.value +
                  (value.enabled ? "' keeps the registry" : "' turns it off, no file"));
    }
}

void test_grant_formula() {
    struct Case {
        int requested, legacy, capacity, sessions;
        long long leased;
        int expected;
        const char* what;
    };
    const Case cases[] = {
        {16, 1, 48, 1, 0, 16, "grant: alone keeps the automatic team"},
        {16, 1, 48, 3, 0, 16, "grant: three sessions fit within capacity"},
        {16, 1, 48, 4, 0, 12, "grant: four sessions get capacity/4"},
        {16, 1, 48, 8, 0, 6, "grant: eight sessions share equally"},
        {16, 1, 48, 2, 40, 8, "grant: a lease of 40 leaves 8"},
        {16, 1, 48, 2, 48, 1, "grant: a full machine keeps the legacy team"},
        {16, 2, 48, 61, 0, 2, "grant: more sessions than processors keep legacy"},
        {16, 2, 48, 2, 100, 2, "grant: oversubscribed leases keep legacy"},
        {4, 8, 48, 8, 0, 4, "grant: a legacy team above the request never raises it"},
        {16, 16, 48, 8, 0, 16, "grant: legacy equal to the request is unchanged"},
        {0, 0, 0, 0, 0, 1, "grant: degenerate inputs give one thread"},
        {16, 1, 48, 0, -5, 16, "grant: impossible census values are clamped"},
    };
    for (const Case& c : cases) {
        check(auto_team_grant(c.requested, c.legacy, c.capacity, c.sessions,
                              c.leased) == c.expected,
              c.what);
    }
}

void test_lifecycle() {
    ::unlink(registry_file().c_str());
    {
        AutoTeamRegistration registration;
        const AutoTeamGrant one = registration.acquire_intent(1, 1, 48, false);
        check(one.team == 1 && !one.reduced && !exists(registry_file()),
              "lifecycle: a one-thread phase is not recorded");
        const AutoTeamGrant intent = registration.acquire_intent(16, 1, 48, false);
        std::vector<AutoTeamRecord> records = live_records();
        check(intent.team == 16 && !intent.reduced && records.size() == 1 &&
                  records[0].pid == static_cast<std::int32_t>(::getpid()) &&
                  records[0].kind == kAutoTeamIntent && records[0].team == 16 &&
                  records[0].flags == 0,
              "lifecycle: an intent alone keeps its team and is recorded");
        check(records.size() == 1 &&
                  records[0].start_time == start_time_of(::getpid()),
              "lifecycle: the record carries the process start time");
        const std::uint64_t lease = records.empty() ? 0 : records[0].lease;
        const AutoTeamGrant leased = registration.acquire_lease(16, 1, 48, false);
        records = live_records();
        check(leased.team == 16 && records.size() == 1 &&
                  records[0].kind == kAutoTeamLease && records[0].lease == lease,
              "lifecycle: the lease updates the same record");
    }
    struct stat st;
    const bool present = ::stat(registry_file().c_str(), &st) == 0;
    check(present && live_records().empty() &&
              st.st_size == static_cast<off_t>(sizeof(Header)),
          "lifecycle: the destructor releases the record and trims the file");
    check(present && S_ISREG(st.st_mode) && (st.st_mode & 0777) == 0600,
          "lifecycle: the registry is a regular file with mode 0600");
}

void test_nesting() {
    ::unlink(registry_file().c_str());
    {
        AutoTeamRegistration outer;
        outer.acquire_intent(16, 1, 48, false);
        {
            AutoTeamRegistration inner;
            inner.acquire_intent(16, 1, 48, false);
            inner.acquire_lease(16, 1, 48, false);
            const std::vector<AutoTeamRecord> records = live_records();
            check(records.size() == 1 && records[0].kind == kAutoTeamLease,
                  "nesting: an inner registration acts on the outer record");
        }
        check(live_records().size() == 1,
              "nesting: the inner registration does not release the record");
    }
    check(live_records().empty(), "nesting: the outer registration releases it");
}

void test_other_processes(const std::vector<Helper>& helpers) {
    write_registry(intents_of(helpers, 7));
    {
        AutoTeamRegistration registration;
        const AutoTeamGrant grant = registration.acquire_intent(16, 1, 48, false);
        check(grant.team == 6 && grant.reduced,
              "others: seven live intents give the equal share 48/8 = 6");
    }
    check(live_records().size() == 7,
          "others: a release keeps the other sessions' records");
    {
        AutoTeamRegistration registration;
        const AutoTeamGrant grant = registration.acquire_lease(4, 2, 4, false);
        check(grant.team == 2 && grant.reduced,
              "others: more sessions than processors keep the legacy team");
    }
    {
        AutoTeamRegistration registration;
        const AutoTeamGrant grant = registration.acquire_lease(40, 40, 48, true);
        const std::vector<AutoTeamRecord> records = live_records();
        const AutoTeamRecord* mine = own_record(records);
        check(grant.team == 40 && !grant.reduced && mine != nullptr &&
                  mine->flags == kAutoTeamExplicitFlag && mine->team == 40 &&
                  mine->kind == kAutoTeamLease,
              "others: an explicit request is never reduced and is recorded");
    }
    write_registry({make_record(helpers[0].pid, helpers[0].start, 200,
                                kAutoTeamLease, 40, true)});
    {
        AutoTeamRegistration registration;
        const AutoTeamGrant grant = registration.acquire_lease(16, 1, 48, false);
        check(grant.team == 8 && grant.reduced,
              "others: an explicit lease of 40 leaves 8 of 48");
    }
    write_registry({make_record(helpers[0].pid, helpers[0].start, 201,
                                kAutoTeamLease, 48, true)});
    {
        AutoTeamRegistration registration;
        const AutoTeamGrant grant = registration.acquire_lease(16, 1, 48, false);
        check(grant.team == 1 && grant.reduced,
              "others: a full machine leaves the legacy team");
    }
    // Another fit of this same process (another thread, or another copy of
    // the library such as xfepout.plugin next to xhdfe.plugin) is a session.
    write_registry({make_record(::getpid(), start_time_of(::getpid()), 300,
                                kAutoTeamLease, 40)});
    {
        AutoTeamRegistration registration;
        const AutoTeamGrant grant = registration.acquire_lease(16, 1, 48, false);
        check(grant.team == 8 && grant.reduced,
              "others: another fit of this process is counted");
    }
    check(holds_lease(300) && live_records().size() == 1,
          "others: the other fit of this process keeps its record");

    // A real registration in another process, then that process ending.
    ::unlink(registry_file().c_str());
    int ready[2];
    if (::pipe(ready) != 0) {
        check(false, "setup: pipe");
        return;
    }
    std::fflush(nullptr);
    const pid_t child = ::fork();
    if (child == 0) {
        ::close(ready[0]);
        AutoTeamRegistration registration;
        registration.acquire_lease(16, 1, 48, false);
        const char byte = 'x';
        if (::write(ready[1], &byte, 1) != 1) {
            ::_exit(1);
        }
        for (;;) {
            ::pause();
        }
    }
    ::close(ready[1]);
    char byte = 0;
    const bool child_ready = child > 0 && ::read(ready[0], &byte, 1) == 1;
    ::close(ready[0]);
    {
        AutoTeamRegistration registration;
        const AutoTeamGrant grant = registration.acquire_lease(16, 1, 20, false);
        check(child_ready && grant.team == 4 && grant.reduced,
              "others: a lease of another process is seen (20 - 16 = 4)");
    }
    if (child > 0) {
        ::kill(child, SIGKILL);
        ::waitpid(child, nullptr, 0);
    }
    {
        AutoTeamRegistration registration;
        const AutoTeamGrant grant = registration.acquire_lease(16, 1, 20, false);
        check(grant.team == 16 && !grant.reduced && live_records().size() == 1,
              "others: the record of a process that died is discarded");
    }
}

void test_stale_records(const std::vector<Helper>& helpers) {
    Helper dead = spawn_helper();
    const pid_t dead_pid = dead.pid;
    const std::uint64_t dead_start = dead.start;
    stop_helper(dead);
    std::vector<AutoTeamRecord> records = {
        make_record(dead_pid, dead_start, 400, kAutoTeamLease, 40),
        make_record(helpers[1].pid, 0, 402, kAutoTeamIntent, 16),
        make_record(helpers[2].pid, helpers[2].start, 403, 9, 16),
    };
#if defined(__linux__)
    // A reused pid (another start time) and an earlier process with this
    // pid; only Linux reads start times.
    const std::uint64_t self_start = start_time_of(::getpid());
    records.push_back(
        make_record(helpers[0].pid, helpers[0].start + 1, 401, kAutoTeamLease, 40));
    records.push_back(
        make_record(::getpid(), self_start + 1, 404, kAutoTeamLease, 40));
    const bool start_times = helpers[0].start != 0 && self_start != 0;
#else
    const bool start_times = true;
#endif
    write_registry(records);
    {
        AutoTeamRegistration registration;
        const AutoTeamGrant grant = registration.acquire_lease(16, 1, 48, false);
        check(grant.team == 16 && !grant.reduced,
              "stale: dead, reused, unknown-kind and earlier-process records are ignored");
    }
    const std::vector<AutoTeamRecord> left = live_records();
    check(start_times && left.size() == 1 && left[0].lease == 402,
          "stale: they are discarded; a record without start time is kept by pid");
}

void test_fail_open(const std::vector<Helper>& helpers) {
    const std::vector<char> valid = registry_bytes(intents_of(helpers, 7));
    auto expect_open = [](const std::string& path,
                          const std::vector<char>& before, const char* what) {
        AutoTeamRegistration registration;
        const AutoTeamGrant grant = registration.acquire_lease(16, 1, 48, false);
        check(grant.team == 16 && !grant.reduced && file_bytes(path) == before,
              what);
    };

    write_file(registry_file(), valid, 0644);
    expect_open(registry_file(), valid,
                "fail open: a registry readable by others is ignored, unchanged");
    write_file(registry_file(), registry_bytes(intents_of(helpers, 7), 1, "NOTXHDFE"),
               0600);
    expect_open(registry_file(), file_bytes(registry_file()),
                "fail open: a foreign file is ignored and not written");
    write_file(registry_file(), registry_bytes(intents_of(helpers, 7), 2), 0600);
    expect_open(registry_file(), file_bytes(registry_file()),
                "fail open: another layout version is ignored and not written");
    {
        std::vector<AutoTeamRecord> many(1025);
        std::memset(many.data(), 0, many.size() * sizeof(AutoTeamRecord));
        write_file(registry_file(), registry_bytes(many), 0600);
        expect_open(registry_file(), file_bytes(registry_file()),
                    "fail open: an oversized registry is ignored");
    }

    ::unlink(registry_file().c_str());
    write_file(side_file(), valid, 0600);
    check(::symlink(side_file().c_str(), registry_file().c_str()) == 0,
          "setup: symbolic link");
    expect_open(side_file(), valid,
                "fail open: a symbolic link is not followed, its target unchanged");
    ::unlink(registry_file().c_str());
    check(::link(side_file().c_str(), registry_file().c_str()) == 0,
          "setup: hard link");
    expect_open(side_file(), valid, "fail open: a file with two links is ignored");
    ::unlink(registry_file().c_str());
    ::unlink(side_file().c_str());
    check(::mkdir(registry_file().c_str(), 0700) == 0, "setup: directory");
    {
        AutoTeamRegistration registration;
        const AutoTeamGrant grant = registration.acquire_lease(16, 1, 48, false);
        check(grant.team == 16 && !grant.reduced,
              "fail open: a directory in place of the file is ignored");
    }
    ::rmdir(registry_file().c_str());

    // A trailing partial record is not a reason to give up: it is trimmed.
    std::vector<char> tailed = valid;
    tailed.insert(tailed.end(), 5, '\x7f');
    write_file(registry_file(), tailed, 0600);
    {
        AutoTeamRegistration registration;
        const AutoTeamGrant grant = registration.acquire_lease(16, 1, 48, false);
        check(grant.team == 6 && grant.reduced,
              "partial tail: the complete records are still read");
    }
    check(file_bytes(registry_file()).size() == valid.size(),
          "partial tail: the partial record is trimmed");

    // A lock held elsewhere: give up after a short wait, record nothing.
    write_file(registry_file(), valid, 0600);
    const int holder = ::open(registry_file().c_str(), O_RDWR);
    const bool locked = holder >= 0 && ::flock(holder, LOCK_EX) == 0;
    const auto t0 = std::chrono::steady_clock::now();
    expect_open(registry_file(), valid,
                "held lock: no coordination, nothing written");
    const double waited = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - t0).count();
    check(locked && waited < 1.0, "held lock: the fit waits less than a second");
    if (holder >= 0) {
        ::flock(holder, LOCK_UN);
        ::close(holder);
    }
}

void test_pending_release() {
    ::unlink(registry_file().c_str());
    std::uint64_t orphan = 0;
    int holder = -1;
    {
        AutoTeamRegistration registration;
        registration.acquire_lease(16, 1, 48, false);
        const std::vector<AutoTeamRecord> records = live_records();
        orphan = records.size() == 1 ? records[0].lease : 0;
        holder = ::open(registry_file().c_str(), O_RDWR);
        if (holder < 0 || ::flock(holder, LOCK_EX) != 0) {
            check(false, "setup: hold the registry lock");
        }
    }  // the release cannot lock for about a second and is left pending
    check(orphan != 0 && holds_lease(orphan),
          "pending release: a release that cannot lock leaves the record");
    if (holder >= 0) {
        ::flock(holder, LOCK_UN);
        ::close(holder);
    }
    {
        AutoTeamRegistration registration;
        const AutoTeamGrant grant = registration.acquire_lease(16, 1, 48, false);
        check(grant.team == 16 && !holds_lease(orphan) && live_records().size() == 1,
              "pending release: the next operation of the process removes it");
    }
    check(live_records().empty(), "pending release: nothing is left behind");
}

}  // namespace

int main() {
    const char* tmp = std::getenv("TMPDIR");
    std::string base = (tmp != nullptr && tmp[0] == '/') ? tmp : "/tmp";
    base += "/xhdfe-auto-team-XXXXXX";
    std::vector<char> pattern(base.begin(), base.end());
    pattern.push_back('\0');
    if (::mkdtemp(pattern.data()) == nullptr) {
        std::fprintf(stderr, "FAIL setup: mkdtemp %s\n", base.c_str());
        return 1;
    }
    g_dir = pattern.data();
    ::setenv("XHDFE_AUTO_TEAM_REGISTRY_DIR", g_dir.c_str(), 1);
    ::unsetenv("XHDFE_AUTO_TEAM_COORDINATION");

    test_switch();  // before this process reads the switch
    test_grant_formula();
    check(hdfe::detail::auto_team_coordination_enabled(),
          "the registry is on by default");
    test_lifecycle();
    test_nesting();
    std::vector<Helper> helpers;
    for (int i = 0; i < 7; ++i) {
        helpers.push_back(spawn_helper());
    }
    test_other_processes(helpers);
    test_stale_records(helpers);
    test_fail_open(helpers);
    test_pending_release();
    for (Helper& helper : helpers) {
        stop_helper(helper);
    }

    ::unlink(registry_file().c_str());
    ::unlink(side_file().c_str());
    ::rmdir(g_dir.c_str());
    std::printf("%s auto team registry contract: %d failure(s)\n",
                g_failures == 0 ? "OK" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}

#else

int main() {
    std::printf("SKIP auto team registry contract: no registry on this platform\n");
    return 0;
}

#endif
