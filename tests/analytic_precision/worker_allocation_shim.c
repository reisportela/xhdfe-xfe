/* Fault-injection shim (synthetic reproducers only). Mode 1: log every large malloc made by a thread
   other than the arming thread, with a short backtrace. Mode 2: return NULL for the fail_at-th such
   malloc. Derived from code_review_claude_20261005/reproducers/shim.c (log limit raised). */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
extern void* __libc_malloc(size_t);
static atomic_int armed = 0;
static atomic_long hits = 0;
static long fail_at = -1;
static size_t min_size = 262144;
static pid_t main_tid = 0;
static __thread int busy = 0;
void shim_arm(int mode, long at, long min) {
    void* tmp[4]; backtrace(tmp, 4);
    fail_at = at; min_size = (size_t)min; main_tid = (pid_t)syscall(SYS_gettid);
    atomic_store(&hits, 0); atomic_store(&armed, mode);
}
long shim_hits(void) { return atomic_load(&hits); }
void* malloc(size_t size) {
    int mode = atomic_load(&armed);
    if (mode && !busy && size >= min_size && (pid_t)syscall(SYS_gettid) != main_tid) {
        busy = 1;
        long c = atomic_fetch_add(&hits, 1);
        if ((mode == 1 && c < 20000) || (mode == 2 && c == fail_at)) {
            void* frames[10]; int nf = backtrace(frames, 10);
            char line[1024]; int len = snprintf(line, sizeof line, "HIT %ld size=%zu", c, size);
            for (int i = 1; i < nf && len < 900; ++i) {
                Dl_info info;
                if (dladdr(frames[i], &info) && info.dli_fname) {
                    const char* b = strrchr(info.dli_fname, '/');
                    len += snprintf(line + len, sizeof line - len, " %s+0x%lx", b ? b + 1 : info.dli_fname,
                                    (unsigned long)((char*)frames[i] - (char*)info.dli_fbase));
                }
            }
            line[len++] = '\n'; if (write(2, line, len) < 0) {}
        }
        if (mode == 2 && c == fail_at) {
            static const char msg[] = "SHIM: returning NULL for this worker-thread malloc\n";
            if (write(2, msg, sizeof msg - 1) < 0) {}
            atomic_store(&armed, 0); busy = 0; return NULL;
        }
        busy = 0;
    }
    return __libc_malloc(size);
}
