// Offline ELF interposer. It is loaded only by n1_fault_injection_contract.py.
#include "n1_checks.hpp"

#include <dlfcn.h>
#include <stdexcept>

namespace {
using Require = void (*)(const hdfe::detail::N1NormalEvidence*, double);
Require original = nullptr;
int armed = 0;
int calls = 0;
int forwarded = 0;
double tolerances[8] = {};
}

extern "C" void n1_test_require(const hdfe::detail::N1NormalEvidence*, double)
    asm("_ZNK4hdfe6detail16N1NormalEvidence7requireEd");

extern "C" void n1_test_require(const hdfe::detail::N1NormalEvidence* evidence,
                                 double tolerance) {
    if (!original || original == &n1_test_require)
        throw std::logic_error("fault-injection harness: real N1 symbol was not registered");
    if (calls < 8) tolerances[calls] = tolerance;
    ++calls;
    if (armed > 0) {
        --armed;
        throw hdfe::detail::N1Failure("N1: injected offline negative control; no estimates returned");
    }
    ++forwarded;
    original(evidence, tolerance);
}

extern "C" int n1_test_set_original(void* pointer) {
    if (!pointer || pointer == reinterpret_cast<void*>(&n1_test_require)) return 0;
    original = reinterpret_cast<Require>(pointer);
    return 1;
}

extern "C" const char* n1_test_original_path() {
    Dl_info info{};
    return original && dladdr(reinterpret_cast<void*>(original), &info)
        ? info.dli_fname : nullptr;
}

extern "C" void* n1_test_hook_address() {
    return reinterpret_cast<void*>(&n1_test_require);
}

extern "C" void n1_test_arm(int count) {
    armed = count;
    calls = 0;
    forwarded = 0;
    for (double& value : tolerances) value = 0;
}

extern "C" int n1_test_calls() { return calls; }
extern "C" int n1_test_forwarded() { return forwarded; }
extern "C" double n1_test_tolerance(int call) {
    return call >= 0 && call < calls && call < 8 ? tolerances[call] : -1;
}
