#include "n1_continuation.hpp"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace {
std::atomic<int> failures{0};
std::atomic<int> checks{0};

void require(bool condition, const char* message) {
    checks.fetch_add(1);
    if (!condition) {
        failures.fetch_add(1);
        std::fprintf(stderr, "N1_CONTINUATION_CONTEXT_FAIL: %s\n", message);
    }
}

struct PredicateFailure : std::runtime_error {
    PredicateFailure() : std::runtime_error("synthetic predicate failure") {}
};
}

int main(int argc, char** argv) {
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--inject-context-error") != 0))
        return 2;
    const bool inject_error = argc == 2;
    using namespace hdfe::detail;
    N1ContinuationPredicate first = [](const auto&, const auto&) { return false; };
    N1ContinuationPredicate nested = [](const auto&, const auto&) -> bool {
        throw PredicateFailure();
    };
    require(n1_continuation_predicate() == nullptr, "initial context");
    std::atomic<int> rendezvous{0};
    std::thread peer([&] {
        require(n1_continuation_predicate() == nullptr, "peer initial context");
        ScopedN1Continuation scope(&nested);
        rendezvous.fetch_add(1);
        while (rendezvous.load() < 2) std::this_thread::yield();
        require(n1_continuation_predicate() == &nested, "peer thread isolation");
    });
    {
        ScopedN1Continuation scope(&first);
        rendezvous.fetch_add(1);
        while (rendezvous.load() < 2) std::this_thread::yield();
        if (inject_error) n1_continuation_predicate() = nullptr;
        require(n1_continuation_predicate() == &first, "main thread isolation");
        bool caught_expected = false;
        try {
            ScopedN1Continuation inner(&nested);
            (*n1_continuation_predicate())(Eigen::VectorXd{}, Eigen::MatrixXd{});
        } catch (const PredicateFailure&) { caught_expected = true; }
        require(caught_expected, "nested predicate must throw");
        require(n1_continuation_predicate() == &first, "restoration after exception");
        {
            ScopedN1Continuation disabled(nullptr);
            require(n1_continuation_predicate() == nullptr, "disabled context");
        }
        require(n1_continuation_predicate() == &first, "restoration after disabling");
    }
    peer.join();
    require(n1_continuation_predicate() == nullptr, "final restoration");
    if (failures.load() != 0) return 1;
    std::printf("N1_CONTINUATION_CONTEXT_PASS checks=%d\n", checks.load());
    return 0;
}
