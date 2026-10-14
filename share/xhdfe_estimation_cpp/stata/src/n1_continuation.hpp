#ifndef HDFE_N1_CONTINUATION_HPP
#define HDFE_N1_CONTINUATION_HPP

#include <Eigen/Core>
#include <functional>

namespace hdfe {
namespace detail {

using N1ContinuationPredicate =
    std::function<bool(const Eigen::VectorXd&, const Eigen::MatrixXd&)>;

inline const N1ContinuationPredicate*& n1_continuation_predicate() {
    static thread_local const N1ContinuationPredicate* predicate = nullptr;
    return predicate;
}

class ScopedN1Continuation {
    const N1ContinuationPredicate* previous_;
public:
    explicit ScopedN1Continuation(const N1ContinuationPredicate* predicate) noexcept
        : previous_(n1_continuation_predicate()) {
        n1_continuation_predicate() = predicate;
    }
    ~ScopedN1Continuation() noexcept { n1_continuation_predicate() = previous_; }
    ScopedN1Continuation(const ScopedN1Continuation&) = delete;
    ScopedN1Continuation& operator=(const ScopedN1Continuation&) = delete;
};

}  // namespace detail
}  // namespace hdfe
#endif
