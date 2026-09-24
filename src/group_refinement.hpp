#pragma once

namespace hdfe {
namespace detail {

// Internal, task-local capture for a correction solve; no estimator API flag.
bool& group_refinement_capture();

class ScopedGroupRefinementCapture {
    bool previous_;
public:
    ScopedGroupRefinementCapture() : previous_(group_refinement_capture()) {
        group_refinement_capture() = true;
    }
    ~ScopedGroupRefinementCapture() { group_refinement_capture() = previous_; }
    ScopedGroupRefinementCapture(const ScopedGroupRefinementCapture&) = delete;
    ScopedGroupRefinementCapture& operator=(const ScopedGroupRefinementCapture&) = delete;
};

}  // namespace detail
}  // namespace hdfe
