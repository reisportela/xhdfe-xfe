// Offline controls for N1 decisions, including deliberately wrong evidence.
#include "n1_checks.hpp"
#include <iostream>
#include <string>

using namespace hdfe::detail;

int main() {
    int checks = 0, failures = 0;
    auto check = [&](const std::string& name, bool passed) {
        ++checks;
        if (!passed) { ++failures; std::cerr << "FAIL " << name << '\n'; }
    };
    auto accepts = [](auto operation) {
        try { operation(); return true; }
        catch (const N1Failure&) { return false; }
    };

    for (int response_scale : {-200, 0, 200}) {
        for (int score_scale : {-200, 0, 200}) {
            for (int weight_scale : {-200, 0, 200}) {
                const double ys = std::ldexp(1., response_scale);
                const double xs = std::ldexp(1., score_scale);
                const double w = std::ldexp(1., weight_scale);
                N1NormalEvidence good(2, 1), wrong(2, 1);
                for (int i = 0; i < 32; ++i) {
                    const double x = xs * (2 * (i % 2) - 1);
                    const double u = .125 * ys * (2 * ((i / 2) % 2) - 1);
                    const double beta = .75 * ys / xs;
                    const double y = beta * x + u;
                    auto score = [&](int) { return x; };
                    auto coefficient = [&](int) { return beta; };
                    good.chunks[i / 16].add(y, u, w, 1, score, score, coefficient);
                    wrong.chunks[i / 16].add(y, u + .001 * ys * (x / xs), w, 1,
                                            score, score, coefficient);
                }
                check("normal_exact_rescaled", accepts([&] { good.require(1e-9); }));
                check("normal_error_rescaled", !accepts([&] { wrong.require(1e-8); }));
            }
        }
    }

    N1NormalEvidence iv(1, 1), endogenous_score(1, 1);
    for (int i = 0; i < 32; ++i) {
        const double z = 2 * (i % 2) - 1;
        const double v = 2 * ((i / 2) % 2) - 1;
        const double x = z + .5 * v, u = .25 * v, y = .75 * x + u;
        auto instrument = [&](int) { return z; };
        auto actual = [&](int) { return x; };
        auto coefficient = [](int) { return .75; };
        iv.chunks[0].add(y, u, 1., 1, instrument, actual, coefficient);
        endogenous_score.chunks[0].add(y, u, 1., 1, actual, actual, coefficient);
    }
    check("iv_projected_score", accepts([&] { iv.require(1e-9); }));
    check("iv_endogenous_score_is_wrong", !accepts([&] { endogenous_score.require(1e-8); }));

    for (int field = 0; field < 6; ++field) {
        double values[] = {1., .125, 1., 1., 1., .75};
        values[field] = std::numeric_limits<double>::quiet_NaN();
        N1NormalEvidence invalid(1, 1);
        invalid.chunks[0].add(values[0], values[1], values[2], 1,
            [&](int) { return values[3]; }, [&](int) { return values[4]; },
            [&](int) { return values[5]; });
        check("nonfinite_input_" + std::to_string(field),
              !accepts([&] { invalid.require(1e-8); }));
    }

    N1FeEvidence cancellation, wrong_sign;
    N1FeEvidence::Matrix sums(2, 5);
    sums << 3e-6, 4e-6, 16., 16., 16.,
           -3e-6, -4e-6, 16., 16., 16.;
    cancellation.add_block(sums, 2, 1, 2, 1);
    sums(0, 1) = -4e-6;
    wrong_sign.add_block(sums, 2, 1, 2, 1);
    Eigen::VectorXd beta(1); beta << .75;
    check("signed_fe_cancellation", accepts([&] { cancellation.require(beta, {0}, iv, 1e-9); }));
    check("signed_fe_error", !accepts([&] { wrong_sign.require(beta, {0}, iv, 1e-9); }));

    N1ReconstructionEvidence recovered, wrong_effect, wrong_published_u, fe_tolerance;
    for (int i = 0; i < 32; ++i) {
        const double u = .125 * (2 * (i % 2) - 1);
        recovered.add(2. + u, 2., u, u, 1., 8);
        wrong_effect.add(2. + u, 2.1, u, u, 1., 8);
        wrong_published_u.add(2. + u, 2., u + .1, u, 1., 8);
        fe_tolerance.add(2. + u, 2. + 5e-7, u - 5e-7, u, 1., 8);
    }
    check("published_reconstruction", accepts([&] { recovered.require(1e-6); }));
    check("wrong_published_effect", !accepts([&] { wrong_effect.require(1e-6); }));
    check("wrong_published_residual", !accepts([&] { wrong_published_u.require(1e-6); }));
    check("savefe_own_tolerance", accepts([&] { fe_tolerance.require(1e-6); }));
    check("savefe_stricter_requested_tolerance", !accepts([&] { fe_tolerance.require(1e-9); }));

    check("capture_inactive_outside_fit", !n1_capture_active);
    {
        ScopedN1Capture outer;
        { ScopedN1Capture inner; }
        check("nested_capture_retained", n1_capture_active);
    }
    check("capture_restored", !n1_capture_active);
    N1Failure necessary_failure("N1: injected offline necessary-check failure");
    N1RefinementBudget budget{100};
    check("refinement_uses_only_remaining_budget", budget.remaining(37, necessary_failure) == 63);
    check("second_refinement_refused", !accepts([&] { budget.remaining(38, necessary_failure); }));
    N1RefinementBudget exhausted{100};
    check("exhausted_budget_refused", !accepts([&] { exhausted.remaining(100, necessary_failure); }));
    std::cout << "{\"checks\":" << checks << ",\"failures\":" << failures << "}\n";
    return failures ? 1 : 0;
}
