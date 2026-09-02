#include "hdfe/student_t_detail.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#if defined(__SSE__)
#include <xmmintrin.h>
#endif

namespace student = hdfe::detail::student_t_internal;

namespace {

#if defined(_MSC_VER)
#define XHDFE_TEST_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define XHDFE_TEST_NOINLINE __attribute__((noinline))
#else
#define XHDFE_TEST_NOINLINE
#endif

XHDFE_TEST_NOINLINE double public_probability_noinline(
    double t, double df) {
    return student::two_sided_probability(t, df);
}

struct Fixture {
    const char* name;
    double df;
    double t;
    double probability;
    bool has_probability;
    double quantile_probability;
    double critical;
};

struct HighPrecisionFixture {
    double df;
    double target;
    double quantile;
};

constexpr std::array<Fixture, 10> kStataFixtures = {{
    {"patents", 49347.0, 0.50916929119588794, 0.61063584277650618, true,
     0.975, 1.9600120589605514},
    {"schools", 206623.0, 1.473485721671121, 0.14062163446714071, true,
     0.975, 1.9599754657637822},
    {"uniform_easy", 148356.0, 0.93501225146834288, 0.34978350750772269,
     true, 0.975, 1.9599799750645721},
    {"uniform_hard", 112742.0, 0.0074594128213933608, 0.9940483180701507,
     true, 0.975, 1.9599850263546905},
    {"uniform_harder", 118119.0, 1.9594791142984223,
     0.050059051814231349, true, 0.975, 1.9599840684818926},
    {"df_1", 1.0, 0.0, 0.0, false, 0.975, 12.706204736174698},
    {"df_2", 2.0, 0.0, 0.0, false, 0.975, 4.3026527297494654},
    {"df_10", 10.0, 0.0, 0.0, false, 0.975, 2.228138851986273},
    {"df_100", 100.0, 0.0, 0.0, false, 0.975, 1.9839715185235531},
    {"df_1_level_99_9999", 1.0, 0.0, 0.0, false,
     0.5 + 99.9999 / 200.0, 636619.77234592417},
}};

constexpr std::array<HighPrecisionFixture, 8> kHighPrecisionFixtures = {{
    {1e5, 1e-20, 9.3381029370347217},
    {1e5, 1e-50, 14.987921921602688},
    {1e5, 1e-100, 21.330195588157424},
    {1e5, 1e-300, 37.193555717179355},
    {1e6, 1e-20, 9.3362506235384835},
    {1e6, 1e-50, 14.980321647912926},
    {1e6, 1e-100, 21.308363547157575},
    {1e6, 1e-300, 37.078531718230499},
}};

bool within_abs(double observed, double expected, double limit) {
    return hdfe::detail::ieee_finite(observed) &&
           std::abs(observed - expected) <= limit;
}

bool within_scaled(double observed, double expected, double limit) {
    return within_abs(
        observed, expected, limit * std::max(1.0, std::abs(expected)));
}

std::uint64_t double_bits(double value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

template <typename Real>
double legacy_nonfallback_probability(double t_double, double df_double) {
    const Real t = static_cast<Real>(t_double);
    const Real df = static_cast<Real>(df_double);
    if (df >= Real(1e5)) {
        const auto coordinate = student::asymptotic_normal_coordinate(t, df);
        if (!coordinate.certified) {
            throw std::runtime_error(
                "legacy nonfallback fixture unexpectedly requires fallback");
        }
        const Real approximation =
            student::normal_two_sided_from_coordinate(coordinate.z);
        if (approximation >=
                static_cast<Real>(student::kExactTinyTailThreshold) ||
            df > static_cast<Real>(student::kExactTinyTailMaxDf)) {
            return static_cast<double>(approximation);
        }
    }
    return static_cast<double>(
        student::exact_two_sided_probability_real<Real>(t, df));
}

template <typename Real>
bool probability_real_rejects(double t_double, double df_double) {
    try {
        (void)student::two_sided_probability_real<Real>(
            static_cast<Real>(t_double), static_cast<Real>(df_double));
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

int selftest() {
    constexpr double kLimit = 5e-12;
    const char* stage = "Stata fixtures";
    try {
    for (const auto& fixture : kStataFixtures) {
        const bool public_level =
            std::string(fixture.name) == "df_1_level_99_9999";
        const double public_tail = (100.0 - 99.9999) / 200.0;
        const double critical = public_level
            ? student::inverse_survival(public_tail, fixture.df)
            : student::inverse_cdf(fixture.quantile_probability, fixture.df);
        const double binary_critical = public_level
            ? student::inverse_survival_checked<double>(public_tail, fixture.df)
            : student::inverse_cdf_checked<double>(
                  fixture.quantile_probability, fixture.df);
        if (!within_scaled(critical, fixture.critical, kLimit) ||
            !within_scaled(binary_critical, fixture.critical, kLimit)) {
            std::fprintf(stderr,
                         "FAIL: Stata critical fixture differs: %s %.17g %.17g %.17g\n",
                         fixture.name, fixture.critical, critical, binary_critical);
            return 1;
        }
        if (fixture.has_probability) {
            const double probability =
                student::two_sided_probability(std::abs(fixture.t), fixture.df);
            const double binary_probability =
                student::two_sided_probability_with_real<double>(
                    std::abs(fixture.t), fixture.df);
            if (!within_abs(probability, fixture.probability, kLimit) ||
                !within_abs(binary_probability, fixture.probability, kLimit)) {
                std::fprintf(stderr,
                             "FAIL: Stata p fixture differs: %s %.17g %.17g %.17g\n",
                             fixture.name, fixture.probability,
                             probability, binary_probability);
                return 1;
            }
        }
    }

    stage = "Cauchy extreme tails";
    constexpr long double kPi =
        3.141592653589793238462643383279502884L;
    constexpr std::array<double, 5> cauchy_lower_tails = {
        1e-12, 1e-50, 1e-100, 1e-300,
        std::numeric_limits<double>::min()};
    for (const double p : cauchy_lower_tails) {
        const double quantile = student::inverse_cdf(p, 1.0);
        const double binary_quantile =
            student::inverse_cdf_checked<double>(p, 1.0);
        const long double expected =
            -1.0L / std::tan(kPi * static_cast<long double>(p));
        const double target = 2.0 * p;
        const double achieved = student::two_sided_probability(
            std::abs(quantile), 1.0);
        const double binary_achieved =
            student::two_sided_probability_with_real<double>(
                std::abs(binary_quantile), 1.0);
        const double relative = std::abs(achieved - target) / target;
        const double binary_relative =
            std::abs(binary_achieved - target) / target;
        if (!within_scaled(quantile, static_cast<double>(expected), kLimit) ||
            !within_scaled(
                binary_quantile, static_cast<double>(expected), kLimit) ||
            relative > kLimit || binary_relative > kLimit) {
            std::fprintf(stderr,
                         "FAIL: extreme Cauchy tail differs: p=%.17g q=%.17g q64=%.17g rel=%.17g rel64=%.17g\n",
                         p, quantile, binary_quantile, relative, binary_relative);
            return 1;
        }
    }
    const double upper = std::nextafter(1.0, 0.0);
    const double upper_quantile = student::inverse_cdf(upper, 1.0);
    const double upper_target = 2.0 * (1.0 - upper);
    if (!(upper_quantile > 0.0) ||
        std::abs(student::two_sided_probability(upper_quantile, 1.0) -
                 upper_target) / upper_target > kLimit) {
        std::fprintf(stderr, "FAIL: upper-tail Cauchy target lost precision\n");
        return 1;
    }
    const double extreme_public_level = std::nextafter(100.0, 0.0);
    const double extreme_public_tail =
        (100.0 - extreme_public_level) / 200.0;
    const double extreme_public_critical =
        student::inverse_survival(extreme_public_tail, 1.0);
    const long double extreme_public_expected = 1.0L / std::tan(
        kPi * static_cast<long double>(extreme_public_tail));
    if (!within_scaled(
            extreme_public_critical,
            static_cast<double>(extreme_public_expected), kLimit) ||
        std::abs(
            student::two_sided_probability(extreme_public_critical, 1.0) -
            2.0 * extreme_public_tail) /
            (2.0 * extreme_public_tail) > kLimit) {
        std::fprintf(stderr,
                     "FAIL: most-extreme public Cauchy level differs\n");
        return 1;
    }
    bool extreme_threw = false;
    try {
        (void)student::inverse_cdf(
            std::numeric_limits<double>::denorm_min(), 1.0);
    } catch (const std::runtime_error&) {
        extreme_threw = true;
    }
    if (!extreme_threw) {
        std::fprintf(stderr,
                     "FAIL: unrepresentable Cauchy quantile did not fail closed\n");
        return 1;
    }
    bool mutation_threw = false;
    try {
        student::require_relative_tail_postcondition<double>(
            1e-13, 1e-20, "mutated relative tail was accepted");
    } catch (const std::runtime_error&) {
        mutation_threw = true;
    }
    if (!mutation_threw) {
        std::fprintf(stderr,
                     "FAIL: achieved 1e-13 was accepted for target 1e-20\n");
        return 1;
    }
    bool tolerance_underflow_threw = false;
    try {
        (void)student::tail_relative_scale<double>(
            std::numeric_limits<double>::denorm_min());
    } catch (const std::runtime_error&) {
        tolerance_underflow_threw = true;
    }
    if (!tolerance_underflow_threw) {
        std::fprintf(stderr,
                     "FAIL: underflowed relative-tail tolerance was accepted\n");
        return 1;
    }

    stage = "ordinary adversarial grid";
    constexpr std::array<double, 19> dfs = {
        0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 28.0, 30.0, 100.0,
        581.0, 4999.0, 49347.0, 206623.0, 999999.0, 1e6,
        1000001.0, 1e8, 1e12};
    constexpr std::array<double, 17> ts = {
        0.0, 1e-12, 1e-8, 1e-4, 0.0074594128213933608,
        0.1, 0.5, 0.93501225146834288, 1.0, 1.9594791142984223,
        1.96, 2.0, 3.4804508701108352, 5.4518931376060493,
        8.0, 20.0, 40.0};
    for (const double df : dfs) {
        double previous = 1.0;
        for (const double t : ts) {
            const double probability = student::two_sided_probability(t, df);
            const double binary_probability =
                student::two_sided_probability_with_real<double>(t, df);
            if (!(probability >= 0.0 && probability <= 1.0) ||
                probability > previous ||
                !within_abs(probability, binary_probability, kLimit)) {
                std::fprintf(stderr,
                             "FAIL: adversarial p grid differs: df=%.17g t=%.17g p=%.17g p64=%.17g\n",
                             df, t, probability, binary_probability);
                return 1;
            }
            previous = probability;
        }
        const double critical = student::inverse_cdf(0.975, df);
        const double binary_critical =
            student::inverse_cdf_checked<double>(0.975, df);
        const double achieved = student::two_sided_probability(critical, df);
        const double binary_achieved =
            student::two_sided_probability_with_real<double>(
                binary_critical, df);
        if (!within_abs(achieved, 0.05, kLimit) ||
            !within_abs(binary_achieved, 0.05, kLimit) ||
            !within_scaled(critical, binary_critical, kLimit)) {
            std::fprintf(stderr,
                         "FAIL: adversarial inverse grid differs: df=%.17g q=%.17g q64=%.17g p=%.17g p64=%.17g\n",
                         df, critical, binary_critical, achieved, binary_achieved);
            return 1;
        }
    }

    stage = "private extreme inverse grid";
    constexpr std::array<double, 5> extreme_dfs = {
        2.0, 10.0, 100.0, 1e5, 1e6};
    constexpr std::array<double, 4> extreme_targets = {
        1e-20, 1e-50, 1e-100, 1e-300};
    for (const double df : extreme_dfs) {
        for (const double target : extreme_targets) {
            if (df >= 1e5) {
                bool working_threw = false;
                bool binary_threw = false;
                try {
                    (void)student::positive_quantile_from_two_sided_target<
                        student::WorkingReal>(
                        static_cast<student::WorkingReal>(target),
                        static_cast<student::WorkingReal>(df));
                } catch (const std::runtime_error&) {
                    working_threw = true;
                }
                try {
                    (void)student::positive_quantile_from_two_sided_target<
                        double>(target, df);
                } catch (const std::runtime_error&) {
                    binary_threw = true;
                }
                if (!working_threw || !binary_threw) {
                    std::fprintf(stderr,
                                 "FAIL: below-floor large-df inverse was accepted: df=%.17g target=%.17g\n",
                                 df, target);
                    return 1;
                }
                continue;
            }
            const double quantile =
                student::positive_quantile_from_two_sided_target<
                    student::WorkingReal>(
                    static_cast<student::WorkingReal>(target),
                    static_cast<student::WorkingReal>(df));
            const double binary_quantile =
                student::positive_quantile_from_two_sided_target<double>(
                    target, df);
            const double achieved =
                student::two_sided_probability(quantile, df);
            const double binary_achieved =
                student::two_sided_probability_with_real<double>(
                    binary_quantile, df);
            const double relative = std::abs(achieved - target) / target;
            const double binary_relative =
                std::abs(binary_achieved - target) / target;
            if (relative > kLimit || binary_relative > kLimit) {
                std::fprintf(stderr,
                             "FAIL: extreme inverse grid differs: df=%.17g target=%.17g q=%.17g q64=%.17g rel=%.17g rel64=%.17g\n",
                             df, target, quantile, binary_quantile,
                             relative, binary_relative);
                return 1;
            }
        }
    }

    stage = "independent high-precision forward fixtures";
    for (const auto& fixture : kHighPrecisionFixtures) {
        const double achieved = student::two_sided_probability(
            fixture.quantile, fixture.df);
        const double binary_achieved =
            student::two_sided_probability_with_real<double>(
                fixture.quantile, fixture.df);
        if (std::abs(achieved / fixture.target - 1.0) > kLimit ||
            std::abs(binary_achieved / fixture.target - 1.0) > 1e-9) {
            std::fprintf(stderr,
                         "FAIL: independent high-precision forward fixture differs: df=%.17g target=%.17g q=%.17g p=%.17g p64=%.17g\n",
                         fixture.df, fixture.target, fixture.quantile,
                         achieved, binary_achieved);
            return 1;
        }
    }

    stage = "independent subnormal forward fixtures";
    constexpr double subnormal_t = 37.794586197174261;
    constexpr double subnormal_reference = 2.1977036064894894e-310;
    const std::uint64_t subnormal_reference_bits =
        double_bits(subnormal_reference);
    for (const double probability : {
             student::two_sided_probability(subnormal_t, 1e5),
             student::two_sided_probability_with_real<double>(
                 subnormal_t, 1e5)}) {
        const std::uint64_t bits = double_bits(probability);
        const bool is_nonzero_subnormal =
            (bits & UINT64_C(0x7ff0000000000000)) == 0 &&
            (bits & UINT64_C(0x000fffffffffffff)) != 0;
        const std::uint64_t ulp_distance = bits > subnormal_reference_bits
            ? bits - subnormal_reference_bits
            : subnormal_reference_bits - bits;
        if (!is_nonzero_subnormal || ulp_distance > 8) {
            std::fprintf(stderr,
                         "FAIL: representable forward subnormal differs: p=%.17g bits=%llx ulp=%llu\n",
                         probability,
                         static_cast<unsigned long long>(bits),
                         static_cast<unsigned long long>(ulp_distance));
            return 1;
        }
    }
    constexpr double min_subnormal_t = 38.628450615292988;
    constexpr double underflow_t = 38.730721422256442;
    for (const bool binary : {false, true}) {
        const double minimum = binary
            ? student::two_sided_probability_with_real<double>(
                  min_subnormal_t, 1e5)
            : student::two_sided_probability(min_subnormal_t, 1e5);
        const double underflow = binary
            ? student::two_sided_probability_with_real<double>(
                  underflow_t, 1e5)
            : student::two_sided_probability(underflow_t, 1e5);
        if (double_bits(minimum) != UINT64_C(1) ||
            double_bits(underflow) != UINT64_C(0)) {
            std::fprintf(stderr,
                         "FAIL: binary64 underflow boundary differs: binary=%d min=%llx zero=%llx\n",
                         static_cast<int>(binary),
                         static_cast<unsigned long long>(double_bits(minimum)),
                         static_cast<unsigned long long>(double_bits(underflow)));
            return 1;
        }
    }
#if defined(__SSE__)
    stage = "MXCSR FTZ/DAZ public probability";
    const unsigned int saved_mxcsr = _mm_getcsr();
    _mm_setcsr(saved_mxcsr | (1U << 15) | (1U << 6));
    const double mxcsr_subnormal =
        public_probability_noinline(subnormal_t, 1e5);
    const double mxcsr_minimum =
        public_probability_noinline(min_subnormal_t, 1e5);
    _mm_setcsr(saved_mxcsr);
    const std::uint64_t mxcsr_bits = double_bits(mxcsr_subnormal);
    const std::uint64_t mxcsr_distance =
        mxcsr_bits > subnormal_reference_bits
            ? mxcsr_bits - subnormal_reference_bits
            : subnormal_reference_bits - mxcsr_bits;
    if (mxcsr_distance > 8 || double_bits(mxcsr_minimum) != UINT64_C(1)) {
        std::fprintf(stderr,
                     "FAIL: MXCSR FTZ/DAZ changed public p bits: p=%llx min=%llx\n",
                     static_cast<unsigned long long>(mxcsr_bits),
                     static_cast<unsigned long long>(double_bits(mxcsr_minimum)));
        return 1;
    }
#endif

    stage = "certified asymptotic fallback and nonfiring identity";
    struct ObservedHugeT {
        const char* name;
        double t;
    };
    constexpr std::array<ObservedHugeT, 4> observed_huge_t = {{
        {"pf_difficult_10m_2fe", 369963.05334953},
        {"pf_difficult_10m_3fe", 523130.4505001678},
        {"pf_simple_10m_2fe", 419768.0430004782},
        {"pf_simple_10m_3fe", 591108.2222778986},
    }};
    for (const auto& fixture : observed_huge_t) {
        const auto working_coordinate =
            student::asymptotic_normal_coordinate<student::WorkingReal>(
                static_cast<student::WorkingReal>(fixture.t),
                static_cast<student::WorkingReal>(999999.0));
        const auto binary_coordinate =
            student::asymptotic_normal_coordinate<double>(
                fixture.t, 999999.0);
        if (working_coordinate.certified || binary_coordinate.certified) {
            std::fprintf(stderr,
                         "FAIL: observed huge-t asymptotic coordinate unexpectedly certified: %s\n",
                         fixture.name);
            return 1;
        }
        const double working =
            student::two_sided_probability(fixture.t, 999999.0);
        const double binary = student::two_sided_probability_with_real<double>(
            fixture.t, 999999.0);
        if (double_bits(working) != UINT64_C(0) ||
            double_bits(binary) != UINT64_C(0) ||
            std::signbit(working) || std::signbit(binary)) {
            std::fprintf(stderr,
                         "FAIL: observed huge-t fallback is not positive zero: %s %.17g %.17g\n",
                         fixture.name, working, binary);
            return 1;
        }
    }

    struct NonfiringFixture {
        double t;
        double df;
    };
    constexpr std::array<NonfiringFixture, 8> nonfiring = {{
        {2.0, 99999.0}, {2.0, 100000.0}, {5.0, 1e5},
        {8.0, 999999.0}, {8.0, 1000001.0}, {8.0, 1e12},
        {subnormal_t, 1e5}, {min_subnormal_t, 1e5},
    }};
    for (const auto& fixture : nonfiring) {
        const double selected =
            student::two_sided_probability(fixture.t, fixture.df);
        const double legacy =
            legacy_nonfallback_probability<student::WorkingReal>(
                fixture.t, fixture.df);
        const double selected_binary =
            student::two_sided_probability_with_real<double>(
                fixture.t, fixture.df);
        const double legacy_binary = legacy_nonfallback_probability<double>(
            fixture.t, fixture.df);
        if (double_bits(selected) != double_bits(legacy) ||
            double_bits(selected_binary) != double_bits(legacy_binary)) {
            std::fprintf(stderr,
                         "FAIL: nonfiring Student-t path changed bits: df=%.17g t=%.17g\n",
                         fixture.df, fixture.t);
            return 1;
        }
    }

    const double sqrt_overflow_band =
        2.0 * std::sqrt(std::numeric_limits<double>::max()) *
        std::sqrt(1e5);
    constexpr std::array<double, 6> huge_dfs = {
        1e5, 999999.0, 1e6, 1000001.0, 1e9, 1e12};
    const std::array<double, 7> huge_ts = {
        8.0, 40.0, 1e3, 1e6, 1e20, sqrt_overflow_band,
        std::numeric_limits<double>::max()};
    for (const double df : huge_dfs) {
        double previous_working = 1.0;
        double previous_binary = 1.0;
        for (const double t : huge_ts) {
            const double working = student::two_sided_probability(t, df);
            const double binary =
                student::two_sided_probability_with_real<double>(t, df);
            if (!(working >= 0.0 && working <= previous_working) ||
                !(binary >= 0.0 && binary <= previous_binary) ||
                !hdfe::detail::ieee_finite(working) ||
                !hdfe::detail::ieee_finite(binary)) {
                std::fprintf(stderr,
                             "FAIL: huge-t probability is invalid/nonmonotone: df=%.17g t=%.17g p=%.17g p64=%.17g\n",
                             df, t, working, binary);
                return 1;
            }
            if (t >= 40.0 &&
                (double_bits(working) != UINT64_C(0) ||
                 double_bits(binary) != UINT64_C(0) ||
                 std::signbit(working) || std::signbit(binary))) {
                std::fprintf(stderr,
                             "FAIL: huge-t underflow is not positive zero: df=%.17g t=%.17g\n",
                             df, t);
                return 1;
            }
            previous_working = working;
            previous_binary = binary;
        }
    }

    const double large_df_critical = student::inverse_cdf(0.975, 999999.0);
    if (!within_abs(
            large_df_critical, 1.9599663568164793, 4e-15)) {
        std::fprintf(stderr,
                     "FAIL: large-df critical changed under forward fallback\n");
        return 1;
    }

    stage = "supported large-df inverse tails";
    constexpr std::array<double, 4> supported_large_df_targets = {
        student::kLargeDfInverseTailFloor,
        2.0 * student::kLargeDfInverseTailFloor,
        1e-12,
        1e-8};
    for (const double df : {1e5, 1e6}) {
        for (const double target : supported_large_df_targets) {
            const double quantile =
                student::positive_quantile_from_two_sided_target<
                    student::WorkingReal>(
                    static_cast<student::WorkingReal>(target),
                    static_cast<student::WorkingReal>(df));
            const double binary_quantile =
                student::positive_quantile_from_two_sided_target<double>(
                    target, df);
            const double achieved =
                student::two_sided_probability(quantile, df);
            const double binary_achieved =
                student::two_sided_probability_with_real<double>(
                    binary_quantile, df);
            if (std::abs(achieved / target - 1.0) > kLimit ||
                std::abs(binary_achieved / target - 1.0) > kLimit) {
                std::fprintf(stderr,
                             "FAIL: supported large-df exact inverse differs: df=%.17g target=%.17g\n",
                             df, target);
                return 1;
            }
        }
        bool below_floor_threw = false;
        try {
            (void)student::positive_quantile_from_two_sided_target<
                student::WorkingReal>(
                static_cast<student::WorkingReal>(
                    std::nextafter(
                        student::kLargeDfInverseTailFloor, 0.0)),
                static_cast<student::WorkingReal>(df));
        } catch (const std::runtime_error&) {
            below_floor_threw = true;
        }
        if (!below_floor_threw) {
            std::fprintf(stderr,
                         "FAIL: large-df inverse support floor is open\n");
            return 1;
        }
    }

    stage = "large-df cutoff continuity";
    for (const double t : {2.0, 4.0, 5.0}) {
        const double selected = student::two_sided_probability(t, 1e5);
        const double exact = static_cast<double>(
            student::exact_two_sided_probability_real<student::WorkingReal>(
                static_cast<student::WorkingReal>(t),
                static_cast<student::WorkingReal>(1e5)));
        if (std::abs(selected / exact - 1.0) > kLimit) {
            std::fprintf(stderr,
                         "FAIL: large-df fast/exact continuity differs: t=%.17g selected=%.17g exact=%.17g\n",
                         t, selected, exact);
            return 1;
        }
    }
    if (std::abs(
            student::inverse_cdf(0.975, 99999.0) -
            student::inverse_cdf(0.975, 100000.0)) > 1e-8) {
        std::fprintf(stderr,
                     "FAIL: Student-t df cutoff critical is discontinuous\n");
        return 1;
    }

    stage = "invalid-input fail-closed";
    const double quiet_nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    if (!std::isnan(student::two_sided_probability(-1.0, 10.0)) ||
        !std::isnan(student::two_sided_probability(quiet_nan, 1e5)) ||
        !std::isnan(student::two_sided_probability(infinity, 1e5)) ||
        !std::isnan(student::two_sided_probability(1.0, 0.0)) ||
        !std::isnan(student::two_sided_probability(1.0, quiet_nan)) ||
        !std::isnan(student::inverse_cdf(0.975, 0.0))) {
        std::fprintf(stderr, "FAIL: invalid Student-t inputs did not fail closed\n");
        return 1;
    }
    struct InvalidProbabilityInput {
        double t;
        double df;
    };
    constexpr std::array<InvalidProbabilityInput, 6> invalid_probability = {{
        {-1.0, 1e5},
        {std::numeric_limits<double>::quiet_NaN(), 1e5},
        {std::numeric_limits<double>::infinity(), 1e5},
        {1.0, 0.0},
        {1.0, std::numeric_limits<double>::quiet_NaN()},
        {1.0, std::numeric_limits<double>::infinity()},
    }};
    for (const auto& fixture : invalid_probability) {
        if (!probability_real_rejects<student::WorkingReal>(
                fixture.t, fixture.df) ||
            !probability_real_rejects<double>(fixture.t, fixture.df)) {
            std::fprintf(stderr,
                         "FAIL: direct Student-t real path accepted invalid input\n");
            return 1;
        }
    }
    bool threw = false;
    try {
        (void)student::inverse_cdf(1.0, 10.0);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    if (!threw) {
        std::fprintf(stderr, "FAIL: invalid Student-t probability did not throw\n");
        return 1;
    }

    std::cout << "STUDENT_T_PRECISION_PASS fixtures="
              << kStataFixtures.size() << " grid="
              << dfs.size() * ts.size()
              << " extreme_inverse_grid="
              << extreme_dfs.size() * extreme_targets.size()
              << " high_precision_fixtures="
              << kHighPrecisionFixtures.size()
              << " subnormal_bits=PASS"
              << " noinline_mxcsr=PASS"
              << " relative_tail_mutation=PASS"
              << " working_digits="
              << std::numeric_limits<student::WorkingReal>::digits << '\n';
    return 0;
    } catch (const std::runtime_error& error) {
        std::fprintf(stderr, "FAIL: %s: %s\n", stage, error.what());
        return 1;
    }
}

void emit() {
    std::cout << "name,working_p,binary64_p,working_critical,binary64_critical\n";
    std::cout << std::setprecision(17);
    for (const auto& fixture : kStataFixtures) {
        const bool public_level =
            std::string(fixture.name) == "df_1_level_99_9999";
        const double public_tail = (100.0 - 99.9999) / 200.0;
        const double probability = fixture.has_probability
            ? student::two_sided_probability(std::abs(fixture.t), fixture.df)
            : std::numeric_limits<double>::quiet_NaN();
        const double binary_probability = fixture.has_probability
            ? student::two_sided_probability_with_real<double>(
                  std::abs(fixture.t), fixture.df)
            : std::numeric_limits<double>::quiet_NaN();
        const double critical = public_level
            ? student::inverse_survival(public_tail, fixture.df)
            : student::inverse_cdf(fixture.quantile_probability, fixture.df);
        const double binary_critical = public_level
            ? student::inverse_survival_checked<double>(public_tail, fixture.df)
            : student::inverse_cdf_checked<double>(
                  fixture.quantile_probability, fixture.df);
        std::cout << fixture.name << ',' << probability << ','
                  << binary_probability << ',' << critical << ','
                  << binary_critical
                  << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--emit") {
        emit();
        return 0;
    }
    return selftest();
}
