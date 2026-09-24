#ifndef XHDFE_GROUP_FORWARD_CERTIFICATE_HPP
#define XHDFE_GROUP_FORWARD_CERTIFICATE_HPP

#include "fe_absorption.hpp"
#include "group_exact_rank.hpp"
#include "hdfe/ieee_bits.hpp"
#include "hdfe/parallel_work_observer.hpp"
#include <Eigen/QR>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef HDFE_USE_OPENMP
#include <omp.h>
#endif

namespace hdfe {
namespace detail {

inline constexpr std::size_t kGroupForwardCellLimit=8000000;

struct GroupForwardCheck {
    bool eligible = false;
    bool passed = true;
    double worst_ratio = 0.0;
    double limit = 0.0;
    std::string unavailable;
};

struct GroupAffineBound {
    std::vector<long double> norms;
    long double minimum_weight = 1.0L;
};

// Exact binary accumulation is a rare fallback for cancellation in the affine
// witness. Clearing the mean denominator also avoids rounded 1/team_size.
// 34 limbs cover every finite FP64 value, an int-sized multiplier and count.
class GroupExactDyadicSum {
    using Limbs = std::array<std::uint64_t,34>;
    Limbs positive_{}, negative_{};
    bool valid_ = true;
    void add_limb(Limbs& limbs, std::size_t i, std::uint64_t value) {
        while (value) {
            if (i==limbs.size()) { valid_=false; return; }
            const auto before=limbs[i];
            limbs[i]+=value; value=limbs[i]<before ? 1 : 0; ++i;
        }
    }
    void add_shifted(Limbs& limbs, std::uint64_t value, unsigned shift) {
        const auto i=shift/64; const auto offset=shift%64;
        add_limb(limbs,i,value<<offset);
        if (offset) add_limb(limbs,i+1,value>>(64-offset));
    }
public:
    void add(double value, unsigned multiplier=1, bool subtract=false) {
        std::uint64_t bits; std::memcpy(&bits,&value,sizeof(bits));
        const unsigned exponent=static_cast<unsigned>((bits>>52)&2047);
        if (exponent==2047) { valid_=false; return; }
        const std::uint64_t mantissa=(bits&0x000fffffffffffffULL) |
            (exponent ? 0x0010000000000000ULL : 0);
        if (!mantissa || !multiplier) return;
        auto& limbs=((bits>>63)!=static_cast<unsigned>(subtract)) ? negative_ : positive_;
        unsigned shift=exponent ? exponent-1 : 0;
        while (multiplier) {
            if (multiplier&1U) add_shifted(limbs,mantissa,shift);
            multiplier>>=1; ++shift;
        }
    }
    bool exactly_zero() const { return valid_ && positive_ == negative_; }
    long double upper_absolute(unsigned divisor=1) const {
        const auto infinity=std::numeric_limits<long double>::infinity();
        if (!valid_ || !divisor) return infinity;
        int order=0;
        for (int i=33;i>=0 && !order;--i)
            if (positive_[i]!=negative_[i]) order=positive_[i]>negative_[i] ? 1 : -1;
        if (!order) return 0.0L;
        const auto& a=order>0 ? positive_ : negative_;
        const auto& b=order>0 ? negative_ : positive_;
        Limbs difference{}; std::uint64_t borrow=0;
        for (std::size_t i=0;i<a.size();++i) {
            const auto first=a[i]-b[i];
            difference[i]=first-borrow;
            borrow=(a[i]<b[i] || first<borrow) ? 1 : 0;
        }
        long double total=0.0L;
        for (int i=33;i>=0;--i) if (difference[i]) {
            long double coefficient=static_cast<long double>(difference[i]);
            if (std::numeric_limits<long double>::digits<64)
                coefficient=std::nextafter(coefficient,infinity);
            volatile long double term=std::ldexp(coefficient,64*i-1074);
            const long double upper=std::nextafter(static_cast<long double>(term),infinity);
            volatile long double next=total+upper;
            total=std::nextafter(static_cast<long double>(next),infinity);
        }
        volatile long double divided=total/divisor;
        return std::nextafter(static_cast<long double>(divided),infinity);
    }
};

struct GroupForwardSum {
    long double sum = 0.0L, correction = 0.0L;
    void add(long double value) {
        // Volatile intermediates keep the compensation valid under fast-math.
        volatile long double next = sum + value;
        volatile long double first = std::abs(sum) >= std::abs(value) ? sum - next : value - next;
        volatile long double remainder = first + (std::abs(sum) >= std::abs(value) ? value : sum);
        volatile long double update = correction + remainder;
        correction = update;
        sum = next;
    }
    long double value() const { return sum + correction; }
};

struct GroupForwardKeyHash {
    std::size_t operator()(const std::vector<int>& key) const {
        std::uint64_t hash = 1469598103934665603ULL;
        for (int value : key) hash = (hash ^ static_cast<std::uint32_t>(value)) * 1099511628211ULL;
        return static_cast<std::size_t>(hash);
    }
};

struct GroupBasisColumn {
    std::pair<int,int> key;
    PackedRankAccumulator::Row values;
};

struct GroupPatternRows {
    int first = -1;
    std::vector<int> extra;
    void push_back(int row) {
        if (first < 0) first = row;
        else extra.push_back(row);
    }
    std::size_t size() const { return first < 0 ? 0 : 1 + extra.size(); }
    int operator[](std::size_t position) const {
        return position ? extra[position-1] : first;
    }
};

struct GroupForwardWorkspace {
    bool bound = false, patterns_ready = false, had_weights = false;
    GroupIndividualStructure signature;
    std::vector<Eigen::VectorXi> signature_fes;
    Eigen::VectorXd signature_weights;
    std::vector<std::vector<int>> keys;
    std::vector<GroupPatternRows> rows;
    int active_rows = 0;
    std::unique_ptr<ExactRankBudget> budget;
    std::unique_ptr<PackedRankAccumulator> basis;
    std::vector<RankFraction> exact_null;
    bool nullspace_checked = false;
    DisjointNullspace exact_nullspace;
    bool residual_feedback_valid = false;
    Eigen::VectorXd residual_feedback;

    bool matches(const GroupIndividualStructure& gi,
                 const std::vector<Eigen::VectorXi>& fes, const Eigen::VectorXd* weights) const {
        if (!bound || had_weights != (weights != nullptr) ||
            signature.num_groups != gi.num_groups || signature.num_individuals != gi.num_individuals ||
            signature.group_ptr != gi.group_ptr || signature.group_individual != gi.group_individual ||
            signature.group_scale != gi.group_scale || signature_fes.size() != fes.size()) return false;
        for (std::size_t d=0; d<fes.size(); ++d)
            if (signature_fes[d].size() != fes[d].size() ||
                (fes[d].size() && std::memcmp(signature_fes[d].data(),fes[d].data(),
                                             static_cast<std::size_t>(fes[d].size())*sizeof(int)))) return false;
        return !weights || (signature_weights.size() == weights->size() &&
            (!weights->size() || !std::memcmp(signature_weights.data(),weights->data(),
                                              static_cast<std::size_t>(weights->size())*sizeof(double))));
    }

    void bind(const GroupIndividualStructure& gi, const std::vector<Eigen::VectorXi>& fes,
              const Eigen::VectorXd* weights) {
        basis.reset(); budget.reset(); exact_null.clear(); keys.clear(); rows.clear();
        exact_nullspace.clear(); nullspace_checked = false;
        residual_feedback_valid = false; residual_feedback.resize(0);
        patterns_ready = false; active_rows = 0;
        signature.num_groups = gi.num_groups; signature.num_individuals = gi.num_individuals;
        signature.group_ptr = gi.group_ptr; signature.group_individual = gi.group_individual;
        signature.group_scale = gi.group_scale; signature_fes = fes;
        had_weights = weights != nullptr;
        if (weights) signature_weights = *weights; else signature_weights.resize(0);
        bound = true;
    }
};

GroupForwardWorkspace*& group_forward_workspace();

class ScopedGroupForwardWorkspace {
    GroupForwardWorkspace local_;
    GroupForwardWorkspace* previous_;
public:
    explicit ScopedGroupForwardWorkspace(bool correction) : previous_(group_forward_workspace()) {
        if (!correction || !previous_) group_forward_workspace() = &local_;
    }
    ~ScopedGroupForwardWorkspace() { group_forward_workspace() = previous_; }
    ScopedGroupForwardWorkspace(const ScopedGroupForwardWorkspace&) = delete;
    ScopedGroupForwardWorkspace& operator=(const ScopedGroupForwardWorkspace&) = delete;
};

inline void build_sparse_group_basis(
    PackedRankAccumulator& accumulator,
    const std::vector<std::vector<int>>& keys, std::size_t dimensions,
    int individuals, std::size_t cell_limit) {
    std::vector<GroupBasisColumn> columns(individuals);
    for (int i=0; i<individuals; ++i) columns[i].key = {-1,i};
    std::vector<std::unordered_map<int,int>> standard_columns(dimensions);
    std::size_t entries = 0;
    for (int p=0; p<static_cast<int>(keys.size()); ++p) {
        const auto& pattern = keys[p];
        for (std::size_t d=0; d<dimensions; ++d) {
            auto found = standard_columns[d].find(pattern[d+1]);
            if (found == standard_columns[d].end()) {
                const int column = static_cast<int>(columns.size());
                found = standard_columns[d].emplace(pattern[d+1], column).first;
                columns.push_back({{static_cast<int>(d),pattern[d+1]}, {}});
            }
            accumulator.insert(columns[found->second].values, p, pattern[0]);
            ++entries;
        }
        for (std::size_t j=1+dimensions; j<pattern.size(); ++j) {
            accumulator.insert(columns[pattern[j]].values, p, 1);
            ++entries;
        }
        if (entries > cell_limit) throw std::runtime_error("FE-space basis incidence storage limit");
    }
    std::vector<std::size_t> order;
    order.reserve(columns.size());
    for (std::size_t column=0; column<columns.size(); ++column)
        if (!columns[column].values.empty()) order.push_back(column);
    // Eliminate low-degree columns before hub effects to limit exact fill.
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return columns[a].values.size() != columns[b].values.size()
            ? columns[a].values.size() < columns[b].values.size() : columns[a].key < columns[b].key;
    });
    for (std::size_t column : order) accumulator.append(std::move(columns[column].values));
}

inline GroupForwardCheck certify_group_forward_space(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<Eigen::VectorXi>& fes,
    const GroupIndividualStructure& gi,
    const Eigen::VectorXd* weights,
    const HdfeOptions& options,
    const AbsorptionResult& result,
    const GroupAffineBound* affine = nullptr,
    const std::vector<long double>* rhs_targets = nullptr) {
    GroupForwardCheck check;
    const bool profile = std::getenv("XHDFE_GI_CERT_TRACE") != nullptr;
    auto previous_phase = std::chrono::steady_clock::time_point{};
    if (profile) previous_phase = std::chrono::steady_clock::now();
    auto phase = [&](const char* name, int size) {
        if (!profile) return;
        const auto now = std::chrono::steady_clock::now();
        std::fprintf(stderr, "gi_certificate phase=%s size=%d seconds=%.6f\n", name, size,
                     std::chrono::duration<double>(now - previous_phase).count());
        previous_phase = now;
    };
    constexpr std::size_t cell_limit = kGroupForwardCellLimit;
    auto unavailable = [&](const char* reason) {
        check.eligible = true;
        check.passed = false;
        check.worst_ratio = std::numeric_limits<double>::infinity();
        check.unavailable = reason;
        return check;
    };
    const int n = static_cast<int>(y.size());
    if (X.cols() > 4096 || fes.size() > 4096 || n <= 0)
        return unavailable("FE-space verification exceeded its dimension limits; no unchecked estimates");
    const int raw_rhs_count = static_cast<int>(X.cols()) + 1;
    if (rhs_targets && rhs_targets->size()!=static_cast<std::size_t>(raw_rhs_count))
        return unavailable("FE-space verification received inconsistent accuracy targets");
    if (gi.num_groups != n || result.y_tilde.size() != n ||
        result.X_tilde.rows() != n || result.X_tilde.cols() != X.cols()) {
        check.eligible = true;
        check.passed = false;
        check.unavailable = "inconsistent residual dimensions";
        return check;
    }
    GroupForwardWorkspace local_workspace;
    auto& workspace = group_forward_workspace() ? *group_forward_workspace() : local_workspace;
    if (!workspace.matches(gi,fes,weights)) workspace.bind(gi,fes,weights);
    workspace.residual_feedback_valid = false;
    auto& keys = workspace.keys;
    auto& rows = workspace.rows;
    if (!workspace.patterns_ready) {
    // Hashes only select candidates. Full key equality remains mandatory, so
    // collisions never combine distinct FE rows. Store each full key once.
    std::unordered_multimap<std::size_t, int> patterns;
    patterns.reserve(std::min(n,65536));
    std::vector<int> key;
    int active_rows = 0;
    std::size_t key_cells = 0;
    for (int group = 0; group < n; ++group) {
        if (weights && !((*weights)[group] > 0.0)) continue;
        const int begin = gi.group_ptr[group], end = gi.group_ptr[group + 1];
        const int members = end - begin;
        int row_scale = 1;
        if (members <= 0) return unavailable("FE-space verification found an empty group");
        if (gi.group_scale[group] != 1.0) {
            if (gi.group_scale[group] != 1.0 / static_cast<double>(members))
                return unavailable("FE-space verification cannot certify the supplied aggregation scale");
            row_scale = members;
        }
        if (static_cast<std::size_t>(members) + fes.size() > cell_limit)
            return unavailable("FE-space verification exceeded its incidence storage budget");
        key.resize(1 + fes.size() + static_cast<std::size_t>(members));
        key[0] = row_scale;
        for (std::size_t d = 0; d < fes.size(); ++d) key[d + 1] = fes[d][group];
        std::copy(gi.group_individual.begin() + begin, gi.group_individual.begin() + end,
                  key.begin() + 1 + fes.size());
        std::sort(key.begin() + 1 + fes.size(), key.end());
        const auto hash = GroupForwardKeyHash{}(key);
        const auto range = patterns.equal_range(hash);
        int id = -1;
        for (auto found=range.first; found!=range.second; ++found) {
            if (keys[found->second]==key) { id=found->second; break; }
        }
        if (id < 0) {
            if (key.size() > cell_limit - key_cells)
                return unavailable("FE-space verification exceeded its pattern storage budget; no unchecked estimates");
            key_cells += key.size();
            id = static_cast<int>(keys.size());
            patterns.emplace(hash, id);
            keys.push_back(key);
            rows.emplace_back();
        }
        rows[id].push_back(group);
        ++active_rows;
    }
    workspace.active_rows = active_rows;
    workspace.patterns_ready = true;
    }
    const int active_rows = workspace.active_rows;
    const int count = static_cast<int>(keys.size());
    phase("patterns", count);
    if (!count || static_cast<std::size_t>(count) * raw_rhs_count > cell_limit)
        return unavailable("FE-space verification exceeded its moment storage budget; no unchecked estimates");
    Eigen::VectorXd residual_contrast, original_contrast, contrast_beta;
    bool check_contrast = false;
    if (X.cols() > 0) {
        // A fitted residual can be much smaller than y_tilde and X_tilde.
        // This auxiliary contrast only challenges the certificate; its QR
        // coefficients never enter the estimator or determine omitted terms.
        Eigen::MatrixXd design = result.X_tilde;
        Eigen::VectorXd response = result.y_tilde;
        if (weights) {
            const Eigen::VectorXd root_weight = weights->array().sqrt();
            design.array().colwise() *= root_weight.array();
            response.array() *= root_weight.array();
        }
        const Eigen::VectorXd beta = design.colPivHouseholderQr().solve(response);
        if (ieee_all_finite(beta)) {
            contrast_beta=beta;
            residual_contrast = result.y_tilde - result.X_tilde * beta;
            // The roundoff envelope must precede cancellation in y - X beta.
            original_contrast = y.cwiseAbs() + X.cwiseAbs() * beta.cwiseAbs();
            check_contrast = ieee_all_finite(residual_contrast) && ieee_all_finite(original_contrast);
        }
        if (!check_contrast) return unavailable("FE-space residual contrast was not finite");
    }
    const int rhs_count = raw_rhs_count + static_cast<int>(check_contrast);
    phase("contrast", rhs_count);
    if (static_cast<std::size_t>(count) * rhs_count > cell_limit)
        return unavailable("FE-space verification exceeded its contrast storage budget; no unchecked estimates");
    check.eligible = true;
    check.limit = std::max(options.tol, 64.0 * std::numeric_limits<double>::epsilon());
    const long double eps = std::numeric_limits<long double>::epsilon();
    std::vector<GroupForwardSum> original(rhs_count), transformed(rhs_count);
    std::vector<long double> moments(static_cast<std::size_t>(count) * rhs_count), absolute(moments.size());
    std::vector<GroupForwardSum> type_weights(count);
    int capacity = 1;
#ifdef HDFE_USE_OPENMP
    capacity = std::max(1, omp_get_num_procs());
#endif
    const int grain = std::max(1, std::min(2048, active_rows / capacity));
    struct Task { int pattern, begin, end; };
    std::vector<Task> tasks;
    for (int p = 0; p < count; ++p) {
        for (int begin = 0; begin < static_cast<int>(rows[p].size()); begin += grain) {
            tasks.push_back({p, begin, std::min(begin + grain, static_cast<int>(rows[p].size()))});
        }
    }
    const int threads = std::max(1, options.num_threads);
    for (int first_rhs = 0; first_rhs < rhs_count; first_rhs += 8) {
        const int block = std::min(8, rhs_count - first_rhs);
        if (count == active_rows) {
            // No within-pattern reduction is needed for unique FE rows.
            const int chunks = std::min(count, std::max(capacity, (count/2048 + (count%2048 != 0))));
            std::vector<GroupForwardSum> partial(static_cast<std::size_t>(chunks)*2*block);
            if (options.parallel_observer) options.parallel_observer->begin_region(threads);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
            for (int chunk=0; chunk<chunks; ++chunk) {
                if (options.parallel_observer) options.parallel_observer->observe_work();
                auto* values = partial.data()+static_cast<std::size_t>(chunk)*2*block;
                const int begin = static_cast<int>(static_cast<std::int64_t>(count)*chunk/chunks);
                const int end = static_cast<int>(static_cast<std::int64_t>(count)*(chunk+1)/chunks);
                for (int p=begin; p<end; ++p) {
                    const int row = rows[p][0];
                    const long double weight = weights ? (*weights)[row] : 1.0L;
                    if (first_rhs == 0) type_weights[p].add(weight);
                    for (int j=0; j<block; ++j) {
                        const int rhs = first_rhs+j;
                        const long double residual = rhs == raw_rhs_count ? residual_contrast[row] :
                            (rhs ? result.X_tilde(row,rhs-1) : result.y_tilde[row]);
                        const long double input = rhs == raw_rhs_count ? original_contrast[row] :
                            (rhs ? X(row,rhs-1) : y[row]);
                        moments[p*rhs_count+rhs] = weight*residual;
                        absolute[p*rhs_count+rhs] = std::abs(weight*residual);
                        values[2*j].add(weight*input*input);
                        values[2*j+1].add(weight*residual*residual);
                    }
                }
            }
            if (options.parallel_observer) options.parallel_observer->end_region();
            for (int chunk=0; chunk<chunks; ++chunk) for (int j=0; j<block; ++j) {
                const auto* values = partial.data()+static_cast<std::size_t>(chunk)*2*block;
                original[first_rhs+j].add(values[2*j].value());
                transformed[first_rhs+j].add(values[2*j+1].value());
            }
            continue;
        }
        const int stride = 1 + 4 * block;
        std::vector<GroupForwardSum> sums(static_cast<std::size_t>(count) * block), abs_sums(sums.size());
        const std::size_t batch_limit = std::max<std::size_t>(capacity, 16384);
        if (batch_limit > cell_limit / static_cast<std::size_t>(stride))
            return unavailable("FE-space verification cannot provision a reduction batch for runtime capacity");
        for (std::size_t first_task = 0; first_task < tasks.size(); first_task += batch_limit) {
            const std::size_t last_task = std::min(tasks.size(), first_task + batch_limit);
            std::vector<GroupForwardSum> partial((last_task - first_task) * stride);
            if (options.parallel_observer) options.parallel_observer->begin_region(threads);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
            for (int task = static_cast<int>(first_task); task < static_cast<int>(last_task); ++task) {
                if (options.parallel_observer) options.parallel_observer->observe_work();
                auto* values = partial.data() + (static_cast<std::size_t>(task) - first_task) * stride;
                const auto& range = tasks[task];
                for (int position = range.begin; position < range.end; ++position) {
                    const int row = rows[range.pattern][position];
                    const long double weight = weights ? (*weights)[row] : 1.0L;
                    if (first_rhs == 0) values[0].add(weight);
                    for (int j = 0; j < block; ++j) {
                        const int rhs = first_rhs + j;
                        const long double residual = rhs == raw_rhs_count ? residual_contrast[row] :
                            (rhs ? result.X_tilde(row, rhs - 1) : result.y_tilde[row]);
                        const long double input = rhs == raw_rhs_count ? original_contrast[row] :
                            (rhs ? X(row, rhs - 1) : y[row]);
                        values[1 + 4*j].add(weight * residual);
                        values[2 + 4*j].add(std::abs(weight * residual));
                        values[3 + 4*j].add(weight * input * input);
                        values[4 + 4*j].add(weight * residual * residual);
                    }
                }
            }
            if (options.parallel_observer) options.parallel_observer->end_region();
            for (std::size_t task = first_task; task < last_task; ++task) {
                const auto* values = partial.data() + (task - first_task) * stride;
                const int p = tasks[task].pattern;
                if (first_rhs == 0) type_weights[p].add(values[0].value());
                for (int j = 0; j < block; ++j) {
                    sums[p*block + j].add(values[1 + 4*j].value());
                    abs_sums[p*block + j].add(values[2 + 4*j].value());
                    original[first_rhs + j].add(values[3 + 4*j].value());
                    transformed[first_rhs + j].add(values[4 + 4*j].value());
                }
            }
        }
        for (int p = 0; p < count; ++p) for (int j = 0; j < block; ++j) {
            moments[p*rhs_count + first_rhs + j] = sums[p*block + j].value();
            absolute[p*rhs_count + first_rhs + j] = abs_sums[p*block + j].value();
        }
    }
    std::vector<long double> lower_weight(count), targets(rhs_count);
    phase("moments", count);
    GroupForwardSum total_weight;
    long double minimum_weight = std::numeric_limits<long double>::max();
    int maximum_scale = 1;
    for (int p = 0; p < count; ++p) {
        lower_weight[p] = type_weights[p].value() * (1.0L - 16.0L * eps);
        minimum_weight = std::min(minimum_weight, lower_weight[p]);
        total_weight.add(type_weights[p].value());
        maximum_scale = std::max(maximum_scale, keys[p][0]);
    }
    for (int rhs = 0; rhs < rhs_count; ++rhs) {
        targets[rhs] = std::max(
            static_cast<long double>(check.limit) * std::sqrt(std::max(0.0L, transformed[rhs].value() * (1-16*eps))),
            64.0L * std::numeric_limits<double>::epsilon() * std::sqrt(std::max(0.0L, original[rhs].value() * (1-16*eps))));
        if (rhs == raw_rhs_count) {
            targets[rhs] /= std::sqrt(total_weight.value() * (1+16*eps));
        } else if (rhs_targets) {
            const long double requested=(*rhs_targets)[rhs];
            if (!(requested>=0.0L) || !ieee_finite(static_cast<double>(requested)))
                return unavailable("FE-space accuracy target exceeded its numerical range");
            targets[rhs]=std::min(targets[rhs],requested);
        }
    }
    std::vector<long double> affine_bounds(rhs_count,0.0L);
    if (affine) {
        if (affine->norms.size()!=static_cast<std::size_t>(raw_rhs_count) ||
            !(affine->minimum_weight>0.0L))
            return unavailable("inconsistent affine-error bounds; no unchecked estimates");
        std::copy(affine->norms.begin(),affine->norms.end(),affine_bounds.begin());
        if (check_contrast) {
            GroupForwardSum error;
            error.add(affine_bounds[0]);
            for (int j=0;j<contrast_beta.size();++j)
                error.add(std::abs(static_cast<long double>(contrast_beta[j]))*affine_bounds[j+1]);
            affine_bounds[raw_rhs_count]=error.value()/std::sqrt(affine->minimum_weight)*(1+32*eps);
        }
    }
    auto relative_to_target = [&](long double bound, int rhs) {
        // For q=input-D*alpha-residual, total projection error is
        // P*residual-(I-P)*q. The two terms are orthogonal in weighted L2.
        // For the residual infinity target use the safe triangle bound.
        bound=(rhs==raw_rhs_count ? bound+affine_bounds[rhs] :
            std::hypot(bound,affine_bounds[rhs]))*(1+32*eps);
        if (targets[rhs] > 0.0L) return static_cast<double>(bound / targets[rhs]);
        return bound == 0.0L ? 0.0 : std::numeric_limits<double>::infinity();
    };
    auto exact_pattern_projection = [&](const DisjointNullspace* nullspace) {
        // A complete pattern-space description gives an alpha-free witness:
        // M_D input = input - pattern_mean(input) + its exact null projection.
        // This certifies the full residual error, including the affine defect.
        const int dimension=nullspace ? nullspace->dimension : 0;
        std::vector<long double> pattern_weight(count),direction(count,0.0L);
        std::vector<GroupForwardSum> direction_norm(dimension);
        std::vector<long double> maximum(dimension,0.0L),length(dimension);
        for (int p=0;p<count;++p) {
            pattern_weight[p]=type_weights[p].value();
            if (nullspace && nullspace->owner[p]>=0) {
                const int c=nullspace->owner[p];
                direction[p]=nullspace->values[p].value()*keys[p][0]/std::sqrt(pattern_weight[p]);
                maximum[c]=std::max(maximum[c],std::abs(direction[p]));
            }
        }
        for (int p=0;p<count;++p) if (nullspace && nullspace->owner[p]>=0) {
            const int c=nullspace->owner[p];
            direction[p]/=maximum[c]; direction_norm[c].add(direction[p]*direction[p]);
        }
        for (int c=0;c<dimension;++c) length[c]=std::sqrt(direction_norm[c].value());
        for (int p=0;p<count;++p) if (nullspace && nullspace->owner[p]>=0)
            direction[p]/=length[nullspace->owner[p]];
        std::vector<long double> means(static_cast<std::size_t>(count)*raw_rhs_count),mean_error(means.size());
        if (options.parallel_observer) options.parallel_observer->begin_region(threads);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (int p=0;p<count;++p) {
            if (options.parallel_observer) options.parallel_observer->observe_work();
            for (int rhs=0;rhs<raw_rhs_count;++rhs) {
                GroupForwardSum sum,magnitude;
                for (std::size_t position=0;position<rows[p].size();++position) {
                    const int row=rows[p][position];
                    const long double weight=weights ? (*weights)[row] : 1.0L;
                    const long double value=rhs ? X(row,rhs-1) : y[row];
                    sum.add(weight*value); magnitude.add(std::abs(weight*value));
                }
                const auto at=static_cast<std::size_t>(p)*raw_rhs_count+rhs;
                means[at]=sum.value()/pattern_weight[p];
                mean_error[at]=128*eps*magnitude.value()/lower_weight[p];
            }
        }
        if (options.parallel_observer) options.parallel_observer->end_region();
        std::vector<GroupForwardSum> coefficients(static_cast<std::size_t>(dimension)*raw_rhs_count),coefficient_error(coefficients.size());
        for (int p=0;p<count;++p) if (nullspace && nullspace->owner[p]>=0) {
            const int c=nullspace->owner[p];
            const long double factor=direction[p]*std::sqrt(pattern_weight[p]);
            for (int rhs=0;rhs<raw_rhs_count;++rhs) {
                const auto at=static_cast<std::size_t>(p)*raw_rhs_count+rhs;
                const auto out=static_cast<std::size_t>(c)*raw_rhs_count+rhs;
                coefficients[out].add(factor*means[at]);
                coefficient_error[out].add(std::abs(factor)*(mean_error[at]+128*eps*std::abs(means[at])));
            }
        }
        // Fixed chunks preserve reduction order independently of the team.
        const int chunks=std::min(count,std::max(capacity,count/2048+(count%2048!=0)));
        std::vector<GroupForwardSum> partial(static_cast<std::size_t>(chunks)*raw_rhs_count);
        std::vector<long double> contrast_max(chunks,0.0L);
        if (options.parallel_observer) options.parallel_observer->begin_region(threads);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (int chunk=0;chunk<chunks;++chunk) {
            if (options.parallel_observer) options.parallel_observer->observe_work();
            const int begin=static_cast<int>(static_cast<std::int64_t>(count)*chunk/chunks);
            const int end=static_cast<int>(static_cast<std::int64_t>(count)*(chunk+1)/chunks);
            for (int p=begin;p<end;++p) for (std::size_t position=0;position<rows[p].size();++position) {
                const int row=rows[p][position];
                const long double weight=weights ? (*weights)[row] : 1.0L;
                GroupForwardSum contrast,contrast_error;
                for (int rhs=0;rhs<raw_rhs_count;++rhs) {
                    const auto at=static_cast<std::size_t>(p)*raw_rhs_count+rhs;
                    long double null_value=0.0L,null_error=0.0L;
                    if (nullspace && nullspace->owner[p]>=0) {
                        const auto c=static_cast<std::size_t>(nullspace->owner[p])*raw_rhs_count+rhs;
                        const long double factor=direction[p]/std::sqrt(pattern_weight[p]);
                        null_value=factor*coefficients[c].value();
                        null_error=std::abs(factor)*coefficient_error[c].value()+128*eps*std::abs(null_value);
                    }
                    const long double input=rhs ? X(row,rhs-1) : y[row];
                    const long double residual=rhs ? result.X_tilde(row,rhs-1) : result.y_tilde[row];
                    GroupForwardSum difference;
                    difference.add(residual); difference.add(-input); difference.add(means[at]); difference.add(-null_value);
                    const long double error=difference.value();
                    const long double uncertainty=mean_error[at]+null_error+128*eps*
                        (std::abs(residual)+std::abs(input)+std::abs(means[at])+std::abs(null_value));
                    const long double upper=(std::abs(error)+uncertainty)*(1+32*eps);
                    partial[static_cast<std::size_t>(chunk)*raw_rhs_count+rhs].add(weight*upper*upper);
                    if (check_contrast) {
                        const long double beta=rhs ? -static_cast<long double>(contrast_beta[rhs-1]) : 1.0L;
                        contrast.add(beta*error);
                        contrast_error.add(std::abs(beta)*(uncertainty+64*eps*std::abs(error)));
                    }
                }
                if (check_contrast) contrast_max[chunk]=std::max(contrast_max[chunk],
                    (std::abs(contrast.value())+contrast_error.value())*(1+64*eps));
            }
        }
        if (options.parallel_observer) options.parallel_observer->end_region();
        check.passed=true; check.worst_ratio=0.0;
        for (int rhs=0;rhs<rhs_count;++rhs) {
            long double bound=0.0L;
            if (rhs==raw_rhs_count) bound=*std::max_element(contrast_max.begin(),contrast_max.end());
            else {
                GroupForwardSum squared;
                for (int chunk=0;chunk<chunks;++chunk) squared.add(partial[static_cast<std::size_t>(chunk)*raw_rhs_count+rhs].value());
                bound=std::sqrt(std::max(0.0L,squared.value()))*(1+32*eps);
            }
            const double ratio=targets[rhs]>0 ? static_cast<double>(bound/targets[rhs]) :
                (bound==0 ? 0.0 : std::numeric_limits<double>::infinity());
            if (profile) std::fprintf(stderr,"gi_certificate direct_rhs=%d ratio=%.9e target=%.9Le\n",rhs,ratio,targets[rhs]);
            check.passed=check.passed && ieee_finite(ratio) && ratio<=1.0;
            check.worst_ratio=std::max(check.worst_ratio,ratio);
        }
        return check;
    };
    // Orthogonality to every row-pattern intercept is a cheap sufficient test.
    bool pattern_pass = true;
    for (int rhs = 0; rhs < rhs_count; ++rhs) {
        GroupForwardSum norm;
        long double maximum = 0.0L;
        for (int p = 0; p < count; ++p) {
            const long double bound = std::abs(moments[p*rhs_count + rhs]) + 32*eps*absolute[p*rhs_count + rhs];
            norm.add(bound * bound / lower_weight[p]);
            maximum = std::max(maximum, bound / lower_weight[p]);
        }
        const long double bound = rhs == raw_rhs_count ? maximum :
            std::sqrt(std::max(0.0L, norm.value()));
        const double ratio = relative_to_target(bound * (1+16*eps), rhs);
        pattern_pass = pattern_pass && ieee_finite(ratio) && ratio <= 1.0;
        check.worst_ratio = std::max(check.worst_ratio, ratio);
    }
    if (pattern_pass) return check;

    try {
        if (!workspace.basis) {
            auto budget = std::make_unique<ExactRankBudget>();
            auto basis = std::make_unique<PackedRankAccumulator>(count,*budget,cell_limit);
            build_sparse_group_basis(*basis,keys,fes.size(),gi.num_individuals,cell_limit);
            workspace.budget = std::move(budget);
            workspace.basis = std::move(basis);
        }
        auto& accumulator = *workspace.basis;
        phase("basis", accumulator.rank());
        const auto& basis = accumulator.echelon_basis();
        if (basis.size() == static_cast<std::size_t>(count)) {
            if (affine) return exact_pattern_projection(nullptr);
            // Full pattern rank makes the earlier pattern test necessary too.
            check.passed = false;
            return check;
        }
        if (!workspace.nullspace_checked) {
            try { workspace.exact_nullspace = accumulator.disjoint_nullspace(count); }
            catch (const std::runtime_error&) { workspace.exact_nullspace.clear(); }
            workspace.nullspace_checked = true;
        }
        if (!workspace.exact_nullspace.empty()) {
            if (affine) return exact_pattern_projection(&workspace.exact_nullspace);
            // Exact, disjoint null vectors are orthogonal after row scaling.
            // Other designs retain the mandatory implicit-basis bound below.
            const int dimension = workspace.exact_nullspace.dimension;
            std::vector<long double> null_vector(count, 0.0L);
            std::vector<int> owner(count,-1);
            std::vector<long double> maximum(dimension,0.0L), length(dimension);
            std::vector<GroupForwardSum> norm(dimension);
            for (int p=0; p<count; ++p) {
                const int j=workspace.exact_nullspace.owner[p];
                const auto& value = workspace.exact_nullspace.values[p];
                if (j<0) continue;
                owner[p] = j;
                null_vector[p] = value.value()*keys[p][0]/std::sqrt(lower_weight[p]);
                maximum[j] = std::max(maximum[j],std::abs(null_vector[p]));
            }
            for (int p=0; p<count; ++p) if (owner[p]>=0) {
                null_vector[p] /= maximum[owner[p]];
                norm[owner[p]].add(null_vector[p]*null_vector[p]);
            }
            for (int j=0; j<dimension; ++j) length[j] = std::sqrt(norm[j].value());
            for (int p=0; p<count; ++p) if (owner[p]>=0) null_vector[p] /= length[owner[p]];
            check.passed = true;
            check.worst_ratio = 0.0;
            for (int rhs = 0; rhs < rhs_count; ++rhs) {
                std::vector<GroupForwardSum> coefficient(dimension);
                GroupForwardSum magnitude, projection_error;
                long double maximum = 0.0L;
                for (int p = 0; p < count; ++p) {
                    const long double value = moments[p*rhs_count + rhs] / std::sqrt(lower_weight[p]);
                    const long double abs_value = absolute[p*rhs_count + rhs] / std::sqrt(lower_weight[p]);
                    if (owner[p]>=0) coefficient[owner[p]].add(null_vector[p] * value);
                    magnitude.add(abs_value * abs_value);
                }
                for (int p = 0; p < count; ++p) {
                    const long double value = moments[p*rhs_count + rhs] / std::sqrt(lower_weight[p]);
                    const long double error = value - (owner[p]>=0 ?
                        null_vector[p] * coefficient[owner[p]].value() : 0.0L);
                    projection_error.add(error * error);
                    maximum = std::max(maximum, std::abs(error) / std::sqrt(lower_weight[p]));
                }
                long double rounding = 128*eps*std::sqrt(std::max(0.0L, magnitude.value()));
                if (rhs == raw_rhs_count) rounding /= std::sqrt(minimum_weight);
                const long double bound = (rhs == raw_rhs_count ? maximum :
                    std::sqrt(std::max(0.0L, projection_error.value()))) * (1+16*eps) + rounding;
                const double ratio = relative_to_target(bound, rhs);
                check.passed = check.passed && ieee_finite(ratio) && ratio <= 1.0;
                check.worst_ratio = std::max(check.worst_ratio, ratio);
            }
            return check;
        }
        check.worst_ratio = 0.0;
        check.passed = true;
        std::vector<long double> reduced_moments(static_cast<std::size_t>(count) * rhs_count);
        std::vector<long double> reduced_errors(reduced_moments.size());
        std::vector<double> ratios(rhs_count);
        if (options.parallel_observer) options.parallel_observer->begin_region(threads);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (int rhs = 0; rhs < rhs_count; ++rhs) {
            if (options.parallel_observer) options.parallel_observer->observe_work();
            GroupForwardSum norm;
            auto* reduced_moment = reduced_moments.data() + static_cast<std::size_t>(rhs) * count;
            auto* reduced_error = reduced_errors.data() + static_cast<std::size_t>(rhs) * count;
            // If U is the echelon basis and H its unit upper-triangular pivot
            // block, B=H^-1 U has an identity pivot block. Apply this inverse
            // to the moments with propagated roundoff bounds, without forming B.
            for (auto row = basis.rbegin(); row != basis.rend(); ++row) {
                GroupForwardSum moment, magnitude;
                for (const auto& entry : row->second) {
                    const int p = entry.first;
                    const long double coefficient = entry.second.value() / keys[p][0];
                    moment.add(coefficient * moments[p*rhs_count + rhs]);
                    magnitude.add(std::abs(coefficient) * absolute[p*rhs_count + rhs]);
                }
                GroupForwardSum propagated;
                for (auto entry = std::next(row->second.begin()); entry != row->second.end(); ++entry) {
                    const int p = entry->first;
                    if (!accumulator.is_pivot(p)) continue;
                    const long double coefficient = entry->second.value();
                    moment.add(-coefficient * reduced_moment[p]);
                    magnitude.add(std::abs(coefficient * reduced_moment[p]));
                    propagated.add(std::abs(coefficient) * reduced_error[p]);
                }
                reduced_moment[row->first] = moment.value();
                reduced_error[row->first] = (propagated.value() + 64*eps*magnitude.value()) * (1+16*eps);
                const long double upper = std::abs(moment.value()) + reduced_error[row->first];
                norm.add(upper * upper);
            }
            // RREF has an identity pivot block, hence sigma_min(B') >= 1.
            // W^.5 S^-1 B' therefore has sigma_min >= sqrt(min W)/max S.
            long double bound = std::sqrt(std::max(0.0L, norm.value())) * maximum_scale /
                                      std::sqrt(minimum_weight) * (1+32*eps);
            if (rhs == raw_rhs_count) bound /= std::sqrt(minimum_weight);
            ratios[rhs] = relative_to_target(bound, rhs);
        }
        if (options.parallel_observer) options.parallel_observer->end_region();
        for (int rhs=0; rhs<rhs_count; ++rhs) {
            const double ratio = ratios[rhs];
            if (profile) std::fprintf(stderr,"gi_certificate rhs=%d ratio=%.9e target=%.9Le\n",
                                      rhs,ratio,targets[rhs]);
            check.passed = check.passed && ieee_finite(ratio) && ratio <= 1.0;
            check.worst_ratio = std::max(check.worst_ratio, ratio);
        }
        // Feedback belongs only to this just-checked candidate. If all X
        // columns are already certified, correct the fitted residual while
        // retaining X; the composed y/X must still pass every original gate.
        bool x_certified = check_contrast;
        for (int rhs=1; rhs<raw_rhs_count; ++rhs)
            x_certified = x_certified && ieee_finite(ratios[rhs]) && ratios[rhs] <= 1.0;
        if (!check.passed && x_certified) {
            workspace.residual_feedback = residual_contrast;
            workspace.residual_feedback_valid = true;
        }
    } catch (const std::runtime_error& error) {
        check.passed = false;
        check.unavailable = error.what();
    }
    phase("bound", count);
    return check;
}

inline bool deflate_group_within_patterns(
    const Eigen::VectorXd& y, const Eigen::MatrixXd& X,
    const std::vector<Eigen::VectorXi>& fes, const GroupIndividualStructure& gi,
    const Eigen::VectorXd* weights, Eigen::VectorXd& deflated_y, Eigen::MatrixXd& deflated_X) {
    const int n = static_cast<int>(y.size());
    if (n > 65536 || X.cols() > 128) return false;
    const int rhs_count = static_cast<int>(X.cols()) + 1;
    std::unordered_map<std::vector<int>, int, GroupForwardKeyHash> patterns;
    std::vector<int> row_pattern(n), key;
    std::vector<GroupForwardSum> totals, sums;
    for (int row = 0; row < n; ++row) {
        const int begin = gi.group_ptr[row], end = gi.group_ptr[row+1];
        const int members = end - begin;
        int scale = 1;
        if (gi.group_scale[row] != 1.0) {
            if (members <= 0 || gi.group_scale[row] != 1.0 / members) return false;
            scale = members;
        }
        key.resize(1 + fes.size() + static_cast<std::size_t>(members));
        key[0] = scale;
        for (std::size_t d=0; d<fes.size(); ++d) key[d+1] = fes[d][row];
        std::copy(gi.group_individual.begin()+begin, gi.group_individual.begin()+end, key.begin()+1+fes.size());
        std::sort(key.begin()+1+fes.size(), key.end());
        auto found = patterns.find(key);
        if (found == patterns.end()) {
            if (patterns.size() >= 8000000U/static_cast<std::size_t>(rhs_count)) return false;
            const int id = static_cast<int>(patterns.size());
            found = patterns.emplace(key,id).first;
            totals.emplace_back(); sums.resize(patterns.size()*rhs_count);
        }
        const int p = found->second;
        row_pattern[row] = p;
        const long double weight = weights ? (*weights)[row] : 1.0L;
        totals[p].add(weight);
        sums[p*rhs_count].add(weight*y[row]);
        for (int rhs=1; rhs<rhs_count; ++rhs) sums[p*rhs_count+rhs].add(weight*X(row,rhs-1));
    }
    std::vector<long double> null_vector(patterns.size(), 0.0L), null_coefficients(rhs_count, 0.0L);
    std::vector<int> null_owner(patterns.size(),-1);
    try {
        ExactRankBudget budget;
        PackedRankAccumulator accumulator(static_cast<int>(patterns.size()), budget, 2000000U);
        std::vector<std::vector<int>> keys(patterns.size());
        std::vector<int> scales(patterns.size());
        for (const auto& entry : patterns) {
            const auto& pattern = entry.first;
            const int p = entry.second;
            scales[p] = pattern[0];
            keys[p] = pattern;
        }
        build_sparse_group_basis(accumulator, keys, fes.size(), gi.num_individuals, 8000000U);
        const auto exact_nullspace = accumulator.disjoint_nullspace(static_cast<int>(patterns.size()));
        if (!exact_nullspace.empty()) {
            // Remove disjoint exact null directions too. They cannot change
            // D'W*RHS, but would otherwise dominate a nearly converged correction.
            const int dimension = exact_nullspace.dimension;
            bool positive = true;
            for (const auto& total : totals) positive = positive && total.value() > 0.0L;
            if (positive) {
                std::vector<long double> maximum(dimension,0.0L);
                std::vector<GroupForwardSum> norm(dimension);
                null_coefficients.assign(dimension*rhs_count,0.0L);
                for (std::size_t p=0; p<patterns.size(); ++p) {
                    const int j=exact_nullspace.owner[p];
                    if (j<0) continue;
                    null_owner[p] = j;
                    null_vector[p] = exact_nullspace.values[p].value()*scales[p]/totals[p].value();
                    maximum[j] = std::max(maximum[j],std::abs(null_vector[p]));
                }
                for (std::size_t p=0; p<patterns.size(); ++p) if (null_owner[p]>=0) {
                    null_vector[p] /= maximum[null_owner[p]];
                    norm[null_owner[p]].add(totals[p].value()*null_vector[p]*null_vector[p]);
                }
                for (int rhs=0; rhs<rhs_count; ++rhs) {
                    std::vector<GroupForwardSum> moment(dimension);
                    for (std::size_t p=0; p<patterns.size(); ++p)
                        if (null_owner[p]>=0) moment[null_owner[p]].add(null_vector[p]*sums[p*rhs_count+rhs].value());
                    for (int j=0; j<dimension; ++j)
                        null_coefficients[j*rhs_count+rhs] = moment[j].value()/norm[j].value();
                }
            }
        }
    } catch (const std::runtime_error&) {
        // Within-pattern deflation remains valid without this optional step;
        // the final certificate still applies all its independent rank limits.
        std::fill(null_vector.begin(), null_vector.end(), 0.0L);
        std::fill(null_owner.begin(),null_owner.end(),-1);
    }
    deflated_y = y; deflated_X = X;
    for (int row=0; row<n; ++row) {
        const int p = row_pattern[row];
        const long double total = totals[p].value();
        if (!(total > 0.0L)) continue;
        deflated_y[row] = static_cast<double>(sums[p*rhs_count].value()/total -
            (null_owner[p]>=0 ? null_vector[p]*null_coefficients[null_owner[p]*rhs_count] : 0.0L));
        for (int rhs=1; rhs<rhs_count; ++rhs)
            deflated_X(row,rhs-1) = static_cast<double>(sums[p*rhs_count+rhs].value()/total -
                (null_owner[p]>=0 ? null_vector[p]*null_coefficients[null_owner[p]*rhs_count+rhs] : 0.0L));
    }
    return true;
}

}  // namespace detail
}  // namespace hdfe
#endif
