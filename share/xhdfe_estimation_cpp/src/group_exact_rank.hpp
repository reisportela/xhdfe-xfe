#ifndef XHDFE_GROUP_EXACT_RANK_HPP
#define XHDFE_GROUP_EXACT_RANK_HPP

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hdfe {
namespace detail {

// Group membership becomes an integer design after clearing the mean row
// denominators. Checked rational elimination certifies its algebraic rank;
// floating pivot thresholds cannot distinguish exact from near dependence.
class RankFraction {
public:
    RankFraction(std::int64_t numerator = 0, std::int64_t denominator = 1)
        : numerator_(numerator), denominator_(denominator) {
        if (denominator <= 0 || numerator == std::numeric_limits<std::int64_t>::min()) fail();
        if (numerator == 0) {
            denominator_ = 1;
        } else if (denominator_ != 1) {
            const auto gcd = std::gcd(std::abs(numerator_), denominator_);
            numerator_ /= gcd;
            denominator_ /= gcd;
        }
    }

    bool zero() const { return numerator_ == 0; }
    long double value() const {
        if (denominator_ == 1) return static_cast<long double>(numerator_);
        return static_cast<long double>(numerator_) / static_cast<long double>(denominator_);
    }

    RankFraction operator*(const RankFraction& other) const {
        if (zero() || other.zero()) return {};
        if (denominator_ == 1 && numerator_ == 1) return other;
        if (other.denominator_ == 1 && other.numerator_ == 1) return *this;
        if (denominator_ == 1 && numerator_ == -1) return RankFraction(-other.numerator_, other.denominator_);
        if (other.denominator_ == 1 && other.numerator_ == -1) return RankFraction(-numerator_, denominator_);
        const auto first = std::gcd(std::abs(numerator_), other.denominator_);
        const auto second = std::gcd(std::abs(other.numerator_), denominator_);
        return RankFraction(multiply(numerator_ / first, other.numerator_ / second),
                            multiply(denominator_ / second, other.denominator_ / first));
    }

    RankFraction operator/(const RankFraction& other) const {
        if (other.zero()) fail();
        if (other.denominator_ == 1 && other.numerator_ == 1) return *this;
        return *this * RankFraction(other.numerator_ < 0 ? -other.denominator_ : other.denominator_,
                                    std::abs(other.numerator_));
    }

    RankFraction operator-(const RankFraction& other) const {
        if (other.zero()) return *this;
        if (zero()) return RankFraction(-other.numerator_, other.denominator_);
        if (denominator_ == other.denominator_) {
            if (numerator_ == other.numerator_) return {};
            return RankFraction(add(numerator_, -other.numerator_), denominator_);
        }
        const auto gcd = std::gcd(denominator_, other.denominator_);
        const auto left = multiply(numerator_, other.denominator_ / gcd);
        const auto right = multiply(other.numerator_, denominator_ / gcd);
        return RankFraction(add(left, -right), multiply(denominator_ / gcd, other.denominator_));
    }

private:
    static void fail() {
        throw std::runtime_error(
            "dofadjustments(exact): rational rank exceeded the checked integer range; "
            "no estimates returned and no approximate rank fallback");
    }

    static std::int64_t multiply(std::int64_t a, std::int64_t b) {
        constexpr auto limit = std::numeric_limits<std::int64_t>::max();
        if (a && b && std::abs(a) > limit / std::abs(b)) fail();
        return a * b;
    }

    static std::int64_t add(std::int64_t a, std::int64_t b) {
        constexpr auto limit = std::numeric_limits<std::int64_t>::max();
        if ((b > 0 && a > limit - b) || (b < 0 && a < -limit - b)) fail();
        return a + b;
    }

    std::int64_t numerator_;
    std::int64_t denominator_;
};

struct ExactRankBudget {
    std::uint64_t operations = 0;
    void step() {
        if (++operations > 20000000ULL) {
            throw std::runtime_error(
                "dofadjustments(exact): rational rank exceeded the operation limit; "
                "no estimates returned and no approximate rank fallback");
        }
    }
};

class ExactRankAccumulator {
public:
    using Row = std::map<int, RankFraction>;

    explicit ExactRankAccumulator(ExactRankBudget& budget,
                                  std::size_t maximum_entries = 2000000U)
        : budget_(budget), maximum_entries_(maximum_entries) {}

    void insert(Row& row, int column, std::int64_t value) {
        budget_.step();
        if (value && !row.emplace(column, RankFraction(value)).second) {
            throw std::runtime_error(
                "dofadjustments(exact): rational rank found a duplicate incidence column; "
                "no estimates returned");
        }
        check_storage(row.size());
    }

    void append(Row row) {
        while (!row.empty()) {
            const int column = row.begin()->first;
            const RankFraction factor = row.begin()->second;
            const auto pivot = pivots_.find(column);
            if (pivot == pivots_.end()) {
                for (auto& entry : row) {
                    budget_.step();
                    entry.second = entry.second / factor;
                }
                check_storage(row.size());
                stored_entries_ += row.size();
                pivots_.emplace(column, std::move(row));
                return;
            }
            for (const auto& entry : pivot->second) {
                budget_.step();
                const auto position = row.find(entry.first);
                const RankFraction original = position == row.end() ? RankFraction{} : position->second;
                const RankFraction difference = original - factor * entry.second;
                if (difference.zero()) {
                    if (position != row.end()) row.erase(position);
                } else if (position == row.end()) {
                    row.emplace(entry.first, difference);
                    check_storage(row.size());
                } else {
                    position->second = difference;
                }
            }
        }
    }

    int rank() const { return static_cast<int>(pivots_.size()); }

    const std::map<int, Row>& echelon_basis() const { return pivots_; }

    std::vector<RankFraction> null_vector(int columns) {
        if (rank() + 1 != columns) {
            throw std::runtime_error("exact null-vector construction requires nullity one");
        }
        int free_column = 0;
        while (pivots_.count(free_column)) ++free_column;
        std::vector<RankFraction> values(columns);
        values[free_column] = RankFraction(1);
        for (auto row = pivots_.rbegin(); row != pivots_.rend(); ++row) {
            RankFraction value;
            for (auto entry = std::next(row->second.begin()); entry != row->second.end(); ++entry) {
                budget_.step();
                value = value - entry->second * values[entry->first];
            }
            values[row->first] = value;
        }
        return values;
    }

    const std::map<int, Row>& reduced_basis() {
        for (auto pivot = pivots_.rbegin(); pivot != pivots_.rend(); ++pivot) {
            for (auto prior = pivots_.begin(); prior->first < pivot->first; ++prior) {
                auto coefficient = prior->second.find(pivot->first);
                if (coefficient == prior->second.end()) continue;
                const RankFraction factor = coefficient->second;
                for (const auto& entry : pivot->second) {
                    budget_.step();
                    auto position = prior->second.find(entry.first);
                    const RankFraction original = position == prior->second.end() ? RankFraction{} : position->second;
                    const RankFraction difference = original - factor * entry.second;
                    if (difference.zero()) {
                        if (position != prior->second.end()) {
                            prior->second.erase(position);
                            --stored_entries_;
                        }
                    } else if (position == prior->second.end()) {
                        prior->second.emplace(entry.first, difference);
                        ++stored_entries_;
                        check_storage(0);
                    } else {
                        position->second = difference;
                    }
                }
            }
        }
        return pivots_;
    }

private:
    void check_storage(std::size_t row_entries) const {
        if (stored_entries_ + row_entries > maximum_entries_) {
            throw std::runtime_error(
                "dofadjustments(exact): rational rank exceeded the sparse storage limit "
                "(stored=" + std::to_string(stored_entries_) +
                ", pending=" + std::to_string(row_entries) +
                ", limit=" + std::to_string(maximum_entries_) +
                ", rank=" + std::to_string(pivots_.size()) +
                ", operations=" + std::to_string(budget_.operations) + "); "
                "no estimates returned and no approximate rank fallback");
        }
    }

    ExactRankBudget& budget_;
    std::size_t maximum_entries_;
    std::size_t stored_entries_ = 0;
    std::map<int, Row> pivots_;
};

struct DisjointNullspace {
    int dimension = 0;
    std::vector<int> owner;
    std::vector<RankFraction> values;
    bool empty() const { return dimension == 0; }
    void clear() { dimension = 0; owner.clear(); values.clear(); }
};

// The forward verifier builds its incidence columns in increasing row order.
// Contiguous rows avoid one allocation/tree lookup per rational coefficient.
class PackedRankAccumulator {
public:
    using Row = std::vector<std::pair<int, RankFraction>>;
    PackedRankAccumulator(int columns, ExactRankBudget& budget, std::size_t limit)
        : budget_(budget), limit_(limit), lookup_(columns, nullptr) {}
    PackedRankAccumulator(const PackedRankAccumulator&) = delete;
    PackedRankAccumulator& operator=(const PackedRankAccumulator&) = delete;

    void insert(Row& row, int column, std::int64_t value) {
        budget_.step();
        if (!value) return;
        if (!row.empty() && row.back().first >= column)
            throw std::runtime_error("exact sparse basis requires strictly ordered incidence rows");
        row.emplace_back(column, RankFraction(value));
        check_storage(row.size());
    }

    void append(Row row) {
        Row next;
        while (!row.empty()) {
            const int pivot = row.front().first;
            const RankFraction factor = row.front().second;
            if (!lookup_[pivot]) {
                for (auto& entry : row) {
                    budget_.step();
                    entry.second = entry.second / factor;
                }
                check_storage(row.size());
                stored_ += row.size();
                lookup_[pivot] = &pivots_.emplace(pivot, std::move(row)).first->second;
                return;
            }
            const Row& prior = *lookup_[pivot];
            next.clear();
            next.reserve(std::min(limit_ - stored_ + 1, row.size() + prior.size()));
            std::size_t a = 0, b = 0;
            while (a < row.size() || b < prior.size()) {
                if (b == prior.size() || (a < row.size() && row[a].first < prior[b].first)) {
                    next.push_back(row[a++]);
                } else {
                    budget_.step();
                    const int column = prior[b].first;
                    RankFraction original;
                    if (a < row.size() && row[a].first == column) original = row[a++].second;
                    const RankFraction difference = original - factor * prior[b++].second;
                    if (!difference.zero()) next.emplace_back(column, difference);
                }
                check_storage(next.size());
            }
            row.swap(next);
        }
    }

    int rank() const { return static_cast<int>(pivots_.size()); }
    const std::map<int, Row>& echelon_basis() const { return pivots_; }
    bool is_pivot(int column) const { return lookup_[column] != nullptr; }

    DisjointNullspace disjoint_nullspace(int columns) {
        const int dimension = columns-rank();
        if (dimension <= 0) return {};
        // At the final pivot every following coordinate is free. Two such
        // entries already prove overlap, without allocating a large basis.
        if (!pivots_.empty() && pivots_.rbegin()->second.size()>2) return {};
        DisjointNullspace result;
        result.owner.assign(columns,-1);
        result.values.resize(columns);
        int next=0;
        for (int free=0; free<columns; ++free) if (!lookup_[free]) {
            result.owner[free]=next++;
            result.values[free]=RankFraction(1);
        }
        for (auto row=pivots_.rbegin(); row!=pivots_.rend(); ++row) {
            std::map<int,RankFraction> terms;
            for (auto entry=std::next(row->second.begin()); entry!=row->second.end(); ++entry) {
                const int owner=result.owner[entry->first];
                if (owner<0) continue;
                budget_.step();
                const auto found=terms.find(owner);
                const RankFraction prior=found==terms.end() ? RankFraction{} : found->second;
                const RankFraction value=prior-entry->second*result.values[entry->first];
                if (value.zero()) {
                    if (found!=terms.end()) terms.erase(found);
                } else terms[owner]=value;
            }
            if (terms.size()>1) return {};
            if (!terms.empty()) {
                result.owner[row->first]=terms.begin()->first;
                result.values[row->first]=terms.begin()->second;
            }
        }
        result.dimension=dimension;
        return result;
    }

    std::vector<RankFraction> null_vector(int columns) {
        if (rank() + 1 != columns)
            throw std::runtime_error("exact null-vector construction requires nullity one");
        int free_column = 0;
        while (lookup_[free_column]) ++free_column;
        std::vector<RankFraction> values(columns);
        values[free_column] = RankFraction(1);
        for (auto row = pivots_.rbegin(); row != pivots_.rend(); ++row) {
            RankFraction value;
            for (auto entry = std::next(row->second.begin()); entry != row->second.end(); ++entry) {
                budget_.step();
                value = value - entry->second * values[entry->first];
            }
            values[row->first] = value;
        }
        return values;
    }

private:
    void check_storage(std::size_t pending) const {
        if (stored_ + pending > limit_)
            throw std::runtime_error("exact sparse basis exceeded its checked storage budget; no unchecked estimates");
    }
    ExactRankBudget& budget_;
    std::size_t limit_, stored_ = 0;
    std::map<int, Row> pivots_;
    std::vector<Row*> lookup_;
};

}  // namespace detail
}  // namespace hdfe

#endif
