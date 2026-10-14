#ifndef XHDFE_AC_ARENA_STORAGE_HPP
#define XHDFE_AC_ARENA_STORAGE_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace hdfe {
namespace detail {

// Initial adjacency lists share slabs. A grown list owns its buffer so that
// replacement and elimination can reclaim storage without changing edge order.
template<class Edge, std::size_t BlockSlots = (std::size_t{1} << 22)>
class AcArenaStorage {
    static_assert(std::is_trivially_copyable<Edge>::value,
                  "adjacency relocation must preserve every edge value");
    static_assert(BlockSlots > 0, "an initial slab must contain slots");
    struct Block {
        std::unique_ptr<Edge[]> data;
        std::size_t slots;
        int owners;
    };
public:
    struct Statistics {
        std::size_t initial_live_slots = 0;
        std::size_t heap_live_slots = 0;
        std::size_t peak_owned_slots = 0;
        std::size_t initial_allocated_slots = 0;
        std::size_t growths = 0;
        std::size_t copied_slots = 0;
        std::size_t blocks_released = 0;
    };

    explicit AcArenaStorage(int vertices, bool statistics = false)
        : owner_(vertex_count(vertices), -1),
          grown_(vertex_count(vertices)), grown_capacity_(vertex_count(vertices), 0),
          track_(statistics) {}
    AcArenaStorage(const AcArenaStorage&) = delete;
    AcArenaStorage& operator=(const AcArenaStorage&) = delete;

    Edge* initial(int vertex, int capacity) {
        if (sealed_ || vertex < 0 || static_cast<std::size_t>(vertex) >= owner_.size() ||
            capacity < 0 || owner_[static_cast<std::size_t>(vertex)] >= 0 ||
            grown_[static_cast<std::size_t>(vertex)]) {
            throw std::logic_error("invalid MLSMR initial adjacency allocation");
        }
        if (capacity == 0) return nullptr;
        const std::size_t slots = static_cast<std::size_t>(capacity);
        int id;
        Edge* out;
        if (slots > BlockSlots) {
            id = add_block(slots);
            out = blocks_[static_cast<std::size_t>(id)].data.get();
        } else {
            if (current_ < 0 || slots > remaining_) {
                current_ = add_block(BlockSlots);
                cursor_ = blocks_[static_cast<std::size_t>(current_)].data.get();
                remaining_ = BlockSlots;
            }
            id = current_;
            out = cursor_;
            cursor_ += slots;
            remaining_ -= slots;
        }
        owner_[static_cast<std::size_t>(vertex)] = id;
        ++blocks_[static_cast<std::size_t>(id)].owners;
        return out;
    }

    void finish_initial() noexcept {
        sealed_ = true;
        cursor_ = nullptr;
        remaining_ = 0;
        current_ = -1;
    }

    static int growth_capacity(int required) {
        if (required <= 0) throw std::length_error("MLSMR adjacency list is too large");
        int capacity = 1;
        while (capacity < required) {
            if (capacity > std::numeric_limits<int>::max() / 2)
                throw std::length_error("MLSMR adjacency list is too large");
            capacity *= 2;
        }
        return capacity;
    }

    static std::unique_ptr<Edge[]> growth_buffer(int capacity) {
        check_slots(static_cast<std::size_t>(capacity));
        return std::unique_ptr<Edge[]>(new Edge[static_cast<std::size_t>(capacity)]);
    }

    // The caller publishes the new List pointer/capacity before this commit.
    // Allocation and copying have already succeeded; the commit cannot throw.
    void install_growth(int vertex, std::unique_ptr<Edge[]> fresh,
                        int old_capacity, int new_capacity, int copied) noexcept {
        const std::size_t v = static_cast<std::size_t>(vertex);
        assert(fresh && v < owner_.size());
        if (track_) {
            record_peak(static_cast<std::size_t>(new_capacity));
            stats_.heap_live_slots += static_cast<std::size_t>(new_capacity);
            if (grown_[v]) stats_.heap_live_slots -= static_cast<std::size_t>(grown_capacity_[v]);
            ++stats_.growths;
            stats_.copied_slots += static_cast<std::size_t>(copied);
        }
        assert(!grown_[v] || grown_capacity_[v] == old_capacity);
        grown_[v] = std::move(fresh);
        grown_capacity_[v] = new_capacity;
        release_initial(v);
        if (track_) record_peak();
    }

    bool eliminate(int vertex, int capacity) noexcept {
        const std::size_t v = static_cast<std::size_t>(vertex);
        if (owner_[v] >= 0) {
            release_initial(v);
            return true;
        }
        assert(!grown_[v] || grown_capacity_[v] == capacity);
        if (grown_capacity_[v] > 64 && grown_[v]) {
            grown_[v].reset();
            if (track_) stats_.heap_live_slots -= static_cast<std::size_t>(grown_capacity_[v]);
            grown_capacity_[v] = 0;
            return true;
        }
        return false;
    }

    const Statistics& statistics() const noexcept { return stats_; }

    // Offline ownership tests only; never called by an ordinary fit.
    bool owns(int vertex, const Edge* pointer, int capacity) const noexcept {
        if (vertex < 0 || static_cast<std::size_t>(vertex) >= owner_.size() || capacity < 0)
            return false;
        const std::size_t v = static_cast<std::size_t>(vertex);
        if (owner_[v] < 0) return grown_[v].get() == pointer &&
            capacity <= grown_capacity_[v] && (pointer || capacity == 0);
        const Block& block = blocks_[static_cast<std::size_t>(owner_[v])];
        if (!block.data) return false;
        const auto start = reinterpret_cast<std::uintptr_t>(block.data.get());
        const auto at = reinterpret_cast<std::uintptr_t>(pointer);
        const std::size_t bytes = block.slots * sizeof(Edge);
        return at >= start && at - start <= bytes &&
               static_cast<std::size_t>(capacity) <= (bytes - (at - start)) / sizeof(Edge);
    }

    bool owners_valid() const {
        std::vector<int> counts(blocks_.size(), 0);
        for (std::size_t v = 0; v < owner_.size(); ++v) {
            if (owner_[v] < 0) {
                if (static_cast<bool>(grown_[v]) != (grown_capacity_[v] > 0)) return false;
                continue;
            }
            const auto id = static_cast<std::size_t>(owner_[v]);
            if (id >= blocks_.size() || !blocks_[id].data || grown_[v]) return false;
            ++counts[id];
        }
        for (std::size_t id = 0; id < counts.size(); ++id)
            if (counts[id] != blocks_[id].owners ||
                (counts[id] > 0) != static_cast<bool>(blocks_[id].data)) return false;
        return true;
    }

private:
    static std::size_t vertex_count(int vertices) {
        if (vertices < 0) throw std::invalid_argument("negative MLSMR vertex count");
        return static_cast<std::size_t>(vertices);
    }
    static void check_slots(std::size_t slots) {
        if (slots == 0 || slots > std::numeric_limits<std::size_t>::max() / sizeof(Edge))
            throw std::length_error("MLSMR adjacency list is too large");
    }
    int add_block(std::size_t slots) {
        check_slots(slots);
        if (blocks_.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::length_error("too many MLSMR adjacency blocks");
        std::unique_ptr<Edge[]> data(new Edge[slots]);
        blocks_.push_back({std::move(data), slots, 0});
        if (track_) {
            stats_.initial_live_slots += slots;
            stats_.initial_allocated_slots += slots;
            record_peak();
        }
        return static_cast<int>(blocks_.size() - 1);
    }
    void release_initial(std::size_t vertex) noexcept {
        const int id = owner_[vertex];
        if (id < 0) return;
        owner_[vertex] = -1;
        Block& block = blocks_[static_cast<std::size_t>(id)];
        assert(block.owners > 0);
        if (--block.owners == 0) {
            block.data.reset();
            if (track_) {
                stats_.initial_live_slots -= block.slots;
                ++stats_.blocks_released;
            }
        }
    }
    void record_peak(std::size_t transient = 0) noexcept {
        stats_.peak_owned_slots = std::max(stats_.peak_owned_slots,
            stats_.initial_live_slots + stats_.heap_live_slots + transient);
    }
    std::vector<Block> blocks_;
    std::vector<int> owner_;
    std::vector<std::unique_ptr<Edge[]>> grown_;
    std::vector<int> grown_capacity_;
    Edge* cursor_ = nullptr;
    std::size_t remaining_ = 0;
    int current_ = -1;
    bool sealed_ = false;
    bool track_ = false;
    Statistics stats_;
};

}  // namespace detail
}  // namespace hdfe
#endif
