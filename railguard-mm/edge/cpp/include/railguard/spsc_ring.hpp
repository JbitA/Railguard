#pragma once
#include <array>
#include <atomic>
#include <cstddef>
#include <optional>
#include <utility>

namespace railguard {
template <typename T, std::size_t Capacity>
class SpscRing {
    static_assert(Capacity >= 2);
public:
    bool try_push(T value) noexcept(std::is_nothrow_move_assignable_v<T>) {
        const auto head = head_.load(std::memory_order_relaxed);
        const auto next = increment(head);
        if (next == tail_.load(std::memory_order_acquire)) return false;
        storage_[head] = std::move(value);
        head_.store(next, std::memory_order_release);
        return true;
    }
    std::optional<T> try_pop() noexcept(std::is_nothrow_move_constructible_v<T>) {
        const auto tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) return std::nullopt;
        T value = std::move(storage_[tail]);
        tail_.store(increment(tail), std::memory_order_release);
        return value;
    }
    [[nodiscard]] std::size_t size_approx() const noexcept {
        const auto h=head_.load(std::memory_order_acquire), t=tail_.load(std::memory_order_acquire);
        return h >= t ? h - t : Capacity - (t - h);
    }
private:
    static constexpr std::size_t increment(std::size_t v) noexcept { return (v + 1) % Capacity; }
    std::array<T, Capacity> storage_{};
    alignas(64) std::atomic_size_t head_{0};
    alignas(64) std::atomic_size_t tail_{0};
};
}
