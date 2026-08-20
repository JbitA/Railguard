#include "railguard/clock_alignment.hpp"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>

int main() {
    railguard::UtcMonotonicAligner a(64, 8);
    constexpr std::uint64_t clock_offset = 1'700'000'000'000'000'000ull;
    constexpr std::uint64_t base_mono = 10'000'000'000ull;
    const std::uint64_t delays_ms[] = {8, 5, 4, 7, 3, 6, 4, 5, 3, 4, 50, 5, 4, 3, 6, 4};
    for (std::size_t i=0; i<16; ++i) {
        const auto event_mono = base_mono + i * 20'000'000ull;
        const auto utc = event_mono + clock_offset;
        const auto receive = event_mono + delays_ms[i] * 1'000'000ull;
        a.observe(utc, receive);
    }
    const auto st = a.status();
    assert(st.locked);
    assert(st.samples == 16);
    assert(st.jitter_ms > 0.0);

    const auto target_mono = base_mono + 400'000'000ull;
    const auto mapped = a.monotonic_from_utc(target_mono + clock_offset);
    assert(mapped.has_value());
    // The estimator intentionally retains approximately the minimum serial latency,
    // but must not be fooled by the injected 50 ms transport outlier.
    const auto err_ms = std::abs(static_cast<double>(*mapped - target_mono)) / 1e6;
    assert(err_ms < 8.0);
    std::cout << "clock alignment test: PASS\n";
}
