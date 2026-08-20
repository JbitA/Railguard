#include "railguard/protocol.hpp"
#include <algorithm>
#include <bit>
#include <cstring>
#include <cmath>
#include <type_traits>

namespace railguard {
namespace {
template <typename T>
void append_le(std::vector<std::uint8_t>& out, T value) {
    static_assert(std::is_trivially_copyable_v<T>);
    std::array<std::uint8_t, sizeof(T)> bytes{};
    std::memcpy(bytes.data(), &value, sizeof(T));
    if constexpr (std::endian::native == std::endian::big) std::reverse(bytes.begin(), bytes.end());
    out.insert(out.end(), bytes.begin(), bytes.end());
}

template <typename T>
bool read_le(std::span<const std::uint8_t> in, std::size_t off, T& value) noexcept {
    if (off + sizeof(T) > in.size()) return false;
    std::array<std::uint8_t, sizeof(T)> bytes{};
    std::copy_n(in.begin() + static_cast<std::ptrdiff_t>(off), sizeof(T), bytes.begin());
    if constexpr (std::endian::native == std::endian::big) std::reverse(bytes.begin(), bytes.end());
    std::memcpy(&value, bytes.data(), sizeof(T));
    return true;
}
}

std::uint32_t crc32_ieee(std::span<const std::uint8_t> bytes) noexcept {
    std::uint32_t crc = 0xFFFFFFFFu;
    for (auto byte : bytes) {
        crc ^= byte;
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1u) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}

std::vector<std::uint8_t> encode_feature_packet(const PacketHeader& h, const FeaturePayload& p) {
    std::vector<std::uint8_t> out;
    out.reserve(kHeaderBytes + kFeaturePayloadBytes + 4);
    out.insert(out.end(), kSync.begin(), kSync.end());
    out.push_back(h.version); out.push_back(h.type);
    append_le<std::uint16_t>(out, static_cast<std::uint16_t>(kFeaturePayloadBytes));
    append_le(out, h.sequence); append_le(out, h.pps_epoch); append_le(out, h.sub_us);
    append_le(out, p.ax_rms); append_le(out, p.ay_rms); append_le(out, p.az_rms);
    append_le(out, p.temperature_c); append_le(out, p.latitude); append_le(out, p.longitude); append_le(out, p.speed_mps);
    append_le(out, crc32_ieee(out));
    return out;
}

std::vector<std::uint8_t> encode_sensor_feature_packet(const PacketHeader& h, const SensorFeaturePayload& p) {
    std::vector<std::uint8_t> out;
    out.reserve(kHeaderBytes + kSensorFeaturePayloadBytes + 4);
    out.insert(out.end(), kSync.begin(), kSync.end());
    out.push_back(h.version); out.push_back(h.type);
    append_le<std::uint16_t>(out, static_cast<std::uint16_t>(kSensorFeaturePayloadBytes));
    append_le(out, h.sequence); append_le(out, h.pps_epoch); append_le(out, h.sub_us);
    out.push_back(p.sensor_id); out.push_back(p.flags); append_le(out, p.window_samples); append_le(out, p.sample_rate_hz);
    for (float v : p.axis_rms) append_le(out, v);
    append_le(out, p.rms); append_le(out, p.peak); append_le(out, p.kurtosis); append_le(out, p.crest_factor);
    for (float v : p.band_energy) append_le(out, v);
    append_le(out, p.temperature_c); append_le(out, p.humidity); append_le(out, p.latitude); append_le(out, p.longitude); append_le(out, p.speed_mps);
    append_le(out, crc32_ieee(out));
    return out;
}

bool decode_sensor_feature_payload(std::span<const std::uint8_t> in, SensorFeaturePayload& p) noexcept {
    if (in.size() != kSensorFeaturePayloadBytes) return false;
    std::size_t o = 0;
    p.sensor_id = in[o++];
    p.flags = in[o++];
    if (!read_le(in, o, p.window_samples)) return false;
    o += 2;
    if (!read_le(in, o, p.sample_rate_hz)) return false;
    o += 4;
    for (auto& v : p.axis_rms) { if (!read_le(in, o, v)) return false; o += 4; }
    if (!read_le(in, o, p.rms)) return false;
    o += 4;
    if (!read_le(in, o, p.peak)) return false;
    o += 4;
    if (!read_le(in, o, p.kurtosis)) return false;
    o += 4;
    if (!read_le(in, o, p.crest_factor)) return false;
    o += 4;
    for (auto& v : p.band_energy) { if (!read_le(in, o, v)) return false; o += 4; }
    if (!read_le(in, o, p.temperature_c)) return false;
    o += 4;
    if (!read_le(in, o, p.humidity)) return false;
    o += 4;
    if (!read_le(in, o, p.latitude)) return false;
    o += 4;
    if (!read_le(in, o, p.longitude)) return false;
    o += 4;
    return read_le(in, o, p.speed_mps);
}

std::optional<std::uint64_t> packet_timestamp_ns(const PacketHeader& h) noexcept {
    // UTC becomes trustworthy only after a GNSS/PPS fix. Reject sub-second fields
    // outside their protocol range and epochs outside a deliberately broad replay
    // window (2000-01-01 through 2100-01-01). A CRC proves transport integrity,
    // not that a producer supplied a semantically plausible clock value.
    constexpr std::uint32_t kMinEpoch = 946684800u;
    constexpr std::uint32_t kMaxEpoch = 4102444800u;
    if (h.pps_epoch < kMinEpoch || h.pps_epoch >= kMaxEpoch || h.sub_us >= 1'000'000u) {
        return std::nullopt;
    }
    return static_cast<std::uint64_t>(h.pps_epoch) * 1'000'000'000ull +
           static_cast<std::uint64_t>(h.sub_us) * 1000ull;
}

std::optional<std::uint64_t> sensor_feature_center_timestamp_ns(
    const PacketHeader& h, const SensorFeaturePayload& p) noexcept {
    const auto end_ns = packet_timestamp_ns(h);
    if (!end_ns || !(p.sample_rate_hz > 0.0f) || !std::isfinite(p.sample_rate_hz) || p.window_samples == 0) {
        return std::nullopt;
    }
    const auto half_window_ns = static_cast<std::uint64_t>(
        (0.5 * static_cast<double>(p.window_samples) / static_cast<double>(p.sample_rate_hz)) * 1e9);
    return *end_ns > half_window_ns ? *end_ns - half_window_ns : *end_ns;
}

bool validate_sensor_feature_payload(const SensorFeaturePayload& p) noexcept {
    const auto finite_nonnegative = [](float v) { return std::isfinite(v) && v >= 0.0f; };
    if (p.sensor_id >= 3 || p.window_samples == 0 || p.window_samples > 8192) return false;
    if (!std::isfinite(p.sample_rate_hz) || p.sample_rate_hz < 1.0f || p.sample_rate_hz > 100000.0f) return false;
    for (float v : p.axis_rms) if (!finite_nonnegative(v)) return false;
    if (!finite_nonnegative(p.rms) || !finite_nonnegative(p.peak) || !finite_nonnegative(p.kurtosis) || !finite_nonnegative(p.crest_factor)) return false;
    for (float v : p.band_energy) if (!finite_nonnegative(v)) return false;
    if ((p.flags & 0x01u) != 0u) {
        if (!std::isfinite(p.latitude) || p.latitude < -90.0f || p.latitude > 90.0f) return false;
        if (!std::isfinite(p.longitude) || p.longitude < -180.0f || p.longitude > 180.0f) return false;
        if (!std::isfinite(p.speed_mps) || p.speed_mps < 0.0f || p.speed_mps > 200.0f) return false;
    }
    if ((p.flags & 0x02u) != 0u) {
        if (!std::isfinite(p.temperature_c) || p.temperature_c < -80.0f || p.temperature_c > 125.0f) return false;
        if (!std::isfinite(p.humidity) || p.humidity < 0.0f || p.humidity > 1.0f) return false;
    }
    return true;
}

bool decode_feature_payload(std::span<const std::uint8_t> in, FeaturePayload& p) noexcept {
    if (in.size() != kFeaturePayloadBytes) return false;
    std::size_t o=0;
    if(!read_le(in,o,p.ax_rms)) return false;
    o+=4; if(!read_le(in,o,p.ay_rms)) return false;
    o+=4; if(!read_le(in,o,p.az_rms)) return false;
    o+=4; if(!read_le(in,o,p.temperature_c)) return false;
    o+=4; if(!read_le(in,o,p.latitude)) return false;
    o+=4; if(!read_le(in,o,p.longitude)) return false;
    o+=4;
    return read_le(in,o,p.speed_mps);
}

bool decode_feature_packet(std::span<const std::uint8_t> in, PacketHeader& h, FeaturePayload& p) noexcept {
    if (in.size() != kHeaderBytes + kFeaturePayloadBytes + 4 || in[0] != 'R' || in[1] != 'G') return false;
    h.version = in[2]; h.type = in[3];
    if (h.version != 1 || h.type != 1) return false;
    if (!read_le(in, 4, h.payload_len) || h.payload_len != kFeaturePayloadBytes) return false;
    if (!read_le(in, 6, h.sequence) || !read_le(in, 10, h.pps_epoch) || !read_le(in, 14, h.sub_us)) return false;
    std::uint32_t expected{}; if (!read_le(in, in.size() - 4, expected)) return false;
    if (crc32_ieee(in.first(in.size() - 4)) != expected) return false;
    return decode_feature_payload(in.subspan(kHeaderBytes, kFeaturePayloadBytes), p);
}

bool decode_sensor_feature_packet(std::span<const std::uint8_t> in, PacketHeader& h, SensorFeaturePayload& p) noexcept {
    if (in.size() != kHeaderBytes + kSensorFeaturePayloadBytes + 4 || in[0] != 'R' || in[1] != 'G') return false;
    h.version = in[2]; h.type = in[3];
    if (h.version != 2 || h.type != 2) return false;
    if (!read_le(in, 4, h.payload_len) || h.payload_len != kSensorFeaturePayloadBytes) return false;
    if (!read_le(in, 6, h.sequence) || !read_le(in, 10, h.pps_epoch) || !read_le(in, 14, h.sub_us)) return false;
    std::uint32_t expected{}; if (!read_le(in, in.size() - 4, expected)) return false;
    if (crc32_ieee(in.first(in.size() - 4)) != expected) return false;
    return decode_sensor_feature_payload(in.subspan(kHeaderBytes, kSensorFeaturePayloadBytes), p);
}

StreamDecoder::StreamDecoder(std::size_t reserve_bytes) { buffer_.reserve(reserve_bytes); }

std::vector<Packet> StreamDecoder::feed(std::span<const std::uint8_t> input) {
    buffer_.insert(buffer_.end(), input.begin(), input.end());
    std::vector<Packet> out;
    std::size_t cursor = 0;
    while (buffer_.size() - cursor >= kHeaderBytes) {
        auto it = std::search(buffer_.begin() + static_cast<std::ptrdiff_t>(cursor), buffer_.end(), kSync.begin(), kSync.end());
        if (it == buffer_.end()) { cursor = buffer_.size() > 1 ? buffer_.size() - 1 : 0; break; }
        cursor = static_cast<std::size_t>(std::distance(buffer_.begin(), it));
        if (buffer_.size() - cursor < kHeaderBytes) break;
        std::uint16_t payload_len{};
        read_le(std::span<const std::uint8_t>(buffer_), cursor + 4, payload_len);
        const std::size_t total = kHeaderBytes + payload_len + 4;
        if (payload_len > 4096) { ++cursor; continue; }
        if (buffer_.size() - cursor < total) {
            // A false sync in arbitrary serial noise can advertise a plausible but
            // incomplete length and otherwise head-of-line block a complete valid
            // frame already buffered behind it. Search later sync candidates and
            // skip forward only when one is structurally complete and CRC-valid.
            bool recovered_later = false;
            auto search_from = buffer_.begin() + static_cast<std::ptrdiff_t>(cursor + 1);
            while (search_from < buffer_.end()) {
                auto next = std::search(search_from, buffer_.end(), kSync.begin(), kSync.end());
                if (next == buffer_.end()) break;
                const auto next_cursor = static_cast<std::size_t>(std::distance(buffer_.begin(), next));
                if (buffer_.size() - next_cursor < kHeaderBytes) break;
                std::uint16_t next_len{};
                read_le(std::span<const std::uint8_t>(buffer_), next_cursor + 4, next_len);
                if (next_len <= 4096) {
                    const auto next_total = kHeaderBytes + static_cast<std::size_t>(next_len) + 4;
                    if (buffer_.size() - next_cursor >= next_total) {
                        const auto next_candidate = std::span<const std::uint8_t>(buffer_).subspan(next_cursor, next_total);
                        std::uint32_t next_expected{};
                        read_le(next_candidate, next_total - 4, next_expected);
                        if (crc32_ieee(next_candidate.first(next_total - 4)) == next_expected) {
                            cursor = next_cursor;
                            recovered_later = true;
                            break;
                        }
                    }
                }
                search_from = next + 1;
            }
            if (recovered_later) continue;
            break;
        }
        const auto candidate = std::span<const std::uint8_t>(buffer_).subspan(cursor, total);
        std::uint32_t expected{}; read_le(candidate, total - 4, expected);
        if (crc32_ieee(candidate.first(total - 4)) != expected) { ++cursor; continue; }
        Packet packet;
        packet.header.version = candidate[2]; packet.header.type = candidate[3]; packet.header.payload_len = payload_len;
        read_le(candidate,6,packet.header.sequence); read_le(candidate,10,packet.header.pps_epoch); read_le(candidate,14,packet.header.sub_us);
        packet.payload.assign(candidate.begin()+static_cast<std::ptrdiff_t>(kHeaderBytes), candidate.end()-4);
        out.push_back(std::move(packet));
        cursor += total;
    }
    if (cursor) buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(cursor));
    return out;
}
} // namespace railguard
