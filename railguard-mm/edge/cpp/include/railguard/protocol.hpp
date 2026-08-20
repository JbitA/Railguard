#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <optional>
#include <vector>

namespace railguard {

constexpr std::array<std::uint8_t, 2> kSync{'R', 'G'};
constexpr std::size_t kHeaderBytes = 18;
constexpr std::size_t kFeaturePayloadBytes = 28;
constexpr std::size_t kSensorFeaturePayloadBytes = 72;

struct PacketHeader {
    std::uint8_t version{};
    std::uint8_t type{};
    std::uint16_t payload_len{};
    std::uint32_t sequence{};
    std::uint32_t pps_epoch{};
    std::uint32_t sub_us{};
};

struct FeaturePayload {
    float ax_rms{};
    float ay_rms{};
    float az_rms{};
    float temperature_c{};
    float latitude{};
    float longitude{};
    float speed_mps{};
};


struct SensorFeaturePayload {
    std::uint8_t sensor_id{};
    std::uint8_t flags{}; // bit0: GNSS fix valid, bit1: environment valid
    std::uint16_t window_samples{};
    float sample_rate_hz{};
    std::array<float,3> axis_rms{};
    float rms{};
    float peak{};
    float kurtosis{};
    float crest_factor{};
    std::array<float,4> band_energy{};
    float temperature_c{};
    float humidity{};
    float latitude{};
    float longitude{};
    float speed_mps{};
};

struct Packet {
    PacketHeader header;
    std::vector<std::uint8_t> payload;
};

std::uint32_t crc32_ieee(std::span<const std::uint8_t> bytes) noexcept;
std::vector<std::uint8_t> encode_feature_packet(const PacketHeader&, const FeaturePayload&);
std::vector<std::uint8_t> encode_sensor_feature_packet(const PacketHeader&, const SensorFeaturePayload&);
bool decode_feature_packet(std::span<const std::uint8_t>, PacketHeader&, FeaturePayload&) noexcept;
bool decode_sensor_feature_packet(std::span<const std::uint8_t>, PacketHeader&, SensorFeaturePayload&) noexcept;
bool decode_feature_payload(std::span<const std::uint8_t>, FeaturePayload&) noexcept;
bool decode_sensor_feature_payload(std::span<const std::uint8_t>, SensorFeaturePayload&) noexcept;
bool validate_sensor_feature_payload(const SensorFeaturePayload&) noexcept;
std::optional<std::uint64_t> packet_timestamp_ns(const PacketHeader&) noexcept;
std::optional<std::uint64_t> sensor_feature_center_timestamp_ns(const PacketHeader&, const SensorFeaturePayload&) noexcept;

class StreamDecoder {
public:
    explicit StreamDecoder(std::size_t reserve_bytes = 64 * 1024);
    std::vector<Packet> feed(std::span<const std::uint8_t> input);
    [[nodiscard]] std::size_t buffered_bytes() const noexcept { return buffer_.size(); }
private:
    std::vector<std::uint8_t> buffer_;
};

} // namespace railguard
