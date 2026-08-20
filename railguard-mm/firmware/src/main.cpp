#include <Arduino.h>
#include <stdint.h>
#include <string.h>

static const uint8_t SYNC0 = 'R';
static const uint8_t SYNC1 = 'G';
static uint32_t seq_no = 0;
static uint32_t pps_epoch = 0;
static uint32_t last_emit_ms = 0;

uint32_t crc32_ieee(const uint8_t* data, size_t len) {
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < len; ++i) {
    crc ^= data[i];
    for (int b = 0; b < 8; ++b)
      crc = (crc >> 1) ^ (0xEDB88320u & (-(int32_t)(crc & 1u)));
  }
  return ~crc;
}

#pragma pack(push, 1)
struct Header {
  uint8_t sync[2];
  uint8_t version;
  uint8_t type;
  uint16_t payload_len;
  uint32_t seq;
  uint32_t pps_epoch;
  uint32_t sub_us;
};

struct FeaturePayload {
  float ax_rms;
  float ay_rms;
  float az_rms;
  float temperature_c;
  float latitude;
  float longitude;
  float speed_mps;
};
#pragma pack(pop)

FeaturePayload read_vibration_window() {
  // Nucleo protocol bring-up signal. Hardware acquisition is implemented in firmware/stm32_hal/.
  const float t = millis() * 0.001f;
  FeaturePayload p{};
  p.ax_rms = 0.4f + 0.05f * sinf(t);
  p.ay_rms = 0.5f + 0.04f * sinf(0.7f * t);
  p.az_rms = 0.8f + 0.08f * sinf(1.2f * t);
  p.temperature_c = 23.0f;
  p.latitude = 40.235f;
  p.longitude = -77.885f;
  p.speed_mps = 6.0f;
  return p;
}

void send_packet(const FeaturePayload& payload) {
  Header h{};
  h.sync[0] = SYNC0;
  h.sync[1] = SYNC1;
  h.version = 1;
  h.type = 1;
  h.payload_len = sizeof(payload);
  h.seq = seq_no++;
  h.pps_epoch = pps_epoch;
  h.sub_us = micros() % 1000000u;

  uint8_t buffer[sizeof(Header) + sizeof(FeaturePayload)];
  memcpy(buffer, &h, sizeof(h));
  memcpy(buffer + sizeof(h), &payload, sizeof(payload));
  uint32_t crc = crc32_ieee(buffer, sizeof(buffer));

  Serial.write(buffer, sizeof(buffer));
  Serial.write(reinterpret_cast<uint8_t*>(&crc), sizeof(crc));
}

void setup() {
  Serial.begin(921600);
  while (!Serial && millis() < 3000) {}
  // This PlatformIO target exercises framing/transport only; the STM32Cube/HAL target owns physical acquisition.
}

void loop() {
  // PPS discipline is exercised in firmware/stm32_hal/pps_clock.c on the hardware target.
  if (millis() - last_emit_ms >= 100) {
    last_emit_ms += 100;
    auto payload = read_vibration_window();
    send_packet(payload);
  }
}
