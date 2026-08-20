#include "railguard/clock_alignment.hpp"
#include "railguard/inference.hpp"
#include "railguard/model_contract.hpp"
#include "railguard/protocol.hpp"
#include "railguard/resampler.hpp"
#include "railguard/serial_source.hpp"
#include "railguard/spatial_fusion.hpp"
#include "railguard/synchronizer.hpp"
#include "railguard/v4l2_camera.hpp"

#include <array>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
volatile std::sig_atomic_t running = 1;
void on_signal(int) { running = 0; }

std::string iso_utc(std::uint64_t utc_ns) {
    const auto epoch = static_cast<std::uint32_t>(utc_ns / 1'000'000'000ull);
    const auto us = static_cast<std::uint32_t>((utc_ns % 1'000'000'000ull) / 1000ull);
    if (epoch < 946684800u) return "1970-01-01T00:00:00.000000Z";
    std::time_t t = epoch;
    std::tm tm{};
    gmtime_r(&t, &tm);
    char b[64];
    std::snprintf(b, sizeof b, "%04d-%02d-%02dT%02d:%02d:%02d.%06uZ",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec, us);
    return b;
}

bool valid_device_id(const std::string& value) {
    if (value.empty() || value.size() > 64) return false;
    const auto valid_char = [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
    };
    if (!((value[0] >= 'a' && value[0] <= 'z') || (value[0] >= 'A' && value[0] <= 'Z') ||
          (value[0] >= '0' && value[0] <= '9'))) return false;
    for (unsigned char c : value) if (!valid_char(c)) return false;
    return true;
}

struct Args {
    std::string serial, camera, engine, model_version, device_id = "railguard-001";
    int baud = 921600;
    std::size_t seq_len = 32;
    double max_sync_ms = 50.0;
    double max_sensor_skew_ms = 50.0;
    double model_step_ms = 100.0;
};

Args args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
            return std::string(argv[++i]);
        };
        if (k == "--serial") a.serial = next();
        else if (k == "--camera") a.camera = next();
        else if (k == "--engine") a.engine = next();
        else if (k == "--model-version") a.model_version = next();
        else if (k == "--device-id") a.device_id = next();
        else if (k == "--baud") a.baud = std::stoi(next());
        else if (k == "--seq-len") a.seq_len = std::stoul(next());
        else if (k == "--max-sync-ms") a.max_sync_ms = std::stod(next());
        else if (k == "--max-sensor-skew-ms") a.max_sensor_skew_ms = std::stod(next());
        else if (k == "--model-step-ms") a.model_step_ms = std::stod(next());
        else if (k == "--help") {
            std::cout << "railguard_edge --serial /dev/ttyACM0 --camera /dev/video0 "
                         "[--engine model.engine --model-version fusion-<id>] [--seq-len 32] [--model-step-ms 100] "
                         "[--max-sync-ms 50] [--max-sensor-skew-ms 50]\n";
            std::exit(0);
        } else throw std::runtime_error("unknown argument: " + k);
    }
    if (a.seq_len == 0) throw std::runtime_error("--seq-len must be non-zero");
    if (!(a.model_step_ms > 0.0)) throw std::runtime_error("--model-step-ms must be positive");
    if (!valid_device_id(a.device_id)) throw std::runtime_error("--device-id must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}");
    if (!a.engine.empty() && a.model_version.empty()) throw std::runtime_error("--engine requires --model-version from the deployment manifest");
    if (a.model_version.size() > 128) throw std::runtime_error("--model-version must be <=128 characters");
    return a;
}

void emit_sensors(const std::array<railguard::SensorFeaturePayload, 3>& s) {
    std::cout << '[';
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << "{\"sensor_id\":" << static_cast<int>(s[i].sensor_id)
                  << ",\"rms_ms2\":" << s[i].rms
                  << ",\"peak_ms2\":" << s[i].peak
                  << ",\"kurtosis\":" << s[i].kurtosis
                  << ",\"crest_factor\":" << s[i].crest_factor
                  << ",\"band_energy\":[" << s[i].band_energy[0] << ',' << s[i].band_energy[1]
                  << ',' << s[i].band_energy[2] << ',' << s[i].band_energy[3] << "]}";
    }
    std::cout << ']';
}

std::array<float, railguard::kSensorFeatureDim> model_sensor_vector(const railguard::SpatialAggregate& s, const railguard::CameraFrame& frame) {
    return {
        s.rms, s.peak, s.kurtosis, s.crest_factor,
        frame.motion, frame.contrast, s.speed_mps,
        (s.context_flags & 0x02u) ? s.temperature_c : 0.0f,
        (s.context_flags & 0x02u) ? s.humidity : 0.0f,
    };
}
}

int main(int argc, char** argv) {
    try {
        const auto a = args(argc, argv);

        if (a.serial.empty()) {
            auto reference = railguard::make_reference_engine();
            constexpr std::size_t T = 8, H = 96, W = 96;
            std::vector<float> frames(T * 3 * H * W, .5f), sensors(T * 9, 0.f);
            sensors[sensors.size() - 9] = .56f;
            const auto p = reference->infer(frames, sensors, T, H, W);
            std::cout << "# railguard native backend=" << reference->backend() << " (self-test only)\n";
            std::cout << "{\"mode\":\"self-test\",\"anomaly_probability\":" << p.anomaly_probability << "}\n";
            return 0;
        }

        std::unique_ptr<railguard::InferenceEngine> engine;
        if (!a.engine.empty()) {
            if (a.camera.empty()) throw std::runtime_error("--engine requires --camera for the deployed multimodal model");
            engine = railguard::make_inference_engine(a.engine);
        }
        std::cout << "# railguard native backend=" << (engine ? engine->backend() : "disabled") << "\n";

        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);

        railguard::PosixSerialSource serial(a.serial, a.baud);
        railguard::StreamDecoder decoder;
        std::array<std::uint8_t, 8192> rx{};
        railguard::FrameSynchronizer frame_sync(160, a.max_sync_ms);
        railguard::UtcMonotonicAligner clock_aligner(128, 8);
        railguard::SpatialVibrationAggregator spatial(a.max_sensor_skew_ms);
        railguard::FixedRateSpatialResampler resampler(a.model_step_ms);

        std::jthread camera_thread;
        if (!a.camera.empty()) {
            camera_thread = std::jthread([&](std::stop_token stop) {
                try {
                    railguard::V4L2Camera cam(a.camera);
                    while (!stop.stop_requested() && running) {
                        railguard::CameraFrame f;
                        if (cam.capture(f)) frame_sync.push(std::move(f));
                    }
                } catch (const std::exception& e) {
                    std::cerr << "camera thread: " << e.what() << "\n";
                }
            });
        }

        std::deque<std::vector<float>> frame_seq;
        std::deque<std::array<float, railguard::kSensorFeatureDim>> sensor_seq;
        std::uint32_t last_seq = 0, packet_loss = 0;
        bool have_seq = false;

        while (running) {
            const auto n = serial.read_some(rx);
            if (!n) continue;
            for (auto& pkt : decoder.feed(std::span(rx.data(), n))) {
                // CRC-valid bytes are not yet trustworthy timing observations. Only
                // production v2 packets with semantically valid payloads and plausible
                // PPS timestamps are allowed to influence packet-loss accounting or the
                // UTC<->monotonic camera clock alignment.
                if (pkt.header.type != 2 || pkt.header.version != 2) continue;
                railguard::SensorFeaturePayload sf;
                if (!railguard::decode_sensor_feature_payload(pkt.payload, sf) ||
                    !railguard::validate_sensor_feature_payload(sf)) continue;
                const auto raw_utc_ns = railguard::packet_timestamp_ns(pkt.header);
                const auto center_utc_ns = railguard::sensor_feature_center_timestamp_ns(pkt.header, sf);
                if (!raw_utc_ns || !center_utc_ns) continue;

                const auto receive_mono_ns = railguard::steady_now_ns();
                clock_aligner.observe(*raw_utc_ns, receive_mono_ns);

                if (have_seq) {
                    const std::uint32_t delta = pkt.header.sequence - last_seq; // wrap-safe unsigned arithmetic
                    if (delta > 1u && delta < 0x80000000u) packet_loss += delta - 1u;
                }
                last_seq = pkt.header.sequence;
                have_seq = true;

                const auto fused = spatial.update(sf, *center_utc_ns);
                if (!fused) continue;

                for (const auto& sample : resampler.push(*fused)) {
                    railguard::CameraFrame neutral;
                    neutral.rgb_chw.assign(3 * 96 * 96, .5f);
                    const railguard::CameraFrame* frame = &neutral;
                    double sync_error_ms = 0.0;
                    bool camera_matched = false;
                    std::optional<railguard::FrameMatch> match;

                    if (!a.camera.empty()) {
                        const auto sensor_mono = clock_aligner.monotonic_from_utc(sample.utc_ns);
                        if (sensor_mono) {
                            match = frame_sync.nearest(*sensor_mono);
                            if (match) {
                                frame = match->frame.get();
                                sync_error_ms = match->delta_ms;
                                camera_matched = true;
                            }
                        }
                    }

                    const bool context_complete = (sample.context_flags & 0x03u) == 0x03u;
                    std::optional<railguard::Prediction> pred;
                    if (camera_matched && context_complete && engine) {
                        frame_seq.push_back(frame->rgb_chw);
                        sensor_seq.push_back(model_sensor_vector(sample, *frame));
                        while (frame_seq.size() > a.seq_len) frame_seq.pop_front();
                        while (sensor_seq.size() > a.seq_len) sensor_seq.pop_front();
                        if (frame_seq.size() == a.seq_len) {
                            std::vector<float> frames;
                            std::vector<float> sensors;
                            frames.reserve(a.seq_len * 3 * 96 * 96);
                            sensors.reserve(a.seq_len * 9);
                            for (const auto& v : frame_seq) frames.insert(frames.end(), v.begin(), v.end());
                            for (const auto& v : sensor_seq) sensors.insert(sensors.end(), v.begin(), v.end());
                            auto candidate = engine->infer(frames, sensors, a.seq_len, 96, 96);
                            if (!railguard::validate_prediction(candidate)) {
                                throw std::runtime_error("inference backend produced invalid physical outputs");
                            }
                            pred = candidate;
                        }
                    } else if (!camera_matched || !context_complete) {
                        // Do not hide missing modalities/context by concatenating samples on either
                        // side of a gap into an apparently contiguous Transformer sequence. The
                        // training data has complete operating context, so live inference fails
                        // closed rather than injecting physical zeroes as out-of-distribution inputs.
                        frame_seq.clear();
                        sensor_seq.clear();
                    }

                    const auto clock = clock_aligner.status();
                    std::cout << "{\"schema_version\":1,\"device_id\":\"" << a.device_id << "\",\"ts\":\"" << iso_utc(sample.utc_ns)
                              << "\",\"seq\":" << pkt.header.sequence
                              << ",\"sample_period_ms\":" << a.model_step_ms
<< ",\"gps\":{\"lat\":";
                    if (sample.context_flags & 0x01u) std::cout << sample.latitude; else std::cout << "null";
                    std::cout << ",\"lon\":";
                    if (sample.context_flags & 0x01u) std::cout << sample.longitude; else std::cout << "null";
                    std::cout << ",\"speed_mps\":";
                    if (sample.context_flags & 0x01u) std::cout << sample.speed_mps; else std::cout << "null";
                    std::cout << "}" << ",\"environment\":{\"temperature_c\":";
                    if (sample.context_flags & 0x02u) std::cout << sample.temperature_c; else std::cout << "null";
                    std::cout << ",\"humidity\":";
                    if (sample.context_flags & 0x02u) std::cout << sample.humidity; else std::cout << "null";
                    std::cout << "}"
                                                            << ",\"vibration\":{\"rms_ms2\":" << sample.rms << ",\"peak_ms2\":" << sample.peak
                              << ",\"kurtosis\":" << sample.kurtosis << ",\"crest_factor\":" << sample.crest_factor
                              << ",\"band_energy\":[" << sample.band_energy[0] << ',' << sample.band_energy[1] << ','
                              << sample.band_energy[2] << ',' << sample.band_energy[3] << "],\"sensors\":";
                    emit_sensors(sample.sensors);
                    std::cout << "}"
                              << ",\"vision\":{\"motion_score\":" << frame->motion << ",\"contrast\":" << frame->contrast
                              << ",\"sharpness\":" << frame->sharpness << ",\"frame_ref\":null}"
                              << ",\"health\":{\"packet_loss\":" << packet_loss << ",\"spool_depth\":0"
                              << ",\"camera_matched\":" << (camera_matched ? "true" : "false")
                              << ",\"sync_error_ms\":";
                    if (camera_matched) std::cout << sync_error_ms; else std::cout << "null";
                    std::cout << ",\"sensor_skew_ms\":" << sample.sensor_skew_ms
                              << ",\"clock_alignment_locked\":" << (clock.locked ? "true" : "false")
                              << ",\"clock_jitter_ms\":" << clock.jitter_ms
                              << ",\"clock_samples\":" << clock.samples
                              << ",\"context_flags\":" << static_cast<int>(sample.context_flags) << "}";
                    if (pred) {
                        std::cout << ",\"prediction\":{\"model_version\":\"" << a.model_version
                                  << "\",\"horizons\":[1,5,10],\"step_ms\":" << a.model_step_ms
                                  << ",\"vibration_rms\":[" << pred->vibration[0] << ',' << pred->vibration[1] << ',' << pred->vibration[2]
                                  << "],\"vision_motion\":[" << pred->vision[0] << ',' << pred->vision[1] << ',' << pred->vision[2]
                                  << "],\"anomaly_probability\":" << pred->anomaly_probability << "}";
                    }
                    std::cout << "}\n";
                }
            }
        }

        if (camera_thread.joinable()) camera_thread.request_stop();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "railguard_edge: " << e.what() << "\n";
        return 2;
    }
}
