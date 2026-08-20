#include "railguard/resampler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace railguard {
namespace {
float lerp(float a, float b, double x) { return static_cast<float>(a + (b - a) * x); }

SensorFeaturePayload interpolate_sensor(const SensorFeaturePayload& a, const SensorFeaturePayload& b, double x) {
    SensorFeaturePayload o{};
    o.sensor_id = a.sensor_id;
    o.flags = static_cast<std::uint8_t>(a.flags & b.flags);
    o.window_samples = b.window_samples;
    o.sample_rate_hz = b.sample_rate_hz;
    for (std::size_t i=0;i<3;++i) o.axis_rms[i] = lerp(a.axis_rms[i], b.axis_rms[i], x);
    o.rms = lerp(a.rms,b.rms,x); o.peak = lerp(a.peak,b.peak,x);
    o.kurtosis = lerp(a.kurtosis,b.kurtosis,x); o.crest_factor = lerp(a.crest_factor,b.crest_factor,x);
    for (std::size_t i=0;i<4;++i) o.band_energy[i] = lerp(a.band_energy[i],b.band_energy[i],x);
    if (o.flags & 0x02u) { o.temperature_c=lerp(a.temperature_c,b.temperature_c,x); o.humidity=lerp(a.humidity,b.humidity,x); }
    if (o.flags & 0x01u) { o.latitude=lerp(a.latitude,b.latitude,x); o.longitude=lerp(a.longitude,b.longitude,x); o.speed_mps=lerp(a.speed_mps,b.speed_mps,x); }
    return o;
}
}

FixedRateSpatialResampler::FixedRateSpatialResampler(double period_ms, double max_gap_periods) {
    if (!(period_ms > 0.0) || !(max_gap_periods >= 1.0)) throw std::invalid_argument("invalid resampler configuration");
    period_ns_ = static_cast<std::uint64_t>(std::llround(period_ms * 1e6));
    max_gap_ns_ = static_cast<std::uint64_t>(std::llround(period_ms * max_gap_periods * 1e6));
}

std::uint64_t FixedRateSpatialResampler::ceil_to_grid(std::uint64_t t) const noexcept {
    const auto rem = t % period_ns_;
    return rem == 0 ? t : t + (period_ns_ - rem);
}

SpatialAggregate FixedRateSpatialResampler::interpolate(const SpatialAggregate& a, const SpatialAggregate& b, std::uint64_t target_ns) {
    const auto span = b.utc_ns - a.utc_ns;
    const double x = span == 0 ? 0.0 : std::clamp(
        static_cast<double>(target_ns - a.utc_ns) / static_cast<double>(span), 0.0, 1.0);
    SpatialAggregate o{};
    o.utc_ns = target_ns;
    o.sensor_skew_ms = std::max(a.sensor_skew_ms, b.sensor_skew_ms); // conservative quality metric
    o.context_flags = static_cast<std::uint8_t>(a.context_flags & b.context_flags);
    for(std::size_t i=0;i<3;++i) o.sensors[i] = interpolate_sensor(a.sensors[i], b.sensors[i], x);
    o.rms=lerp(a.rms,b.rms,x); o.peak=lerp(a.peak,b.peak,x); o.kurtosis=lerp(a.kurtosis,b.kurtosis,x); o.crest_factor=lerp(a.crest_factor,b.crest_factor,x);
    for(std::size_t i=0;i<4;++i) o.band_energy[i]=lerp(a.band_energy[i],b.band_energy[i],x);
    if(o.context_flags & 0x02u){o.temperature_c=lerp(a.temperature_c,b.temperature_c,x);o.humidity=lerp(a.humidity,b.humidity,x);}
    if(o.context_flags & 0x01u){o.latitude=lerp(a.latitude,b.latitude,x);o.longitude=lerp(a.longitude,b.longitude,x);o.speed_mps=lerp(a.speed_mps,b.speed_mps,x);}
    return o;
}

std::vector<SpatialAggregate> FixedRateSpatialResampler::push(const SpatialAggregate& sample) {
    std::vector<SpatialAggregate> out;
    if(sample.utc_ns == 0) return out;
    if(!previous_){
        previous_=sample;
        next_target_ns_=ceil_to_grid(sample.utc_ns);
        if(next_target_ns_==sample.utc_ns){out.push_back(sample);next_target_ns_+=period_ns_;}
        return out;
    }
    if(sample.utc_ns <= previous_->utc_ns || sample.utc_ns - previous_->utc_ns > max_gap_ns_){
        previous_=sample;next_target_ns_=ceil_to_grid(sample.utc_ns);return out;
    }
    while(next_target_ns_ <= sample.utc_ns){
        if(next_target_ns_ >= previous_->utc_ns) out.push_back(interpolate(*previous_, sample, next_target_ns_));
        next_target_ns_ += period_ns_;
    }
    previous_=sample;
    return out;
}

} // namespace railguard
