#include "railguard/spatial_fusion.hpp"
#include <algorithm>
#include <cmath>

namespace railguard {
SpatialVibrationAggregator::SpatialVibrationAggregator(double max_sensor_skew_ms)
    : max_sensor_skew_ms_(std::max(0.0, max_sensor_skew_ms)) {}

std::optional<SpatialAggregate> SpatialVibrationAggregator::update(const SensorFeaturePayload& p, std::uint64_t utc_ns){
    if(utc_ns==0 || !validate_sensor_feature_payload(p)) return std::nullopt;
    const auto bit=static_cast<std::uint8_t>(1u<<p.sensor_id);
    latest_[p.sensor_id]=p; timestamps_[p.sensor_id]=utc_ns; valid_mask_|=bit; fresh_mask_|=bit;
    if(valid_mask_!=0x07u || fresh_mask_!=0x07u) return std::nullopt;
    const auto [mn,mx]=std::minmax_element(timestamps_.begin(),timestamps_.end());
    const double skew=static_cast<double>(*mx-*mn)/1e6;
    // Always consume one fresh sample from every location. If the skew is too large,
    // refuse the fused window instead of silently pretending it is synchronous.
    fresh_mask_=0;
    if(skew>max_sensor_skew_ms_) return std::nullopt;
    SpatialAggregate a{};a.sensors=latest_;a.utc_ns=*mx;a.sensor_skew_ms=skew;
    for(const auto&s:latest_){
        a.rms+=s.rms/3.0f;
        a.kurtosis+=s.kurtosis/3.0f;
        for(std::size_t i=0;i<4;i++)a.band_energy[i]+=s.band_energy[i]/3.0f;
        a.peak=std::max(a.peak,s.peak);
        a.context_flags|=static_cast<std::uint8_t>(s.flags & ~0x03u);
    }
    a.crest_factor=a.rms>1e-9f?a.peak/a.rms:0.0f;

    // Context validity is per field, not per fused window.  Never OR a valid bit from
    // one packet and then copy contextual values from a different invalid packet.
    // Select the newest valid source independently for GNSS and environment.
    std::optional<std::size_t> newest_gnss;
    std::optional<std::size_t> newest_env;
    for(std::size_t i=0;i<latest_.size();++i){
        if((latest_[i].flags & 0x01u) && (!newest_gnss || timestamps_[i] > timestamps_[*newest_gnss])) newest_gnss=i;
        if((latest_[i].flags & 0x02u) && (!newest_env || timestamps_[i] > timestamps_[*newest_env])) newest_env=i;
    }
    if(newest_gnss){
        const auto&i=latest_[*newest_gnss];
        a.latitude=i.latitude;a.longitude=i.longitude;a.speed_mps=i.speed_mps;
        a.context_flags|=0x01u;
    }
    if(newest_env){
        const auto&i=latest_[*newest_env];
        a.temperature_c=i.temperature_c;a.humidity=i.humidity;
        a.context_flags|=0x02u;
    }
    return a;
}
} // namespace railguard
