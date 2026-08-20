#include "railguard/spatial_fusion.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

static railguard::SensorFeaturePayload sample(unsigned id,float rms,float peak){
    railguard::SensorFeaturePayload p{};p.sensor_id=static_cast<std::uint8_t>(id);p.rms=rms;p.peak=peak;p.kurtosis=3.0f+id;p.crest_factor=peak/rms;p.flags=1;p.latitude=40.0f;p.longitude=-77.0f;p.speed_mps=6.0f;for(auto&b:p.band_energy)b=rms;return p;
}
int main(){
    railguard::SpatialVibrationAggregator agg(25.0);
    assert(!agg.update(sample(0,1,2),1000000000ull));
    assert(!agg.update(sample(1,2,4),1005000000ull));
    auto a=agg.update(sample(2,3,7),1010000000ull);assert(a);
    assert(std::abs(a->rms-2.0f)<1e-6f);assert(std::abs(a->peak-7.0f)<1e-6f);assert(std::abs(a->sensor_skew_ms-10.0)<1e-6);
    // One fresh observation from every location is required for the next fused step.
    assert(!agg.update(sample(0,2,3),1020000000ull));
    assert(!agg.update(sample(1,2,3),1021000000ull));
    auto b=agg.update(sample(2,2,3),1022000000ull);assert(b);
    railguard::SpatialVibrationAggregator strict(1.0);
    assert(!strict.update(sample(0,1,1),1000000000ull));assert(!strict.update(sample(1,1,1),1002000000ull));assert(!strict.update(sample(2,1,1),1004000000ull));
    // Validity bits and values must come from the same packet.  A newer invalid
    // context packet must not overwrite an older valid source while leaving the
    // valid bit set.
    railguard::SpatialVibrationAggregator ctx(25.0);
    auto p0=sample(0,1,1);p0.flags=0x03u;p0.temperature_c=21.0f;p0.humidity=.4f;p0.latitude=40.1f;
    auto p1=sample(1,1,1);p1.flags=0x00u;p1.temperature_c=99.0f;p1.humidity=.99f;p1.latitude=0.0f;
    auto p2=sample(2,1,1);p2.flags=0x00u;p2.temperature_c=88.0f;p2.humidity=.88f;p2.latitude=0.0f;
    assert(!ctx.update(p0,2000000000ull));assert(!ctx.update(p1,2005000000ull));auto c=ctx.update(p2,2010000000ull);assert(c);
    assert((c->context_flags & 0x03u)==0x03u);assert(std::abs(c->temperature_c-21.0f)<1e-6f);assert(std::abs(c->latitude-40.1f)<1e-5f);
    std::cout<<"spatial fusion tests: PASS\n";
}
