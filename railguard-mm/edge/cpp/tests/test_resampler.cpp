#include "railguard/resampler.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

static railguard::SpatialAggregate s(std::uint64_t ns,float rms){
    railguard::SpatialAggregate a{};a.utc_ns=ns;a.rms=rms;a.peak=2*rms;a.kurtosis=3.0f;a.crest_factor=2.0f;a.context_flags=0x03u;a.temperature_c=20.0f;a.humidity=.5f;a.latitude=40.0f;a.longitude=-77.0f;a.speed_mps=5.0f;
    for(unsigned i=0;i<3;++i){a.sensors[i].sensor_id=i;a.sensors[i].flags=0x03u;a.sensors[i].window_samples=512;a.sensors[i].sample_rate_hz=26667.0f;a.sensors[i].rms=rms+i;}
    return a;
}
int main(){
    railguard::FixedRateSpatialResampler r(100.0);
    std::vector<railguard::SpatialAggregate> out;
    for(int i=0;i<13;++i){auto v=r.push(s(1'000'000'000ull + static_cast<std::uint64_t>(i)*20'000'000ull, static_cast<float>(i)));out.insert(out.end(),v.begin(),v.end());}
    assert(out.size()==3); // 1.0, 1.1, 1.2 seconds
    assert(out[0].utc_ns==1'000'000'000ull);assert(out[1].utc_ns==1'100'000'000ull);assert(out[2].utc_ns==1'200'000'000ull);
    assert(std::abs(out[1].rms-5.0f)<1e-6f);
    assert(std::abs(out[2].rms-10.0f)<1e-6f);
    std::cout<<"resampler tests: PASS\n";
}
