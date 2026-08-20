#include "railguard/dsp.hpp"
#include "railguard/protocol.hpp"
#include "railguard/spsc_ring.hpp"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numbers>
#include <vector>
using clock_type=std::chrono::steady_clock;
template<class F> double bench_ms(F&& f){auto a=clock_type::now();f();auto b=clock_type::now();return std::chrono::duration<double,std::milli>(b-a).count();}
int main(int argc,char** argv){
    const std::size_t n=argc>1?std::stoull(argv[1]):200000;
    railguard::PacketHeader h{1,1,28,0,1700000000,1000}; railguard::FeaturePayload p{.4f,.5f,.8f,23.f,40.2f,-77.8f,6.f};
    auto bytes=railguard::encode_feature_packet(h,p); std::size_t ok=0;
    const double parse_ms=bench_ms([&]{for(std::size_t i=0;i<n;++i){railguard::PacketHeader dh;railguard::FeaturePayload dp;ok+=railguard::decode_feature_packet(bytes,dh,dp);}});
    railguard::PacketHeader h2{2,2,72,0,1700000000,1000};railguard::SensorFeaturePayload sp{};sp.sensor_id=1;sp.flags=1;sp.window_samples=512;sp.sample_rate_hz=26667;sp.axis_rms={.4f,.5f,.8f};sp.rms=.65f;sp.peak=1.8f;sp.kurtosis=3.2f;sp.crest_factor=2.77f;sp.band_energy={.1f,.2f,.3f,.4f};sp.humidity=.55f;sp.latitude=40.2f;sp.longitude=-77.8f;sp.speed_mps=6.f;auto sensor_bytes=railguard::encode_sensor_feature_packet(h2,sp);std::size_t sensor_ok=0;
    const std::size_t warmup=std::min<std::size_t>(n,10000);for(std::size_t i=0;i<warmup;++i){railguard::PacketHeader wh;railguard::SensorFeaturePayload wp;(void)railguard::decode_sensor_feature_packet(sensor_bytes,wh,wp);(void)railguard::validate_sensor_feature_payload(wp);(void)railguard::sensor_feature_center_timestamp_ns(wh,wp);}
    const double sensor_parse_ms=bench_ms([&]{for(std::size_t i=0;i<n;++i){railguard::PacketHeader dh;railguard::SensorFeaturePayload dp;sensor_ok+=railguard::decode_sensor_feature_packet(sensor_bytes,dh,dp);}});
    std::size_t sensor_accept_ok=0;
    const double sensor_accept_ms=bench_ms([&]{for(std::size_t i=0;i<n;++i){railguard::PacketHeader dh;railguard::SensorFeaturePayload dp;if(railguard::decode_sensor_feature_packet(sensor_bytes,dh,dp)&&railguard::validate_sensor_feature_payload(dp)&&railguard::sensor_feature_center_timestamp_ns(dh,dp).has_value())++sensor_accept_ok;}});
    std::vector<float> sig(512);for(size_t i=0;i<sig.size();++i)sig[i]=.8f*std::sin(2*std::numbers::pi*75.0*i/2000.0);volatile float sink=0;
    const double dsp_ms=bench_ms([&]{for(std::size_t i=0;i<n/20+1;++i)sink+=railguard::vibration_features(sig).rms;});
    railguard::SpscRing<std::uint64_t,1024> ring; std::size_t ring_ok=0;
    const double ring_ms=bench_ms([&]{for(std::size_t i=0;i<n;++i){if(ring.try_push(i)){auto v=ring.try_pop();ring_ok+=v.has_value();}}});
    std::cout<<"{\"language\":\"cpp20\",\"iterations\":"<<n<<",\"warmup_iterations\":"<<warmup
             <<",\"packet_decode_ns\":"<<(parse_ms*1e6/n)<<",\"packet_decode_mpps\":"<<(n/(parse_ms/1000.0)/1e6)
             <<",\"sensor_packet_decode_ns\":"<<(sensor_parse_ms*1e6/n)<<",\"sensor_packet_decode_mpps\":"<<(n/(sensor_parse_ms/1000.0)/1e6)
             <<",\"sensor_packet_accept_ns\":"<<(sensor_accept_ms*1e6/n)<<",\"sensor_packet_accept_mpps\":"<<(n/(sensor_accept_ms/1000.0)/1e6)
             <<",\"dsp_window_us\":"<<(dsp_ms*1000.0/(n/20+1))<<",\"ring_roundtrip_ns\":"<<(ring_ms*1e6/n)
             <<",\"valid_packets\":"<<ok<<",\"valid_sensor_packets\":"<<sensor_ok<<",\"accepted_sensor_packets\":"<<sensor_accept_ok<<",\"ring_ops\":"<<ring_ok<<",\"sink\":"<<sink<<"}\n";
}
