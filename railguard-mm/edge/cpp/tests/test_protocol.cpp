#include "railguard/protocol.hpp"
#include <cassert>
#include <cmath>
#include <limits>
#include <iomanip>
#include <sstream>
#include <random>
#include <string>
static std::string hex(const std::vector<std::uint8_t>& v){std::ostringstream o;for(auto b:v)o<<std::hex<<std::setw(2)<<std::setfill('0')<<(int)b;return o.str();}
int main(){
 railguard::PacketHeader h{1,1,28,42,1700000000u,123456u};railguard::FeaturePayload p{.4f,.5f,.8f,23.f,40.2f,-77.8f,6.f};
 auto bytes=railguard::encode_feature_packet(h,p);
 assert(hex(bytes)=="524701011c002a00000000f1536540e20100cdcccc3e0000003fcdcc4c3f0000b841cdcc20429a999bc20000c04037e4af42");
 railguard::PacketHeader dh{};railguard::FeaturePayload dp{};assert(railguard::decode_feature_packet(bytes,dh,dp));assert(dh.sequence==42);assert(std::abs(dp.longitude+77.8f)<1e-4f);
 auto corrupt=bytes;corrupt[20]^=0x01;assert(!railguard::decode_feature_packet(corrupt,dh,dp));
 railguard::StreamDecoder d;std::vector<std::uint8_t> noisy{0,1,2};noisy.insert(noisy.end(),bytes.begin(),bytes.end());auto packets=d.feed(noisy);assert(packets.size()==1&&packets[0].header.sequence==42);
 railguard::PacketHeader h2{2,2,72,43,1700000000u,124000u};railguard::SensorFeaturePayload s{};s.sensor_id=2;s.flags=1;s.window_samples=512;s.sample_rate_hz=26667.0f;s.axis_rms={1,2,3};s.rms=2.1f;s.peak=6.2f;s.kurtosis=3.4f;s.crest_factor=2.95f;s.band_energy={.1f,.2f,.3f,.4f};s.temperature_c=0.0f;s.humidity=.55f;s.latitude=40.2f;s.longitude=-77.8f;s.speed_mps=6.0f;
 auto v2=railguard::encode_sensor_feature_packet(h2,s);assert(v2.size()==94);
 // Frozen 94-byte vector emitted by firmware/stm32_hal/Core/Src/railguard_packet.c: proves C17/C++20 wire compatibility.
 assert(hex(v2)=="5247020248002b00000000f1536560e40100020100020056d0460000803f0000004000004040666606406666c6409a995940cdcc3c40cdcccc3dcdcc4c3e9a99993ecdcccc3e00000000cdcc0c3fcdcc20429a999bc20000c0406baafed1");
 railguard::StreamDecoder d2;auto ps=d2.feed(v2);assert(ps.size()==1&&ps[0].header.type==2);railguard::SensorFeaturePayload q{};assert(railguard::decode_sensor_feature_payload(ps[0].payload,q));assert(q.sensor_id==2&&q.window_samples==512&&std::abs(q.band_energy[3]-.4f)<1e-6f&&std::abs(q.humidity-.55f)<1e-6f);
 railguard::StreamDecoder d3;std::vector<std::uint8_t> wedged{'R','G',9,9,0xa0,0x0f,0,0,0,0,0,0,0,0,0,0,0,0};wedged.insert(wedged.end(),v2.begin(),v2.end());auto recovered=d3.feed(wedged);assert(recovered.size()==1&&recovered[0].header.sequence==43);
 assert(railguard::validate_sensor_feature_payload(q));

 // A valid CRC does not make an impossible GNSS/PPS timestamp trustworthy.
 auto valid_ts=railguard::packet_timestamp_ns(h2);assert(valid_ts.has_value());
 auto bad_time=h2;bad_time.sub_us=1'000'000u;assert(!railguard::packet_timestamp_ns(bad_time));
 bad_time=h2;bad_time.pps_epoch=1u;bad_time.sub_us=0u;assert(!railguard::packet_timestamp_ns(bad_time));
 auto center=railguard::sensor_feature_center_timestamp_ns(h2,s);assert(center.has_value()&&*center<*valid_ts);
 q.rms=std::numeric_limits<float>::quiet_NaN();assert(!railguard::validate_sensor_feature_payload(q));
 q=s;q.flags=0x03u;q.latitude=95.0f;assert(!railguard::validate_sensor_feature_payload(q));
 q=s;q.flags=0x03u;q.humidity=1.2f;assert(!railguard::validate_sensor_feature_payload(q));

 // Contract-version mismatch must not be accepted by typed decoders even if CRC is valid.
 auto wrong_version=v2;wrong_version[2]=3;
 auto crc=railguard::crc32_ieee(std::span<const std::uint8_t>(wrong_version).first(wrong_version.size()-4));
 for(int i=0;i<4;++i) wrong_version[wrong_version.size()-4+i]=static_cast<std::uint8_t>((crc>>(8*i))&0xffu);
 railguard::PacketHeader vh{};railguard::SensorFeaturePayload vq{};assert(!railguard::decode_sensor_feature_packet(wrong_version,vh,vq));

 // Deterministic serial-noise stress: valid frames inserted between random bytes must
 // still emerge in sequence despite false sync bytes and CRC failures.
 std::mt19937 rng(7);std::uniform_int_distribution<int> byte_dist(0,255);
 std::vector<std::uint8_t> stress;
 for(std::uint32_t seq=100;seq<120;++seq){
   for(int i=0;i<37;++i) stress.push_back(static_cast<std::uint8_t>(byte_dist(rng)));
   auto hh=h2;hh.sequence=seq;auto frame=railguard::encode_sensor_feature_packet(hh,s);stress.insert(stress.end(),frame.begin(),frame.end());
 }
 railguard::StreamDecoder ds;std::vector<railguard::Packet> decoded;
 for(std::size_t off=0;off<stress.size();){auto n=std::min<std::size_t>(17,stress.size()-off);auto batch=ds.feed(std::span<const std::uint8_t>(stress).subspan(off,n));decoded.insert(decoded.end(),batch.begin(),batch.end());off+=n;}
 assert(decoded.size()==20);for(std::size_t i=0;i<decoded.size();++i)assert(decoded[i].header.sequence==100+i);
}
