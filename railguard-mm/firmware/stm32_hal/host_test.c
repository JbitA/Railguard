#include "railguard_packet.h"
#include "gnss_time.h"
#include "gnss_stream.h"
#include "dsp_features.h"
#include "environment_decode.h"
#include "transport_queue.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static unsigned gnss_lines;
static char last_gnss_line[RG_GNSS_LINE_MAX];
static void capture_gnss_line(const char *line,size_t len,void *context){(void)context;assert(len<sizeof last_gnss_line);memcpy(last_gnss_line,line,len);last_gnss_line[len]=0;gnss_lines++;}
static rg_transport_start_result_t fake_start(const uint8_t *data,uint16_t len,void *context){
 (void)data;(void)len;unsigned *calls=(unsigned*)context;(*calls)++;return RG_TRANSPORT_STARTED;
}
int main(void){
 gnss_fix_t fix={0};const char*r="$GNRMC,123519.00,A,4807.038,N,01131.000,E,10.0,0.0,230394,,,A*00";assert(gnss_rmc_fix(r,strlen(r),&fix));assert(fix.epoch==764426119u);assert(fabsf(fix.latitude-48.1173f)<1e-3f);assert(fabsf(fix.longitude-11.516667f)<1e-3f);assert(fabsf(fix.speed_mps-5.14444f)<1e-3f);
 rg_gnss_stream_t gs;rg_gnss_stream_init(&gs);const uint8_t c1[]="noise$GNRMC,123519.00,A,4807.";const uint8_t c2[]="038,N,01131.000,E,10.0,0.0,230394,,,A*00\r\n";rg_gnss_stream_feed(&gs,c1,sizeof(c1)-1u,capture_gnss_line,NULL);assert(gnss_lines==0u);rg_gnss_stream_feed(&gs,c2,sizeof(c2)-1u,capture_gnss_line,NULL);assert(gnss_lines==1u);assert(strcmp(last_gnss_line,r)==0);
 rg_xyz_i16_t samples[512];for(unsigned i=0;i<512;i++){samples[i].x=(int16_t)(1000.0*sin(2.0*3.141592653589793*75.0*i/26667.0));samples[i].y=(int16_t)(500.0*sin(2.0*3.141592653589793*200.0*i/26667.0));samples[i].z=0;}rg_vibration_features_t vf;assert(rg_compute_vibration_features(samples,512,26667.0f,&vf));assert(vf.rms>0.1f);assert(vf.peak>=vf.rms);assert(vf.crest_factor>=1.0f);
 rg_sensor_feature_payload_t p={0};p.sensor_id=2;p.flags=RG_FLAG_GNSS_VALID;p.window_samples=512;p.sample_rate_hz=26667.0f;memcpy(p.axis_rms,vf.axis_rms,sizeof p.axis_rms);p.rms=vf.rms;p.peak=vf.peak;p.kurtosis=vf.kurtosis;p.crest_factor=vf.crest_factor;memcpy(p.band_energy,vf.band_energy,sizeof p.band_energy);p.latitude=fix.latitude;p.longitude=fix.longitude;p.speed_mps=fix.speed_mps;
 uint8_t sht[6]={0x66,0x66,0x93,0x80,0x00,0xa2};float tc=0.0f,rh=0.0f;assert(rg_sensirion_crc8(sht,2)==0x93u);assert(rg_sht4x_decode(sht,&tc,&rh));assert(fabsf(tc-25.0f)<1e-4f);assert(fabsf(rh-0.5650095f)<1e-4f);sht[2]^=1u;assert(!rg_sht4x_decode(sht,&tc,&rh));
 rg_timestamp_t ts={42,1700000000u,123456u};uint8_t b[160];size_t n=rg_encode_sensor_feature_packet(b,sizeof b,&ts,&p);assert(n==94u);assert(b[0]=='R'&&b[1]=='G'&&b[3]==RG_PACKET_TYPE_SENSOR_FEATURES);uint32_t got;memcpy(&got,b+n-4,4);assert(got==rg_crc32_ieee(b,n-4));
 rg_transport_queue_t tq;rg_transport_queue_init(&tq);unsigned calls=0;assert(rg_transport_queue_enqueue(&tq,b,(uint16_t)n));assert(rg_transport_queue_depth(&tq)==1u);rg_transport_queue_service(&tq,fake_start,&calls);assert(calls==1u&&tq.in_flight&&rg_transport_queue_depth(&tq)==1u);rg_transport_queue_tx_complete_isr(&tq);rg_transport_queue_service(&tq,fake_start,&calls);assert(!tq.in_flight&&rg_transport_queue_depth(&tq)==0u&&tq.completed==1u);for(unsigned i=0;i<RG_TRANSPORT_QUEUE_DEPTH;i++)assert(rg_transport_queue_enqueue(&tq,b,(uint16_t)n));assert(!rg_transport_queue_enqueue(&tq,b,(uint16_t)n));assert(tq.dropped==1u);
 puts("firmware host tests: PASS");return 0;
}
