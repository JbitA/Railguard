#include "railguard_packet.h"
#include <string.h>
#pragma pack(push,1)
typedef struct { uint8_t sync[2], version, type; uint16_t payload_len; uint32_t seq, pps_epoch, sub_us; } rg_header_t;
#pragma pack(pop)
uint32_t rg_crc32_ieee(const uint8_t *data,size_t len){uint32_t crc=0xffffffffu;for(size_t i=0;i<len;i++){crc^=data[i];for(unsigned b=0;b<8;b++)crc=(crc>>1)^((0u-(crc&1u))&0xedb88320u);}return ~crc;}
static size_t encode(uint8_t *dst,size_t cap,const rg_timestamp_t *ts,const void *payload,uint16_t payload_len,uint8_t type){
 const size_t need=sizeof(rg_header_t)+payload_len+4u;if(!dst||!ts||!payload||cap<need)return 0;
 rg_header_t h={{'R','G'},RG_PACKET_VERSION,type,payload_len,ts->sequence,ts->pps_epoch,ts->sub_us};memcpy(dst,&h,sizeof h);memcpy(dst+sizeof h,payload,payload_len);uint32_t crc=rg_crc32_ieee(dst,sizeof h+payload_len);memcpy(dst+sizeof h+payload_len,&crc,4);return need;
}
size_t rg_encode_feature_packet(uint8_t *dst,size_t cap,const rg_timestamp_t *ts,const rg_feature_payload_t *p){return encode(dst,cap,ts,p,(uint16_t)sizeof(*p),RG_PACKET_TYPE_FEATURES);}
size_t rg_encode_sensor_feature_packet(uint8_t *dst,size_t cap,const rg_timestamp_t *ts,const rg_sensor_feature_payload_t *p){return encode(dst,cap,ts,p,(uint16_t)sizeof(*p),RG_PACKET_TYPE_SENSOR_FEATURES);}
