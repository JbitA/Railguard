#include "iis3dwb_dma.h"
#include <math.h>
#include <string.h>
enum { REG_FIFO_CTRL1=0x07,REG_FIFO_CTRL2=0x08,REG_FIFO_CTRL3=0x09,REG_FIFO_CTRL4=0x0A,REG_INT1_CTRL=0x0D,REG_WHOAMI=0x0F,REG_CTRL1_XL=0x10,REG_CTRL3_C=0x12,REG_CTRL4_C=0x13,REG_FIFO_STATUS1=0x3A,REG_FIFO_DATA=0x78 };
#define SPI_READ 0x80u
#define WHOAMI_VALUE 0x7Bu
#define FIFO_WATERMARK 512u

static int32_t dma_cache_span(uint16_t bytes){return (int32_t)(((uint32_t)bytes+31u)&~31u);}
static void dma_prepare(iis3dwb_bus_t*b,uint16_t bytes){
#if defined(__DCACHE_PRESENT) && (__DCACHE_PRESENT == 1U)
 SCB_CleanDCache_by_Addr((uint32_t*)b->tx,dma_cache_span(bytes));
 SCB_InvalidateDCache_by_Addr((uint32_t*)b->rx,dma_cache_span(bytes));
#else
 (void)b;(void)bytes;
#endif
}
static void dma_complete_cache(iis3dwb_bus_t*b,uint16_t bytes){
#if defined(__DCACHE_PRESENT) && (__DCACHE_PRESENT == 1U)
 SCB_InvalidateDCache_by_Addr((uint32_t*)b->rx,dma_cache_span(bytes));
#else
 (void)b;(void)bytes;
#endif
}
static void cs(iis3dwb_bus_t*b,uint8_t s,GPIO_PinState v){HAL_GPIO_WritePin(b->cs_port[s],b->cs_pin[s],v);}
static bool write_reg(iis3dwb_bus_t*b,uint8_t s,uint8_t reg,uint8_t value){uint8_t x[2]={reg,value};cs(b,s,GPIO_PIN_RESET);HAL_StatusTypeDef st=HAL_SPI_Transmit(b->spi,x,2,20);cs(b,s,GPIO_PIN_SET);return st==HAL_OK;}
static bool read_regs(iis3dwb_bus_t*b,uint8_t s,uint8_t reg,uint8_t*out,uint16_t n){uint8_t cmd=reg|SPI_READ;cs(b,s,GPIO_PIN_RESET);HAL_StatusTypeDef a=HAL_SPI_Transmit(b->spi,&cmd,1,20),c=a==HAL_OK?HAL_SPI_Receive(b->spi,out,n,20):a;cs(b,s,GPIO_PIN_SET);return c==HAL_OK;}
static bool configure(iis3dwb_bus_t*b,uint8_t s){uint8_t who=0;if(!read_regs(b,s,REG_WHOAMI,&who,1)||who!=WHOAMI_VALUE)return false;
 /* BDU + auto increment, SPI-only, ±16 g normal 3-axis mode, FIFO batching at 26.667 kHz, 512-word watermark, continuous mode. */
 return write_reg(b,s,REG_CTRL3_C,0x44)&&write_reg(b,s,REG_CTRL4_C,0x04)&&write_reg(b,s,REG_CTRL1_XL,0xA4)&&
        write_reg(b,s,REG_FIFO_CTRL1,(uint8_t)(FIFO_WATERMARK&0xff))&&write_reg(b,s,REG_FIFO_CTRL2,(uint8_t)((FIFO_WATERMARK>>8)&1))&&
        write_reg(b,s,REG_FIFO_CTRL3,0x0A)&&write_reg(b,s,REG_FIFO_CTRL4,0x06)&&write_reg(b,s,REG_INT1_CTRL,0x08);
}
bool iis3dwb_bus_init(iis3dwb_bus_t*b,SPI_HandleTypeDef*spi,GPIO_TypeDef**ports,const uint16_t*pins){memset(b,0,sizeof *b);b->spi=spi;for(uint8_t i=0;i<IIS3DWB_SENSOR_COUNT;i++){b->cs_port[i]=ports[i];b->cs_pin[i]=pins[i];cs(b,i,GPIO_PIN_SET);}HAL_Delay(12);for(uint8_t i=0;i<IIS3DWB_SENSOR_COUNT;i++)if(!configure(b,i))return false;return true;}
void iis3dwb_mark_watermark(iis3dwb_bus_t*b,uint8_t s){if(s<IIS3DWB_SENSOR_COUNT)b->pending_mask|=(uint8_t)(1u<<s);}
void iis3dwb_service(iis3dwb_bus_t*b){
 if(b->dma_busy||!b->pending_mask)return;uint8_t s=0;while(s<IIS3DWB_SENSOR_COUNT&&!(b->pending_mask&(1u<<s)))s++;if(s>=IIS3DWB_SENSOR_COUNT)return;b->pending_mask&=(uint8_t)~(1u<<s);
 uint8_t status[2]={0};if(!read_regs(b,s,REG_FIFO_STATUS1,status,2))return;uint16_t words=(uint16_t)status[0]|((uint16_t)(status[1]&0x03u)<<8);if(status[1]&0x40u)b->fifo_overruns++;if(!words)return;if(words>IIS3DWB_MAX_DRAIN_WORDS)words=IIS3DWB_MAX_DRAIN_WORDS;
 b->active_sensor=s;b->transfer_words=words;b->samples_ready=0;const uint16_t bytes=(uint16_t)(1u+words*IIS3DWB_FIFO_WORD_BYTES);memset(b->tx,0xff,bytes);b->tx[0]=REG_FIFO_DATA|SPI_READ;
 dma_prepare(b,bytes);
 cs(b,s,GPIO_PIN_RESET);b->dma_busy=true;if(HAL_SPI_TransmitReceive_DMA(b->spi,b->tx,b->rx,bytes)!=HAL_OK){b->dma_busy=false;cs(b,s,GPIO_PIN_SET);b->dma_errors++;}
}
void iis3dwb_spi_complete(iis3dwb_bus_t*b,SPI_HandleTypeDef*spi){
 if(spi!=b->spi||!b->dma_busy)return;cs(b,b->active_sensor,GPIO_PIN_SET);b->dma_busy=false;dma_complete_cache(b,(uint16_t)(1u+b->transfer_words*IIS3DWB_FIFO_WORD_BYTES));b->samples_ready=0;
 for(uint16_t w=0;w<b->transfer_words&&b->samples_ready<IIS3DWB_MAX_DRAIN_WORDS;w++){const uint8_t*d=&b->rx[1u+w*IIS3DWB_FIFO_WORD_BYTES];const uint8_t tag=(uint8_t)(d[0]>>3);if(tag!=0x02u)continue;iis3dwb_xyz_t*x=&b->samples[b->samples_ready++];x->x=(int16_t)((uint16_t)d[1]|((uint16_t)d[2]<<8));x->y=(int16_t)((uint16_t)d[3]|((uint16_t)d[4]<<8));x->z=(int16_t)((uint16_t)d[5]|((uint16_t)d[6]<<8));}
 /* Recheck FIFO status after publishing this burst. This handles data accumulated during the DMA without relying on another rising EXTI edge. */
 b->pending_mask|=(uint8_t)(1u<<b->active_sensor);
}
void iis3dwb_spi_error(iis3dwb_bus_t*b,SPI_HandleTypeDef*spi){if(spi!=b->spi)return;if(b->dma_busy)cs(b,b->active_sensor,GPIO_PIN_SET);b->dma_busy=false;b->dma_errors++;b->pending_mask|=(uint8_t)(1u<<b->active_sensor);}
float iis3dwb_rms_ms2(const iis3dwb_xyz_t*s,uint16_t n,unsigned axis){if(!s||!n||axis>2)return 0;double q=0;for(uint16_t i=0;i<n;i++){int16_t raw=axis==0?s[i].x:axis==1?s[i].y:s[i].z;const double a=(double)raw*0.000488*9.80665;q+=a*a;}return(float)sqrt(q/n);}
