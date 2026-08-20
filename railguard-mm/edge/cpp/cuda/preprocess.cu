#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
namespace railguard {
__global__ void bgr_u8_to_rgb_chw_fp32_kernel(const std::uint8_t* src,float* dst,int width,int height,std::size_t pitch){
    const int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;if(x>=width||y>=height)return;
    const auto* px=src+y*pitch+3*x;const int i=y*width+x;const float s=1.0f/255.0f;dst[i]=px[2]*s;dst[width*height+i]=px[1]*s;dst[2*width*height+i]=px[0]*s;
}
extern "C" cudaError_t railguard_bgr_u8_to_rgb_chw_fp32(const std::uint8_t* src,float* dst,int width,int height,std::size_t pitch,cudaStream_t stream){
    dim3 block(16,16),grid((width+15)/16,(height+15)/16);bgr_u8_to_rgb_chw_fp32_kernel<<<grid,block,0,stream>>>(src,dst,width,height,pitch);return cudaGetLastError();
}
}
