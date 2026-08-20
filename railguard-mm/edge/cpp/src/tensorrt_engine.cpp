#include "railguard/inference.hpp"
#include "railguard/model_contract.hpp"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <array>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace railguard {
namespace {
void cuda_check(cudaError_t e,const char* what){if(e!=cudaSuccess)throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(e));}
class Logger final:public nvinfer1::ILogger{public:void log(Severity s,const char* m) noexcept override{if(s<=Severity::kWARNING)std::fprintf(stderr,"TensorRT: %s\n",m);}};
template<class T> struct TrtDelete{void operator()(T* p)const noexcept{delete p;}};
struct DeviceBuffer{void* p=nullptr;std::size_t capacity=0;~DeviceBuffer(){if(p)cudaFree(p);}void ensure(std::size_t n){if(n<=capacity)return;if(p)cudaFree(p);cuda_check(cudaMalloc(&p,n),"cudaMalloc");capacity=n;}};
static std::size_t volume(const nvinfer1::Dims& d){std::size_t v=1;for(int i=0;i<d.nbDims;i++){if(d.d[i]<0)throw std::runtime_error("unresolved dynamic TensorRT dimension");v*=static_cast<std::size_t>(d.d[i]);}return v;}
static nvinfer1::Dims make_dims(std::initializer_list<int> values){nvinfer1::Dims d{};d.nbDims=static_cast<int>(values.size());int i=0;for(int v:values)d.d[i++]=v;return d;}

class TensorRtEngine final:public InferenceEngine{
public:
 explicit TensorRtEngine(const std::string& path){
   std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("cannot open TensorRT engine: "+path);std::vector<char>b((std::istreambuf_iterator<char>(f)),{});
   runtime_.reset(nvinfer1::createInferRuntime(logger_));if(!runtime_)throw std::runtime_error("createInferRuntime failed");
   engine_.reset(runtime_->deserializeCudaEngine(b.data(),b.size()));if(!engine_)throw std::runtime_error("deserializeCudaEngine failed");
   context_.reset(engine_->createExecutionContext());if(!context_)throw std::runtime_error("createExecutionContext failed");
   cuda_check(cudaStreamCreate(&stream_),"cudaStreamCreate");
   for(const char* name:{"frames","sensors","vibration","vision","anomaly_probability"}){
      if(engine_->getTensorIOMode(name)==nvinfer1::TensorIOMode::kNONE)throw std::runtime_error(std::string("missing TensorRT tensor: ")+name);
      if(engine_->getTensorDataType(name)!=nvinfer1::DataType::kFLOAT)throw std::runtime_error(std::string("expected FP32 I/O tensor: ")+name);
   }
 }
 ~TensorRtEngine() override { if(stream_) cudaStreamDestroy(stream_); }
 Prediction infer(std::span<const float> frames,std::span<const float> sensors,std::size_t t,std::size_t h,std::size_t w) override{
   if(!context_->setInputShape("frames",make_dims({1,(int)t,3,(int)h,(int)w})))throw std::runtime_error("TensorRT rejected frames shape");
   const auto sensor_engine_shape=engine_->getTensorShape("sensors");
   if(sensor_engine_shape.nbDims!=3 || (sensor_engine_shape.d[2]>=0 && sensor_engine_shape.d[2]!=(int)kSensorFeatureDim)) throw std::runtime_error("TensorRT sensor feature contract mismatch");
   if(!context_->setInputShape("sensors",make_dims({1,(int)t,(int)kSensorFeatureDim})))throw std::runtime_error("TensorRT rejected sensors shape");
   const std::size_t nf=volume(context_->getTensorShape("frames")), ns=volume(context_->getTensorShape("sensors"));
   if(frames.size()!=nf||sensors.size()!=ns)throw std::runtime_error("TensorRT input span size does not match engine shape");
   const std::size_t nv=volume(context_->getTensorShape("vibration")), nvis=volume(context_->getTensorShape("vision")), na=volume(context_->getTensorShape("anomaly_probability"));
   if(nv!=3||nvis!=3||na!=1)throw std::runtime_error("unexpected TensorRT output shape");
   frames_.ensure(nf*sizeof(float)); sensors_.ensure(ns*sizeof(float)); vibration_.ensure(3*sizeof(float)); vision_.ensure(3*sizeof(float)); anomaly_.ensure(sizeof(float));
   cuda_check(cudaMemcpyAsync(frames_.p,frames.data(),nf*sizeof(float),cudaMemcpyHostToDevice,stream_),"copy frames H2D");
   cuda_check(cudaMemcpyAsync(sensors_.p,sensors.data(),ns*sizeof(float),cudaMemcpyHostToDevice,stream_),"copy sensors H2D");
   context_->setTensorAddress("frames",frames_.p);context_->setTensorAddress("sensors",sensors_.p);context_->setTensorAddress("vibration",vibration_.p);context_->setTensorAddress("vision",vision_.p);context_->setTensorAddress("anomaly_probability",anomaly_.p);
   if(!context_->enqueueV3(stream_))throw std::runtime_error("TensorRT enqueueV3 failed");
   Prediction out{};cuda_check(cudaMemcpyAsync(out.vibration.data(),vibration_.p,3*sizeof(float),cudaMemcpyDeviceToHost,stream_),"copy vibration D2H");cuda_check(cudaMemcpyAsync(out.vision.data(),vision_.p,3*sizeof(float),cudaMemcpyDeviceToHost,stream_),"copy vision D2H");cuda_check(cudaMemcpyAsync(&out.anomaly_probability,anomaly_.p,sizeof(float),cudaMemcpyDeviceToHost,stream_),"copy anomaly D2H");cuda_check(cudaStreamSynchronize(stream_),"TensorRT synchronize");if(!validate_prediction(out))throw std::runtime_error("TensorRT produced a non-finite or physically invalid prediction");return out;
 }
 std::string backend()const override{return "tensorrt";}
private:
 Logger logger_;cudaStream_t stream_{};std::unique_ptr<nvinfer1::IRuntime,TrtDelete<nvinfer1::IRuntime>>runtime_;std::unique_ptr<nvinfer1::ICudaEngine,TrtDelete<nvinfer1::ICudaEngine>>engine_;std::unique_ptr<nvinfer1::IExecutionContext,TrtDelete<nvinfer1::IExecutionContext>>context_;DeviceBuffer frames_,sensors_,vibration_,vision_,anomaly_;
};
}
std::unique_ptr<InferenceEngine> make_inference_engine(const std::string& path){if(path.empty())throw std::invalid_argument("TensorRT build requires an engine path");return std::make_unique<TensorRtEngine>(path);}
}
