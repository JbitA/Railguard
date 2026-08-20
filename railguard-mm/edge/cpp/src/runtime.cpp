#include "railguard/runtime.hpp"
#include <stdexcept>
namespace railguard {
EdgeRuntime::EdgeRuntime(std::unique_ptr<InferenceEngine> engine):engine_(std::move(engine)){if(!engine_)throw std::invalid_argument("engine must not be null");}
RuntimeResult EdgeRuntime::process(std::span<const float> acceleration,std::span<const float> frames,std::span<const float> sensors,std::size_t t,std::size_t h,std::size_t w){
    return {vibration_features(acceleration),engine_->infer(frames,sensors,t,h,w)};
}
}
