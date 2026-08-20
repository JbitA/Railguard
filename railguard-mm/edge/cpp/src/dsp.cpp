#include "railguard/dsp.hpp"
#include <algorithm>
#include <cmath>
#include <numbers>
namespace railguard {
VibrationFeatures vibration_features(std::span<const float> x, float sample_rate_hz) noexcept {
    VibrationFeatures f{}; if (x.empty()) return f;
    double sum2=0.0,sum4=0.0; float peak=0.0f;
    for(float v:x){ const double d=v; const double d2=d*d; sum2+=d2; sum4+=d2*d2; peak=std::max(peak,std::abs(v)); }
    const double m2=sum2/x.size(); f.rms=static_cast<float>(std::sqrt(m2)); f.peak=peak;
    f.kurtosis=m2>1e-18?static_cast<float>((sum4/x.size())/(m2*m2)):0.0f;
    f.crest_factor=f.rms>1e-9f?f.peak/f.rms:0.0f;
    // Four-bin Goertzel-style diagnostic energies: 25, 75, 200, 500 Hz.
    constexpr std::array<float,4> freqs{25.f,75.f,200.f,500.f};
    for(std::size_t b=0;b<freqs.size();++b){
        const double w=2.0*std::numbers::pi*freqs[b]/sample_rate_hz; const double c=2.0*std::cos(w);
        double s0=0,s1=0,s2=0; for(float v:x){s0=v+c*s1-s2;s2=s1;s1=s0;} const double power=s1*s1+s2*s2-c*s1*s2;
        f.band_energy[b]=static_cast<float>(power/x.size());
    }
    return f;
}
}
