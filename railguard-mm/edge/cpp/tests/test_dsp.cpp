#include "railguard/dsp.hpp"
#include <cassert>
#include <cmath>
#include <numbers>
#include <vector>
int main(){std::vector<float>x(512);for(size_t i=0;i<x.size();++i)x[i]=.8f*std::sin(2*std::numbers::pi*75.0*i/2000.0);auto f=railguard::vibration_features(x);assert(std::abs(f.rms-.5657f)<.01f);assert(f.peak>.79f&&f.peak<.81f);assert(f.band_energy[1]>f.band_energy[0]);}
