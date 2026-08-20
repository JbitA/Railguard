#include "railguard/synchronizer.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

int main(){
    railguard::FrameSynchronizer sync(4, 25.0);
    for (std::uint64_t ms : {1000, 1020, 1040, 1060}) {
        railguard::CameraFrame f; f.monotonic_ns=ms*1000000ull; f.motion=static_cast<float>(ms);
        sync.push(std::move(f));
    }
    auto m=sync.nearest(1033000000ull);
    assert(m); assert(m->frame->monotonic_ns==1040000000ull); assert(std::abs(m->delta_ms-7.0)<1e-6);
    assert(!sync.nearest(1200000000ull));
    railguard::CameraFrame f; f.monotonic_ns=1080000000ull; sync.push(std::move(f));
    assert(sync.size()==4); // bounded history
    std::cout << "synchronizer tests: PASS\n";
}
