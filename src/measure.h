#ifndef __IMGVIEW_MEASURE
#define __IMGVIEW_MEASURE

#include <chrono>

template<typename T>
double measure(T f) {
    using namespace std::chrono;
    high_resolution_clock::time_point t0 = high_resolution_clock::now();
    f();
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t1 - t0);
    return time_span.count();
}

#endif