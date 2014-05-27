#ifndef SCANBENCH_THRUST_H
#define SCANBENCH_THRUST_H

#include <string>
#include "scanbench_algorithms.h"

template<typename T>
class thrust_scan;

template<typename T>
class thrust_sort;

template<typename T>
struct algorithm_factory<thrust_scan<T> >
{
    static scan_algorithm<T> *create(device_type d, const std::vector<T> &h_a);
    static std::string name();
    static std::string api();
};

template<typename T>
struct algorithm_factory<thrust_sort<T> >
{
    static sort_algorithm<T> *create(device_type d, const std::vector<T> &h_a);
    static std::string name();
    static std::string api();
};

#endif
