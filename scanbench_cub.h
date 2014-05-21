#ifndef SCANBENCH_CUB_H
#define SCANBENCH_CUB_H

#include <vector>
#include <string>
#include "scanbench_algorithms.h"

template<typename T>
class cub_scan;

template<typename T>
struct algorithm_factory<cub_scan<T> >
{
    static scan_algorithm<T> *create(const std::vector<T> &h_a);
    static std::string name();
    static std::string api();
};

template<typename T>
class cub_sort;

template<typename T>
struct algorithm_factory<cub_sort<T> >
{
    static sort_algorithm<T> *create(const std::vector<T> &h_a);
    static std::string name();
    static std::string api();
};

#endif
