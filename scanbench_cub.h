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
    static scan_algorithm<T> *create(device_type d, const std::vector<T> &h_a);
    static std::string name();
    static std::string api();
};

template<typename K, typename V>
class cub_sort;

template<typename K, typename V>
struct algorithm_factory<cub_sort<K, V> >
{
    static sort_algorithm<K, V> *create(device_type d,
        const typename vector_of<K>::type &h_keys,
        const typename vector_of<V>::type &h_values);
    static std::string name();
    static std::string api();
};

#endif
