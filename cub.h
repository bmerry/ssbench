#ifndef SCANBENCH_CUB_H
#define SCANBENCH_CUB_H

#include <vector>
#include <string>
#include "algorithms.h"

class cub_algorithm;

template<typename T>
struct algorithm_factory<scan_algorithm<T, cub_algorithm> >
{
    static algorithm *create(device_type d, const std::vector<T> &h_a);
    static std::string api() { return "cub"; }
};

template<typename K>
struct algorithm_factory<sort_algorithm<K, cub_algorithm> >
{
    static algorithm *create(device_type d,
        const std::vector<K> &h_keys);
    static std::string api() { return "cub"; }
};

template<typename K, typename V>
struct algorithm_factory<sort_by_key_algorithm<K, V, cub_algorithm> >
{
    static algorithm *create(device_type d,
        const std::vector<K> &h_keys,
        const std::vector<V> &h_values);
    static std::string api() { return "cub"; }
};

#endif
