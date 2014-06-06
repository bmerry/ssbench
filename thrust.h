#ifndef SCANBENCH_THRUST_H
#define SCANBENCH_THRUST_H

#include <string>
#include "algorithms.h"

class thrust_algorithm;

template<typename T>
struct algorithm_factory<scan_algorithm<T, thrust_algorithm> >
{
    static algorithm *create(device_type d, const std::vector<T> &h_a);
    static std::string api() { return "thrust"; }
};

template<typename K>
struct algorithm_factory<sort_algorithm<K, thrust_algorithm> >
{
    static algorithm *create(device_type d,
        const std::vector<K> &h_keys);
    static std::string api() { return "thrust"; }
};

template<typename K, typename V>
struct algorithm_factory<sort_by_key_algorithm<K, V, thrust_algorithm> >
{
    static algorithm *create(device_type d,
        const std::vector<K> &h_keys,
        const std::vector<V> &h_values);
    static std::string api() { return "thrust"; }
};

#endif
