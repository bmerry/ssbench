#ifndef SCANBENCH_THRUST_H
#define SCANBENCH_THRUST_H

#include <string>
#include "scanbench_algorithms.h"

template<typename T>
class thrust_scan;

template<typename K, typename V>
class thrust_sort;

template<typename T>
struct algorithm_factory<thrust_scan<T> >
{
    static scan_algorithm<T> *create(const std::vector<T> &h_a);
    static std::string name();
    static std::string api();
};

template<typename K, typename V>
struct algorithm_factory<thrust_sort<K, V> >
{
    static sort_algorithm<K, V> *create(
        const typename vector_of<K>::type h_keys,
        const typename vector_of<V>::type h_values);
    static std::string name();
    static std::string api();
};

#endif
