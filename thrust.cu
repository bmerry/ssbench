#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <vector>
#include <string>
#include "algorithms.h"
#include "thrust.h"

class thrust_algorithm
{
public:
    template<typename T>
    struct types
    {
        typedef thrust::device_vector<T> vector;
        typedef vector scan_vector;
        typedef vector sort_vector;
    };

    template<typename T>
    static void create(thrust::device_vector<T> &out, std::size_t elements)
    {
        out.resize(elements);
    }

    template<typename T>
    static void copy(const std::vector<T> &src, thrust::device_vector<T> &dst)
    {
        thrust::copy(src.begin(), src.end(), dst.begin());
    }

    template<typename T>
    static void copy(const thrust::device_vector<T> &src, thrust::device_vector<T> &dst)
    {
        thrust::copy(src.begin(), src.end(), dst.begin());
    }

    template<typename T>
    static void copy(const thrust::device_vector<T> &src, std::vector<T> &dst)
    {
        thrust::copy(src.begin(), src.end(), dst.begin());
    }

    template<typename T>
    static void pre_scan(const thrust::device_vector<T> &src, thrust::device_vector<T> &dst)
    {
    }

    template<typename T>
    static void scan(const thrust::device_vector<T> &src, thrust::device_vector<T> &dst)
    {
        thrust::exclusive_scan(src.begin(), src.end(), dst.begin());
    }

    template<typename K, typename V>
    static void pre_sort_by_key(thrust::device_vector<K> &keys, thrust::device_vector<V> &values) {}

    template<typename K, typename V>
    static void sort_by_key(thrust::device_vector<K> &keys, thrust::device_vector<V> &values)
    {
        thrust::sort_by_key(keys.begin(), keys.end(), values.begin());
    }

    template<typename T>
    static void pre_sort(thrust::device_vector<T> &keys) {}

    template<typename T>
    static void sort(thrust::device_vector<T> &keys)
    {
        thrust::sort(keys.begin(), keys.end());
    }


    static void finish()
    {
        cudaDeviceSynchronize();
    }

    static std::string api() { return "thrust"; }

    explicit thrust_algorithm(device_type d)
    {
        if (d != DEVICE_TYPE_GPU)
            throw device_not_supported();
    }
};

/********************************************************************/

template<typename T>
algorithm *algorithm_factory<scan_algorithm<T, thrust_algorithm> >::create(
    device_type d,
    const std::vector<T> &h_a)
{
    return new scan_algorithm<T, thrust_algorithm>(d, h_a);
}

template<typename K>
algorithm *algorithm_factory<sort_algorithm<K, thrust_algorithm> >::create(
    device_type d,
    const std::vector<K> &h_keys)
{
    return new sort_algorithm<K, thrust_algorithm>(d, h_keys);
}

template<typename K, typename V>
algorithm *algorithm_factory<sort_by_key_algorithm<K, V, thrust_algorithm> >::create(
    device_type d,
    const std::vector<K> &h_keys,
    const std::vector<V> &h_values)
{
    return new sort_by_key_algorithm<K, V, thrust_algorithm>(d, h_keys, h_values);
}

template class algorithm_factory<scan_algorithm<int, thrust_algorithm> >;
template class algorithm_factory<sort_algorithm<unsigned int, thrust_algorithm> >;
template class algorithm_factory<sort_by_key_algorithm<unsigned int, unsigned int, thrust_algorithm> >;
