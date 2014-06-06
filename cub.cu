#include <cub/cub.cuh>
#include <vector>
#include <string>
#include <cstddef>
#include <boost/utility.hpp>
#include "algorithms.h"
#include "cub.h"

template<typename T>
class cuda_vector : public boost::noncopyable
{
private:
    T *ptr;
    std::size_t elements;

public:
    cuda_vector() : ptr(NULL), elements(0) {}

    explicit cuda_vector(std::size_t elements)
    {
        std::size_t bytes = elements * sizeof(T);
        cudaMalloc(&ptr, bytes);
        this->elements = elements;
    }

    ~cuda_vector()
    {
        if (ptr != NULL)
            cudaFree(ptr);
    }

    T *data() const { return ptr; }

    std::size_t size() const
    {
        return elements;
    }

    void swap(cuda_vector<T> &other)
    {
        std::swap(ptr, other.ptr);
        std::swap(elements, other.elements);
    }
};

template<typename T>
struct cub_double_vector : public boost::noncopyable
{
    // mutable because Current() doesn't work on const objects
    mutable cub::DoubleBuffer<T> ptrs;
    std::size_t elements;

    cub_double_vector() : ptrs(NULL, NULL), elements(0) {}
    explicit cub_double_vector(std::size_t size)
    {
        std::size_t bytes = size * sizeof(T);
        cudaMalloc(&ptrs.d_buffers[0], bytes);
        cudaMalloc(&ptrs.d_buffers[1], bytes);
        elements = size;
    }
    ~cub_double_vector()
    {
        if (ptrs.d_buffers[0])
            cudaFree(ptrs.d_buffers[0]);
        if (ptrs.d_buffers[1])
            cudaFree(ptrs.d_buffers[1]);
    }

    std::size_t size() const { return elements; }

    void swap(cub_double_vector<T> &other)
    {
        std::swap(ptrs, other.ptrs);
        std::swap(elements, other.elements);
    }
};

class cub_algorithm
{
private:
    void *d_temp;
    std::size_t d_temp_size;

public:
    template<typename T>
    struct types
    {
        typedef cuda_vector<T> vector;
        typedef vector scan_vector;
        typedef cub_double_vector<T> sort_vector;
    };

    template<typename T>
    static void create(cuda_vector<T> &out, std::size_t elements)
    {
        cuda_vector<T>(elements).swap(out);
    }

    template<typename T>
    static void create(cub_double_vector<T> &out, std::size_t elements)
    {
        cub_double_vector<T>(elements).swap(out);
    }

    template<typename T>
    static void copy(const std::vector<T> &src, cuda_vector<T> &dst)
    {
        cudaMemcpy(dst.data(), &src[0], src.size() * sizeof(T), cudaMemcpyHostToDevice);
    }

    template<typename T>
    static void copy(const cuda_vector<T> &src, cub_double_vector<T> &dst)
    {
        cudaMemcpy(dst.ptrs.Current(), src.data(), src.size() * sizeof(T), cudaMemcpyDeviceToDevice);
    }

    template<typename T>
    static void copy(const cuda_vector<T> &src, std::vector<T> &dst)
    {
        cudaMemcpy(&dst[0], src.data(), src.size() * sizeof(T), cudaMemcpyDeviceToHost);
    }

    template<typename T>
    static void copy(const cub_double_vector<T> &src, std::vector<T> &dst)
    {
        cudaMemcpy(&dst[0], src.ptrs.Current(), src.size() * sizeof(T), cudaMemcpyDeviceToHost);
    }

    template<typename T>
    void pre_scan(const cuda_vector<T> &src, cuda_vector<T> &dst)
    {
        cub::DeviceScan::ExclusiveSum(NULL, d_temp_size, src.data(), dst.data(), src.size());
        cudaMalloc(&d_temp, d_temp_size);
    }

    template<typename T>
    void scan(const cuda_vector<T> &src, cuda_vector<T> &dst)
    {
        cub::DeviceScan::ExclusiveSum(d_temp, d_temp_size, src.data(), dst.data(), src.size());
    }

    template<typename K>
    void pre_sort(cub_double_vector<K> &keys)
    {
        cub::DeviceRadixSort::SortKeys(NULL, d_temp_size, keys.ptrs, keys.size());
        cudaMalloc(&d_temp, d_temp_size);
    }

    template<typename K>
    void sort(cub_double_vector<K> &keys)
    {
        cub::DeviceRadixSort::SortKeys(d_temp, d_temp_size, keys.ptrs, keys.size());
    }

    template<typename K, typename V>
    void pre_sort_by_key(cub_double_vector<K> &keys, cub_double_vector<V> &values)
    {
        cub::DeviceRadixSort::SortPairs(NULL, d_temp_size, keys.ptrs, values.ptrs, keys.size());
        cudaMalloc(&d_temp, d_temp_size);
    }

    template<typename K, typename V>
    void sort_by_key(cub_double_vector<K> &keys, cub_double_vector<V> &values)
    {
        cub::DeviceRadixSort::SortPairs(d_temp, d_temp_size, keys.ptrs, values.ptrs, keys.size());
    }

    static void finish()
    {
        cudaDeviceSynchronize();
    }

    static std::string api() { return "cub"; }

    explicit cub_algorithm(device_type d) : d_temp(NULL), d_temp_size(0)
    {
        if (d != DEVICE_TYPE_GPU)
            throw device_not_supported();
    }

    ~cub_algorithm()
    {
        if (d_temp != NULL)
            cudaFree(d_temp);
    }
};

/********************************************************************/

template<typename T>
algorithm *algorithm_factory<scan_algorithm<T, cub_algorithm> >::create(
    device_type d,
    const std::vector<T> &h_a)
{
    return new scan_algorithm<T, cub_algorithm>(d, h_a);
}

template<typename K>
algorithm *algorithm_factory<sort_algorithm<K, cub_algorithm> >::create(
    device_type d,
    const std::vector<K> &h_keys)
{
    return new sort_algorithm<K, cub_algorithm>(d, h_keys);
}

template<typename K, typename V>
algorithm *algorithm_factory<sort_by_key_algorithm<K, V, cub_algorithm> >::create(
    device_type d,
    const std::vector<K> &h_keys,
    const std::vector<V> &h_values)
{
    return new sort_by_key_algorithm<K, V, cub_algorithm>(d, h_keys, h_values);
}

template class algorithm_factory<scan_algorithm<int, cub_algorithm> >;
template class algorithm_factory<sort_algorithm<unsigned int, cub_algorithm> >;
template class algorithm_factory<sort_by_key_algorithm<unsigned int, unsigned int, cub_algorithm> >;
