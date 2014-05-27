#include <cub/cub.cuh>
#include <vector>
#include <string>
#include <cstddef>
#include <boost/utility.hpp>
#include "scanbench_algorithms.h"
#include "scanbench_cub.h"

template<typename T>
class cuda_vector : public boost::noncopyable
{
private:
    T *ptr;
    std::size_t elements;

public:
    cuda_vector() : ptr(NULL), elements(0) {}
    explicit cuda_vector(const std::vector<T> &v)
    {
        std::size_t bytes = v.size() * sizeof(T);
        cudaMalloc(&ptr, bytes);
        cudaMemcpy(ptr, &v[0], bytes, cudaMemcpyHostToDevice);
        elements = v.size();
    }

    ~cuda_vector()
    {
        if (ptr != NULL)
            cudaFree(ptr);
    }

    operator T*() const
    {
        return ptr;
    }

    std::size_t size() const
    {
        return elements;
    }
};

template<typename T>
struct cub_double_vector : public boost::noncopyable
{
    cub::DoubleBuffer<T> ptrs;
    std::size_t elements;

    double_vector() : ptrs(NULL, NULL), elements(0) {}
    explicit double_vector(std::size_t size)
    {
        std::size_t bytes = size() * sizeof(T);
        cudaMalloc(&ptrs.d_buffers[0], bytes);
        cudaMalloc(&ptrs.d_buffers[1], bytes);
        elements = size;
    }
    ~double_vector()
    {
        if (ptrs.d_buffers[0])
            cudaFree(ptrs.d_buffers[0]);
        if (ptrs.d_buffers[1])
            cudaFree(ptrs.d_buffers[1]);
    }

    std::size_t size() const { return elements; }
};

template<typename T>
struct cub_traits
{
    typedef cuda_vector vector;
    typedef cub_double_vector double_vector;

    static void copy(const vector &src, double_vector &trg)
    {
        cudaMemcpyAsync(trg.ptrs.Current(), src, src.size() * sizeof(T), cudaMemcpyDeviceToDevice);
    }

    static std::vector<T> get(const double_vector &v)
    {
        std::vector<T> ans(v.size());
        cudaMemcpy(ans.data(), const_cast<cub::DoubleBuffer<T> &>(v.ptrs).Current(),
                   v.size() * sizeof(T), cudaMemcpyDeviceToHost);
        return ans;
    }

    template<typename K>
    static void sort(cub_double_vector<K> &keys, double_vector &values,
                     void *d_temp, std::size_t &temp_bytes)
    {
        cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, keys.ptrs, values.ptrs, keys.size());
    }
};

template<>
struct cub_traits<void>
{
    struct vector
    {
        vector() {}
        explicit vector(const void_vector &) {}
        std::size_t size() const { return 0; }
    };

    struct double_vector
    {
        double_vector() {}
        explicit double_vector(std::size_t size) {}
        std::size_t size() const { return 0; }
    };

    static void copy(const vector &src, double_vector &trg)
    {
    }

    static void_vector get(const vector &v)
    {
        return void_vector();
    }

    template<typename K>
    static void sort(cub_double_vector<K> &keys, double_vector &values,
                     void *d_temp, std::size_t &temp_bytes)
    {
        cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, keys, keys.elements);
    }
};

template<typename T>
class cub_scan : public scan_algorithm<T>
{
private:
    T *d_a;
    T *d_scan;
    void *d_temp;
    std::size_t elements;
    std::size_t temp_bytes;

public:
    cub_scan(device_type d, const std::vector<T> &h_a) :
        scan_algorithm<T>(h_a),
        d_a(NULL), d_scan(NULL), d_temp(NULL), elements(h_a.size()),
        temp_bytes(0)
    {
        if (d != DEVICE_TYPE_GPU)
            throw device_not_supported();

        std::size_t bytes = elements * sizeof(T);
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_scan, bytes);
        cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);

        cub::DeviceScan::ExclusiveSum(NULL, temp_bytes, d_a, d_scan, elements);
        cudaMalloc(&d_temp, temp_bytes);
    }

    ~cub_scan()
    {
        cudaFree(d_a);
        cudaFree(d_scan);
        cudaFree(d_temp);
    }

    static std::string name() { return "cub::DeviceScan::ExclusiveSum"; }
    static std::string api() { return "cub"; }
    virtual void finish() { cudaDeviceSynchronize(); }

    virtual void run()
    {
        cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_a, d_scan, elements);
    }

    virtual std::vector<T> get() const
    {
        std::vector<T> ans(elements);
        cudaMemcpy(ans.data(), d_scan, elements * sizeof(T), cudaMemcpyDeviceToHost);
        return ans;
    }
};

template<typename T>
scan_algorithm<T> *algorithm_factory<cub_scan<T> >::create(const std::vector<T> &h_a)
{
    return new cub_scan<T>(h_a);
}

template<typename T>
std::string algorithm_factory<cub_scan<T> >::name() { return cub_scan<T>::name(); }

template<typename T>
std::string algorithm_factory<cub_scan<T> >::api() { return cub_scan<T>::api(); }

template struct algorithm_factory<cub_scan<int> >;

/********************************************************************/

template<typename K, typename V>
class cub_sort : public sort_algorithm<K, V>
{
private:
    typedef typename vector_of<K>::type key_vector;
    typedef typename vector_of<V>::type value_vector;
    typedef typename cub_traits<K>::vector d_key_vector;
    typedef typename cub_traits<V>::vector d_value_vector;
    typedef typename cub_traits<K>::double_vector double_key_vector;
    typedef typename cub_traits<V>::double_vector double_value_vector;

    d_key_vector d_keys;
    double_key_vector d_sorted_keys;
    d_value_vector d_values;
    double_value_vector d_sorted_values;

    void *d_temp;
    std::size_t temp_bytes;

public:
    cub_sort(device_type d, const key_vector &h_keys, const value_vector &h_values)
        : sort_algorithm<K, V>(h_keys, h_values),
        d_keys(h_keys), d_sorted_keys(h_keys.size()),
        d_values(h_values), d_sorted_values(h_values.size()),
        d_temp(NULL),
        temp_bytes(0)
    {
        if (d != DEVICE_TYPE_GPU)
            throw device_not_supported();

        cub_traits<V>::template sort<K>(d_sorted_keys, d_sorted_values, NULL, temp_bytes);
        cudaMalloc(&d_temp, temp_bytes);
    }

    ~cub_sort()
    {
        cudaFree(d_temp);
    }

    static std::string name() { return "cub::DeviceRadixSort::SortKeys"; }
    static std::string api() { return "cub"; }
    virtual void finish() { cudaDeviceSynchronize(); }

    virtual void run()
    {
        cub_traits<K>::copy(d_keys, d_sorted_keys);
        cub_traits<K>::copy(d_values, d_sorted_values);
        cub_traits<V>::template sort<K>(d_sorted_keys, d_sorted_values, d_temp, temp_bytes);
    }

    virtual std::pair<key_vector, value_vector> get() const
    {
        return std::make_pair(
            cub_traits<K>::get(d_sorted_keys),
            cub_traits<V>::get(d_sorted_values));
    }
};

template<typename K, typename V>
sort_algorithm<K, V> *algorithm_factory<cub_sort<K, V> >::create(
    const typename vector_of<K>::type &h_keys,
    const typename vector_of<V>::type &h_values)
{
    return new cub_sort<K, V>(h_keys, h_values);
}

template<typename K, typename V>
std::string algorithm_factory<cub_sort<K, V> >::name() { return cub_sort<K, V>::name(); }

template<typename K, typename V>
std::string algorithm_factory<cub_sort<K, V> >::api() { return cub_sort<K, V>::api(); }

template struct algorithm_factory<cub_sort<unsigned int, void> >;
template struct algorithm_factory<cub_sort<unsigned int, unsigned int> >;
