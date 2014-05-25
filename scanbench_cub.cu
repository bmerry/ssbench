#include <cub/cub.cuh>
#include <vector>
#include <string>
#include <cstddef>
#include "scanbench_algorithms.h"
#include "scanbench_cub.h"

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

template<typename T>
class cub_sort : public sort_algorithm<T>
{
private:
    T *d_a;
    cub::DoubleBuffer<T> d_target;
    void *d_temp;
    std::size_t elements;
    std::size_t temp_bytes;

public:
    cub_sort(device_type d, const std::vector<T> &h_a)
        : sort_algorithm<T>(h_a),
        d_a(NULL), d_temp(NULL), elements(h_a.size()),
        temp_bytes(0)
    {
        if (d != DEVICE_TYPE_GPU)
            throw device_not_supported();

        std::size_t bytes = elements * sizeof(T);
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_target.d_buffers[0], bytes);
        cudaMalloc(&d_target.d_buffers[1], bytes);
        d_target.selector = 0;

        cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);

        cub::DeviceRadixSort::SortKeys(NULL, temp_bytes, d_target, elements);
        cudaMalloc(&d_temp, temp_bytes);
    }

    ~cub_sort()
    {
        cudaFree(d_a);
        cudaFree(d_target.d_buffers[0]);
        cudaFree(d_target.d_buffers[1]);
        cudaFree(d_temp);
    }

    static std::string name() { return "cub::DeviceRadixSort::SortKeys"; }
    static std::string api() { return "cub"; }
    virtual void finish() { cudaDeviceSynchronize(); }

    virtual void run()
    {
        cudaMemcpyAsync(d_target.Current(), d_a, elements * sizeof(T), cudaMemcpyDeviceToDevice);
        cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_target, elements);
    }

    virtual std::vector<T> get() const
    {
        std::vector<T> ans(elements);
        cudaMemcpy(ans.data(), const_cast<cub::DoubleBuffer<T> &>(d_target).Current(),
                   elements * sizeof(T), cudaMemcpyDeviceToHost);
        return ans;
    }
};

template<typename T>
sort_algorithm<T> *algorithm_factory<cub_sort<T> >::create(const std::vector<T> &h_a)
{
    return new cub_sort<T>(h_a);
}

template<typename T>
std::string algorithm_factory<cub_sort<T> >::name() { return cub_sort<T>::name(); }

template<typename T>
std::string algorithm_factory<cub_sort<T> >::api() { return cub_sort<T>::api(); }

template struct algorithm_factory<cub_sort<unsigned int> >;
