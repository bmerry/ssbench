#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <vector>
#include <string>
#include "scanbench_algorithms.h"
#include "scanbench_thrust.h"

template<typename T>
class thrust_scan : public scan_algorithm<T>
{
private:
    thrust::device_vector<T> d_a;
    thrust::device_vector<T> d_scan;

public:
    thrust_scan(device_type d, const std::vector<T> &h_a)
        : scan_algorithm<T>(h_a), d_a(h_a), d_scan(h_a.size())
    {
        if (d != DEVICE_TYPE_GPU)
            throw device_not_supported();
    }

    static std::string name() { return "thrust::exclusive_scan"; }
    static std::string api() { return "thrust"; }
    virtual void finish() { cudaDeviceSynchronize(); }

    virtual void run()
    {
        thrust::exclusive_scan(d_a.begin(), d_a.end(), d_scan.begin());
    }

    virtual std::vector<T> get() const
    {
        std::vector<T> ans(d_scan.size());
        thrust::copy(d_scan.begin(), d_scan.end(), ans.begin());
        return ans;
    }
};

template<typename T>
scan_algorithm<T> *algorithm_factory<thrust_scan<T> >::create(device_type d, const std::vector<T> &h_a)
{
    return new thrust_scan<T>(d, h_a);
}

template<typename T>
std::string algorithm_factory<thrust_scan<T> >::name() { return thrust_scan<T>::name(); }

template<typename T>
std::string algorithm_factory<thrust_scan<T> >::api() { return thrust_scan<T>::api(); }

template struct algorithm_factory<thrust_scan<int> >;

/********************************************************************/

template<typename T>
class thrust_sort : public sort_algorithm<T>
{
private:
    thrust::device_vector<T> d_a;
    thrust::device_vector<T> d_target;

public:
    thrust_sort(device_type d, const std::vector<T> &h_a)
        : sort_algorithm<T>(h_a), d_a(h_a), d_target(h_a.size())
    {
        if (d != DEVICE_TYPE_GPU)
            throw device_not_supported();
    }

    static std::string name() { return "thrust::sort"; }
    static std::string api() { return "thrust"; }
    virtual void finish() { cudaDeviceSynchronize(); }

    virtual void run()
    {
        d_target = d_a;
        thrust::sort(d_target.begin(), d_target.end());
    }

    virtual std::vector<T> get() const
    {
        std::vector<T> ans(d_target.size());
        thrust::copy(d_target.begin(), d_target.end(), ans.begin());
        return ans;
    }
};

template<typename T>
sort_algorithm<T> *algorithm_factory<thrust_sort<T> >::create(device_type d, const std::vector<T> &h_a)
{
    return new thrust_sort<T>(d, h_a);
}

template<typename T>
std::string algorithm_factory<thrust_sort<T> >::name() { return thrust_sort<T>::name(); }

template<typename T>
std::string algorithm_factory<thrust_sort<T> >::api() { return thrust_sort<T>::api(); }

template struct algorithm_factory<thrust_sort<unsigned int> >;
