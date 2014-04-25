#include "scanbench_cuda.h"
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <vector>

template<typename T>
struct thrust_scan<T>::data_t
{
    thrust::device_vector<T> d_a;
    thrust::device_vector<T> d_scan;

    data_t(const std::vector<T> &h_a)
        : d_a(h_a), d_scan(h_a.size())
    {
    }
};

template<typename T>
thrust_scan<T>::thrust_scan(const std::vector<T> &h_a)
    : data(new data_t(h_a))
{
}

template<typename T>
void thrust_scan<T>::run()
{
    thrust::exclusive_scan(data->d_a.begin(), data->d_a.end(), data->d_scan.begin());
}

template<typename T>
std::vector<T> thrust_scan<T>::get() const
{
    std::vector<T> ans(data->d_scan.size());
    thrust::copy(data->d_scan.begin(), data->d_scan.end(), ans.begin());
    return ans;
}

template<typename T>
void thrust_scan<T>::finish()
{
    cudaDeviceSynchronize();
}

template<typename T>
thrust_scan<T>::~thrust_scan()
{
    delete data;
}

/********************************************************************/

template<typename T>
struct thrust_sort<T>::data_t
{
    thrust::device_vector<T> d_a;
    thrust::device_vector<T> d_target;

    data_t(const std::vector<T> &h_a)
        : d_a(h_a), d_target(d_a.size())
    {
    }
};

template<typename T>
thrust_sort<T>::thrust_sort(const std::vector<T> &h_a)
    : data(new data_t(h_a))
{
}

template<typename T>
void thrust_sort<T>::run()
{
    data->d_target = data->d_a;
    thrust::sort(data->d_target.begin(), data->d_target.end());
}

template<typename T>
std::vector<T> thrust_sort<T>::get() const
{
    std::vector<T> ans(data->d_target.size());
    thrust::copy(data->d_target.begin(), data->d_target.end(), ans.begin());
    return ans;
}

template<typename T>
void thrust_sort<T>::finish()
{
    cudaDeviceSynchronize();
}

template<typename T>
thrust_sort<T>::~thrust_sort()
{
    delete data;
}

template class thrust_scan<int>;
template class thrust_sort<unsigned int>;
