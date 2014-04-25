#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>
#undef CL_VERSION_1_2
#include <boost/compute/container/vector.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/exclusive_scan.hpp>
#include <boost/compute/algorithm/sort.hpp>
#include "scanbench_compute.h"

namespace compute = boost::compute;

compute_algorithm::compute_algorithm()
    : queue(new compute::command_queue(compute::system::default_queue()))
{
}

compute_algorithm::~compute_algorithm()
{
}

void compute_algorithm::finish()
{
    queue->finish();
}

/************************************************************************/

template<typename T>
struct compute_scan<T>::data_t
{
    compute::vector<T> d_a;
    compute::vector<T> d_scan;

    data_t(const std::vector<T> &h_a)
        : d_a(h_a), d_scan(h_a.size())
    {
    }
};

template<typename T>
compute_scan<T>::compute_scan(const std::vector<T> &h_a)
    : data(new data_t(h_a))
{
}

template<typename T>
compute_scan<T>::~compute_scan()
{
}

template<typename T>
void compute_scan<T>::run()
{
    compute::exclusive_scan(data->d_a.begin(), data->d_a.end(), data->d_scan.begin());
}

template<typename T>
std::vector<T> compute_scan<T>::get() const
{
    std::vector<T> ans(data->d_scan.size());
    compute::copy(data->d_scan.begin(), data->d_scan.end(), ans.begin(), *queue);
    return ans;
}

template class compute_scan<cl_int>;

/************************************************************************/

template<typename T>
struct compute_sort<T>::data_t
{
    compute::vector<T> d_a;
    compute::vector<T> d_target;

    data_t(const std::vector<T> &h_a)
        : d_a(h_a), d_target(h_a.size())
    {
    }
};

template<typename T>
compute_sort<T>::compute_sort(const std::vector<T> &h_a)
    : data(new data_t(h_a))
{
}

template<typename T>
compute_sort<T>::~compute_sort()
{
}

template<typename T>
void compute_sort<T>::run()
{
    data->d_target = data->d_a;
    compute::sort(data->d_target.begin(), data->d_target.end());
}

template<typename T>
std::vector<T> compute_sort<T>::get() const
{
    std::vector<T> ans(data->d_target.size());
    compute::copy(data->d_target.begin(), data->d_target.end(), ans.begin(), *queue);
    return ans;
}

template class compute_sort<cl_uint>;
