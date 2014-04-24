#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>
#undef CL_VERSION_1_2
#include <boost/compute/container/vector.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/exclusive_scan.hpp>
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

template class compute_scan<cl_int>;
