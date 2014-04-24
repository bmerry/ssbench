#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>
#undef CL_VERSION_1_2
#include <clogs/scan.h>
#include <vexcl/external/clogs.hpp> // TODO: eliminate: needed only for introspection
#include "scanbench_clogs.h"

struct clogs_algorithm::resources_t
{
    cl::Context ctx;
    cl::Device device;
    cl::CommandQueue queue;

    resources_t()
        : ctx(cl::Context::getDefault()),
        device(cl::Device::getDefault()),
        queue(cl::CommandQueue::getDefault())
    {
    }
};

clogs_algorithm::clogs_algorithm()
    : resources(new resources_t())
{
}

clogs_algorithm::~clogs_algorithm()
{
}

void clogs_algorithm::finish()
{
    resources->queue.finish();
}

/************************************************************************/

template<typename T>
struct clogs_scan<T>::data_t
{
    std::size_t elements;
    cl::Buffer d_a;
    cl::Buffer d_scan;
    clogs::Scan scan;

    data_t(const cl::Context &ctx, const cl::Device &device, const std::vector<T> &h_a)
        : elements(h_a.size()),
        d_a(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            h_a.size() * sizeof(T),
            const_cast<T *>(h_a.data())),
        d_scan(ctx, CL_MEM_READ_WRITE, elements * sizeof(T)),
        scan(ctx, device, vex::clogs::clogs_type<T>::type())
    {
    }
};

template<typename T>
clogs_scan<T>::clogs_scan(const std::vector<T> &h_a)
    : data(new data_t(resources->ctx, resources->device, h_a))
{
}

template<typename T>
clogs_scan<T>::~clogs_scan()
{
}

template<typename T>
void clogs_scan<T>::run()
{
    data->scan.enqueue(resources->queue, data->d_a, data->d_scan, data->elements);
}

template class clogs_scan<cl_int>;

/************************************************************************/

template<typename T>
struct clogs_sort<T>::data_t
{
    std::size_t elements;
    cl::Buffer d_a;
    cl::Buffer d_target;
    clogs::Radixsort sort;

    data_t(const cl::Context &ctx, const cl::Device &device, const std::vector<T> &h_a)
        : d_a(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, elements * sizeof(T),
              const_cast<T *>(h_a.data())),
        d_target(ctx, CL_MEM_READ_WRITE, elements * sizeof(T)),
        sort(ctx, device, vex::clogs::clogs_type<T>::type())
    {
    }
};

template<typename T>
clogs_sort<T>::clogs_sort(const std::vector<T> &h_a)
    : data(new data_t(resources->ctx, resources->device, h_a))
{
}

template<typename T>
clogs_sort<T>::~clogs_sort()
{
}

template<typename T>
void clogs_sort<T>::run()
{
    resources->queue.enqueueCopyBuffer(data->d_a, data->d_target, 0, 0, data->elements * sizeof(T));
    data->sort.enqueue(resources->queue, data->d_target, cl::Buffer(), data->elements);
}

template class clogs_sort<cl_uint>;
