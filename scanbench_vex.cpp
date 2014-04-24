#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>
#undef CL_VERSION_1_2
#include <vexcl/vexcl.hpp>
#include <vexcl/external/clogs.hpp>
#include <stdexcept>
#include "scanbench_vex.h"

vex_algorithm::vex_algorithm() : ctx(new vex::Context(vex::Filter::Position(0)))
{
    if (!ctx)
        throw std::runtime_error("No device found for vex");
}

void vex_algorithm::finish()
{
    ctx->finish();
}

vex_algorithm::~vex_algorithm()
{
}

/************************************************************************/

template<typename T>
struct vex_scan<T>::data_t
{
    vex::vector<T> d_a, d_scan;

    data_t(vex::Context &ctx, const std::vector<T> &h_a)
        : d_a(ctx, h_a), d_scan(ctx, h_a.size())
    {
    }
};

template<typename T>
vex_scan<T>::vex_scan(const std::vector<T> &h_a)
    : data(new data_t(*ctx, h_a))
{
}

template<typename T>
void vex_scan<T>::run()
{
    vex::exclusive_scan(data->d_a, data->d_scan);
}

template<typename T>
vex_scan<T>::~vex_scan()
{
}

template<typename T>
void vex_clogs_scan<T>::run()
{
    vex::clogs::exclusive_scan(this->data->d_a, this->data->d_scan);
}

/************************************************************************/

template<typename T>
struct vex_sort<T>::data_t
{
    vex::vector<T> d_a;
    vex::vector<T> d_target;

    data_t(vex::Context &ctx, const std::vector<T> &h_a)
        : d_a(ctx, h_a), d_target(ctx, h_a.size())
    {
    }
};

template<typename T>
vex_sort<T>::vex_sort(const std::vector<T> &h_a)
    : data(new data_t(*ctx, h_a))
{
}

template<typename T>
void vex_sort<T>::run()
{
    data->d_target = data->d_a;
    vex::sort(data->d_target);
}

template<typename T>
vex_sort<T>::~vex_sort()
{
}
