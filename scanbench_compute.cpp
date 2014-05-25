#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>
#undef CL_VERSION_1_2
#include <boost/compute/container/vector.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/exclusive_scan.hpp>
#include <boost/compute/algorithm/sort.hpp>
#include "scanbench_algorithms.h"
#include "scanbench_register.h"
#include "clutils.h"

namespace compute = boost::compute;

class compute_algorithm
{
private:
    static boost::compute::device getDevice(device_type d)
    {
        for (const auto &device : boost::compute::system::devices())
        {
            if (device.type() == type_to_cl_type(d))
                return device;
        }
        throw device_not_supported();
    }

protected:
    boost::compute::device device;
    boost::compute::context ctx;
    mutable boost::compute::command_queue queue;

    explicit compute_algorithm(device_type d)
    : device(getDevice(d)),
    ctx(device),
    queue(ctx, device)
    {
    }
};

/************************************************************************/

template<typename T>
class compute_scan : public scan_algorithm<T>, compute_algorithm
{
private:
    compute::vector<T> d_a;
    compute::vector<T> d_scan;

public:
    compute_scan(device_type d, const std::vector<T> &h_a)
        : scan_algorithm<T>(h_a), compute_algorithm(d), d_a(h_a, queue), d_scan(h_a.size(), ctx)
    {
    }

    static std::string name() { return "compute::exclusive_scan"; }
    static std::string api() { return "compute"; }
    virtual void finish() override { queue.finish(); }

    virtual void run() override
    {
        compute::exclusive_scan(d_a.begin(), d_a.end(), d_scan.begin(), queue);
    }

    std::vector<T> get() const override
    {
        std::vector<T> ans(d_scan.size());
        compute::copy(d_scan.begin(), d_scan.end(), ans.begin(), queue);
        return ans;
    }
};

static register_scan_algorithm<compute_scan> register_compute_scan;

/************************************************************************/

template<typename T>
class compute_sort : public sort_algorithm<T>, public compute_algorithm
{
private:
    compute::vector<T> d_a;
    compute::vector<T> d_target;

public:
    compute_sort(device_type d, const std::vector<T> &h_a)
        : sort_algorithm<T>(h_a), compute_algorithm(d),
        d_a(h_a, queue), d_target(h_a.size(), ctx)
    {
    }

    static std::string name() { return "compute::sort"; }
    static std::string api() { return "compute"; }
    virtual void finish() override { queue.finish(); }

    virtual void run() override
    {
        compute::copy_async(d_a.begin(), d_a.end(), d_target.begin(), queue);
        compute::sort(d_target.begin(), d_target.end(), queue);
    }

    virtual std::vector<T> get() const override
    {
        std::vector<T> ans(d_target.size());
        compute::copy(d_target.begin(), d_target.end(), ans.begin(), queue);
        return ans;
    }
};

static register_sort_algorithm<compute_sort> register_compute_sort;
