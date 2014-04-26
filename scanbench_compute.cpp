#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>
#undef CL_VERSION_1_2
#include <boost/compute/container/vector.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/exclusive_scan.hpp>
#include <boost/compute/algorithm/sort.hpp>
#include "scanbench_algorithms.h"
#include "scanbench_register.h"

namespace compute = boost::compute;

class compute_algorithm
{
protected:
    mutable boost::compute::command_queue queue;

    compute_algorithm()
    : queue(compute::system::default_queue()) {}
};

/************************************************************************/

template<typename T>
class compute_scan : public scan_algorithm<T>, compute_algorithm
{
private:
    compute::vector<T> d_a;
    compute::vector<T> d_scan;

public:
    compute_scan(const std::vector<T> &h_a)
        : scan_algorithm<T>(h_a), d_a(h_a), d_scan(h_a.size())
    {
    }

    virtual std::string name() const override { return "compute::exclusive_scan"; }
    virtual std::string api() const override { return "compute"; }
    virtual void finish() override { queue.finish(); }

    virtual void run() override
    {
        compute::exclusive_scan(d_a.begin(), d_a.end(), d_scan.begin());
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
    compute_sort(const std::vector<T> &h_a)
        : sort_algorithm<T>(h_a), d_a(h_a), d_target(h_a.size())
    {
    }

    virtual std::string name() const override { return "compute::sort"; }
    virtual std::string api() const override { return "compute"; }
    virtual void finish() override { queue.finish(); }

    virtual void run() override
    {
        d_target = d_a;
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
