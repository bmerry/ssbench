#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>
#undef CL_VERSION_1_2
#include <vexcl/vexcl.hpp>
#include <vexcl/external/clogs.hpp>
#include <stdexcept>
#include "scanbench_algorithms.h"
#include "scanbench_register.h"

class vex_algorithm
{
protected:
    vex::Context ctx;

    vex_algorithm() : ctx(vex::Filter::Position(0))
    {
        if (!ctx)
            throw std::runtime_error("No device found for vex");
    }
};

/************************************************************************/

template<typename T>
class vex_scan : public scan_algorithm<T>, public vex_algorithm
{
protected:
    vex::vector<T> d_a, d_scan;

public:
    vex_scan(const std::vector<T> &h_a)
        : scan_algorithm<T>(h_a), d_a(ctx, h_a), d_scan(ctx, h_a.size())
    {
    }

    static std::string name() { return "vex::exclusive_scan"; }
    static std::string api() { return "vex"; }
    virtual void finish() override { ctx.finish(); }

    virtual void run() override
    {
        vex::exclusive_scan(d_a, d_scan);
    }

    virtual std::vector<T> get() const override
    {
        std::vector<T> ans(d_scan.size());
        vex::copy(d_scan, ans);
        return ans;
    }
};


template<typename T>
class vex_clogs_scan : public vex_scan<T>
{
public:
    using vex_scan<T>::vex_scan;

    static std::string name() { return "vex::clogs::exclusive_scan"; }
    static std::string api() { return "vex"; }
    virtual void finish() override { this->ctx.finish(); }

    virtual void run() override
    {
        vex::clogs::exclusive_scan(this->d_a, this->d_scan);
    }
};

static register_scan_algorithm<vex_scan> register_vex_scan;
static register_scan_algorithm<vex_clogs_scan> register_vex_clogs_scan;

/************************************************************************/

template<typename T>
class vex_sort : public sort_algorithm<T>, public vex_algorithm
{
protected:
    vex::vector<T> d_a;
    vex::vector<T> d_target;

public:
    vex_sort(const std::vector<T> &h_a)
        : sort_algorithm<T>(h_a), d_a(ctx, h_a), d_target(ctx, h_a.size())
    {
    }

    static std::string name() { return "vex::sort"; }
    static std::string api() { return "vex"; }
    virtual void finish() override { ctx.finish(); }

    virtual void run() override
    {
        d_target = d_a;
        vex::sort(d_target);
    }

    virtual std::vector<T> get() const override
    {
        std::vector<T> ans(d_target.size());
        vex::copy(d_target, ans);
        return ans;
    }
};

static register_sort_algorithm<vex_sort> register_vex_sort;
