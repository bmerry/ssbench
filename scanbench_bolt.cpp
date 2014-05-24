#include <bolt/cl/scan.h>
#include <bolt/cl/sort.h>
#include <bolt/cl/copy.h>
#include <bolt/cl/device_vector.h>
#include <stdexcept>
#include "scanbench_algorithms.h"
#include "scanbench_register.h"

class bolt_algorithm
{
protected:
    bolt::cl::control control;

    bolt_algorithm() : control()
    {
        control.setUseHost(bolt::cl::control::NoUseHost);
        control.setForceRunMode(bolt::cl::control::OpenCL);
    }
};

/************************************************************************/

template<typename T>
class bolt_scan : public scan_algorithm<T>, public bolt_algorithm
{
protected:
    bolt::cl::device_vector<T> d_a, d_scan;

public:
    bolt_scan(const std::vector<T> &h_a)
        : scan_algorithm<T>(h_a), d_a(h_a.begin(), h_a.end(), CL_MEM_READ_WRITE, control),
        d_scan(h_a.size(), T(), CL_MEM_READ_WRITE, false, control)
    {
    }

    static std::string name() { return "bolt::cl::exclusive_scan"; }
    static std::string api() { return "bolt"; }
    virtual void finish() override { control.getCommandQueue().finish(); }

    virtual void run() override
    {
        bolt::cl::exclusive_scan(control, d_a.begin(), d_a.end(), d_scan.begin());
    }

    virtual std::vector<T> get() const override
    {
        auto ptr = d_scan.data();
        return std::vector<T>(ptr.get(), ptr.get() + d_scan.size());
    }
};

static register_scan_algorithm<bolt_scan> register_bolt_scan;

/************************************************************************/

template<typename T>
class bolt_sort : public sort_algorithm<T>, public bolt_algorithm
{
protected:
    bolt::cl::device_vector<T> d_a, d_target;

public:
    bolt_sort(const std::vector<T> &h_a)
        : sort_algorithm<T>(h_a), d_a(h_a.begin(), h_a.end(), CL_MEM_READ_WRITE, control),
        d_target(h_a.size(), T(), CL_MEM_READ_WRITE, false, control)
    {
    }

    static std::string name() { return "bolt::cl::sort"; }
    static std::string api() { return "bolt"; }
    virtual void finish() override { control.getCommandQueue().finish(); }

    virtual void run() override
    {
        bolt::cl::copy(control, d_a.begin(), d_a.end(), d_target.begin());
        bolt::cl::sort(control, d_target.begin(), d_target.end());
    }

    virtual std::vector<T> get() const override
    {
        auto ptr = d_target.data();
        return std::vector<T>(ptr.get(), ptr.get() + d_target.size());
    }
};

static register_sort_algorithm<bolt_sort> register_bolt_sort;
