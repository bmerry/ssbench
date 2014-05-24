#include <bolt/cl/scan.h>
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
        std::vector<T> ans(d_scan.size());
        //bolt::cl::copy(control, d_scan.begin(), d_scan.end(), ans.begin());
        return ans;
    }
};

static register_scan_algorithm<bolt_scan> register_bolt_scan;
