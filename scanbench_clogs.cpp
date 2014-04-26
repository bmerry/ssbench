#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>
#undef CL_VERSION_1_2
#include <clogs/scan.h>
#include <clogs/radixsort.h>
#include "scanbench_algorithms.h"
#include "scanbench_register.h"

template<typename T>
struct clogs_type
{
};

template<>
struct clogs_type<cl_int>
{
    static clogs::Type type() { return clogs::TYPE_INT; }
};

template<>
struct clogs_type<cl_uint>
{
    static clogs::Type type() { return clogs::TYPE_UINT; }
};

/************************************************************************/

class clogs_algorithm
{
protected:
    cl::Context ctx;
    cl::Device device;
    cl::CommandQueue queue;

    clogs_algorithm()
        : ctx(cl::Context::getDefault()),
        device(cl::Device::getDefault()),
        queue(cl::CommandQueue::getDefault())
    {
    }
};

/************************************************************************/

template<typename T>
class clogs_scan : public scan_algorithm<T>, public clogs_algorithm
{
private:
    std::size_t elements;
    cl::Buffer d_a;
    cl::Buffer d_scan;
    clogs::Scan scan;

public:
    clogs_scan(const std::vector<T> &h_a)
        : scan_algorithm<T>(h_a),
        elements(h_a.size()),
        d_a(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            h_a.size() * sizeof(T),
            const_cast<T *>(h_a.data())),
        d_scan(ctx, CL_MEM_READ_WRITE, elements * sizeof(T)),
        scan(ctx, device, clogs_type<T>::type())
    {
    }

    virtual std::string name() const override { return "clogs::Scan"; }
    virtual std::string api() const override { return "clogs"; }
    virtual void finish() override { queue.finish(); }

    virtual void run() override
    {
        scan.enqueue(queue, d_a, d_scan, elements);
    }

    virtual std::vector<T> get() const override
    {
        std::vector<T> ans(elements);
        cl::copy(const_cast<cl::Buffer &>(d_scan), ans.begin(), ans.end());
        return ans;
    }
};

static register_scan_algorithm<clogs_scan> register_clogs_scan;

/************************************************************************/

template<typename T>
class clogs_sort : public sort_algorithm<T>, public clogs_algorithm
{
private:
    std::size_t elements;
    cl::Buffer d_a;
    cl::Buffer d_target;
    clogs::Radixsort sort;

public:
    clogs_sort(const std::vector<T> &h_a)
        : sort_algorithm<T>(h_a),
        elements(h_a.size()),
        d_a(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, elements * sizeof(T),
              const_cast<T *>(h_a.data())),
        d_target(ctx, CL_MEM_READ_WRITE, h_a.size() * sizeof(T)),
        sort(ctx, device, clogs_type<T>::type())
    {
    }

    virtual std::string name() const override { return "clogs::Radixsort"; }
    virtual std::string api() const override { return "clogs"; }
    virtual void finish() override { queue.finish(); }

    virtual void run() override
    {
        queue.enqueueCopyBuffer(d_a, d_target, 0, 0, elements * sizeof(T));
        sort.enqueue(queue, d_target, cl::Buffer(), elements);
    }

    virtual std::vector<T> get() const override
    {
        std::vector<T> ans(elements);
        cl::copy(const_cast<cl::Buffer &>(d_target), ans.begin(), ans.end());
        return ans;
    }
};

static register_sort_algorithm<clogs_sort> register_clogs_sort;
