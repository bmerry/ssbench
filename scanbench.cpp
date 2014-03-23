#include <boost/compute/container/vector.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/exclusive_scan.hpp>
#include <parallel/numeric>
#include <numeric>
#include <vector>
#include <iostream>
#include <chrono>
#include <vexcl/vexcl.hpp>
#include <vexcl/external/clogs.hpp>
#include <clogs/scan.h>

typedef std::chrono::high_resolution_clock clock_type;
namespace compute = boost::compute;

class vex_algorithm
{
protected:
    vex::Context ctx;

    vex_algorithm() : ctx(vex::Filter::Type(CL_DEVICE_TYPE_GPU))
    {
        if (!ctx)
            throw std::runtime_error("No device found for vex");
    }

public:
    void finish() { ctx.finish(); }
};

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

public:
    void finish() { queue.finish(); }
};

/************************************************************************/

template<typename T>
class compute_scan
{
private:
    compute::command_queue queue;
    compute::vector<T> d_a;
    compute::vector<T> d_scan;
public:
    compute_scan(const std::vector<T> &h_a)
        : d_a(h_a), d_scan(h_a.size())
    {
        compute::device device = compute::system::default_device();
        queue = compute::system::default_queue();
    }

    std::string name() const { return "compute::exclusive_scan"; }
    void run() { compute::exclusive_scan(d_a.begin(), d_a.end(), d_scan.begin()); }
    void finish() { queue.finish(); }
};

template<typename T>
class vex_scan : public vex_algorithm
{
protected:
    vex::vector<T> d_a, d_scan;

public:
    vex_scan(const std::vector<T> &h_a)
        : d_a(vex::vector<T>(ctx, h_a)),
        d_scan(vex::vector<T>(ctx, h_a.size())) {}

    std::string name() const { return "vex::exclusive_scan"; }
    void run() { vex::exclusive_scan(d_a, d_scan); }
};

template<typename T>
class vex_clogs_scan : public vex_scan<T>
{
public:
    vex_clogs_scan(const std::vector<T> &h_a) : vex_scan<T>(h_a) {}

    std::string name() const { return "vex::clogs::exclusive_scan"; }
    void run() { vex::clogs::exclusive_scan(this->d_a, this->d_scan); }
};

template<typename T>
class clogs_scan : public clogs_algorithm
{
private:
    std::size_t elements;
    cl::Buffer d_a;
    cl::Buffer d_scan;
    clogs::Scan scan;

public:
    clogs_scan(const std::vector<T> &h_a)
        : elements(h_a.size()),
        scan(ctx, device, vex::clogs::clogs_type<T>::type())
    {
        std::vector<T> &h_a_nc = const_cast<std::vector<T> &>(h_a);
        cl::Context ctx = cl::Context::getDefault();

        d_a = cl::Buffer(h_a_nc.begin(), h_a_nc.end(), false);
        d_scan = cl::Buffer(ctx, CL_MEM_READ_WRITE, elements * sizeof(T));
    }

    std::string name() const { return "clogs::Scan"; }
    void run() { scan.enqueue(queue, d_a, d_scan, elements); }
};

template<typename T>
class serial_scan
{
private:
    std::vector<T> a;
    std::vector<T> out;
public:
    serial_scan(const std::vector<T> &h_a) : a(h_a), out(h_a.size()) {}

    std::string name() const { return "serial scan"; }
    void run() { std::partial_sum(a.begin(), a.end(), out.begin()); }
    void finish() {};
};

template<typename T>
class parallel_scan
{
private:
    std::vector<T> a;
    std::vector<T> out;
public:
    parallel_scan(const std::vector<T> &h_a) : a(h_a), out(h_a.size()) {}

    std::string name() const { return "parallel scan"; }
    void run() { __gnu_parallel::partial_sum(a.begin(), a.end(), out.begin()); }
    void finish() {};
};

/************************************************************************/

template<typename T>
class vex_sort : public vex_algorithm
{
private:
    vex::vector<T> d_a;
    vex::vector<T> d_target;

public:
    vex_sort(const std::vector<T> &h_a) : d_a(ctx, h_a), d_target(ctx, h_a.size()) {}

    std::string name() const { return "vex::sort"; }
    void run() { d_target = d_a; vex::sort(d_target); }
};

template<typename T>
class clogs_sort : public clogs_algorithm
{
private:
    std::size_t elements;
    cl::Buffer d_a;
    cl::Buffer d_target;
    clogs::Radixsort sort;

public:
    clogs_sort(const std::vector<T> &h_a)
        : elements(h_a.size()),
        d_a(ctx, CL_MEM_READ_WRITE, elements * sizeof(T)),
        d_target(ctx, CL_MEM_READ_WRITE, elements * sizeof(T)),
        sort(ctx, device, vex::clogs::clogs_type<T>::type())
    {
        queue.enqueueWriteBuffer(d_a, CL_TRUE, 0, elements * sizeof(T), h_a.data());
    }

    std::string name() const { return "clogs::Radixsort"; }

    void run()
    {
        queue.enqueueCopyBuffer(d_a, d_target, 0, 0, elements * sizeof(T));
        sort.enqueue(queue, d_target, cl::Buffer(), elements);
    }
};

/************************************************************************/

template<typename T>
static void time_algorithm(T &&alg, int iter)
{
    // Warmup
    alg.run();
    alg.finish();
    auto start = clock_type::now();
    for (int i = 0; i < iter; i++)
        alg.run();
    alg.finish();
    auto stop = clock_type::now();

    std::chrono::duration<double> elapsed(stop - start);
    std::cout << alg.name() << ": " << elapsed.count() << '\n';
}

int main()
{
    const int iter = 10;
    std::vector<cl_int> h_a(2000000);
    for (std::size_t i = 0; i < h_a.size(); i++)
        h_a[i] = i;

    time_algorithm(compute_scan<cl_int>(h_a), iter);
    time_algorithm(vex_scan<cl_int>(h_a), iter);
    time_algorithm(vex_clogs_scan<cl_int>(h_a), iter);
    time_algorithm(clogs_scan<cl_int>(h_a), iter);
    time_algorithm(serial_scan<cl_int>(h_a), iter);
    time_algorithm(parallel_scan<cl_int>(h_a), iter);

    std::vector<cl_uint> rnd(20000000);
    for (std::size_t i = 0; i < rnd.size(); i++)
        rnd[i] = (cl_uint) i * 0x9E3779B9;
    time_algorithm(vex_sort<cl_uint>(rnd), iter);
    time_algorithm(clogs_sort<cl_uint>(rnd), iter);
    return 0;
}
