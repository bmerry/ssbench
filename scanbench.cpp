#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>
#undef CL_VERSION_1_2
#include <boost/compute/container/vector.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/exclusive_scan.hpp>
#include <parallel/numeric>
#include <parallel/algorithm>
#include <numeric>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include "scanbench_cuda.h"
#include "scanbench_vex.h"
#include "scanbench_compute.h"
#include "scanbench_clogs.h"

typedef std::chrono::high_resolution_clock clock_type;

/************************************************************************/

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
class serial_sort
{
protected:
    std::vector<T> d_a;
    std::vector<T> d_target;

public:
    serial_sort(const std::vector<T> &h_a)
        : d_a(h_a)
    {
    }

    std::string name() const { return "serial sort"; }

    void run()
    {
        d_target = d_a;
        std::sort(d_target.begin(), d_target.end());
    }

    void finish() {}
};

template<typename T>
class parallel_sort : public serial_sort<T>
{
public:
    using serial_sort<T>::serial_sort;

    std::string name() const { return "parallel sort"; }

    void run()
    {
        this->d_target = this->d_a;
        __gnu_parallel::sort(this->d_target.begin(), this->d_target.end());
    }
};

/************************************************************************/

template<typename T>
static void time_algorithm(T &&alg, size_t N, int iter)
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
    double time = elapsed.count();
    double rate = (double) N * iter / time / 1e6;
    std::cout << std::setw(20) << std::fixed << std::setprecision(1);
    std::cout << rate << " M/s\t";
    std::cout << std::setw(0) << std::setprecision(6);
    std::cout << time << "\t" << alg.name() << '\n';
}

int main()
{
    const int iter = 16;
    const int N = 16 * 1024 * 1024;
    std::vector<cl_int> h_a(N);
    for (std::size_t i = 0; i < h_a.size(); i++)
        h_a[i] = i;

    time_algorithm(compute_scan<cl_int>(h_a), N, iter);
    time_algorithm(vex_scan<cl_int>(h_a), N, iter);
    time_algorithm(vex_clogs_scan<cl_int>(h_a), N, iter);
    time_algorithm(clogs_scan<cl_int>(h_a), N, iter);
    time_algorithm(thrust_scan<cl_int>(h_a), N, iter);
    time_algorithm(serial_scan<cl_int>(h_a), N, iter);
    time_algorithm(parallel_scan<cl_int>(h_a), N, iter);

    std::cout << "\n";

    std::vector<cl_uint> rnd(N);
    for (std::size_t i = 0; i < rnd.size(); i++)
        rnd[i] = (cl_uint) i * 0x9E3779B9;
    time_algorithm(vex_sort<cl_uint>(rnd), N, iter);
    time_algorithm(clogs_sort<cl_uint>(rnd), N, iter);
    time_algorithm(thrust_sort<cl_uint>(rnd), N, iter);
    time_algorithm(serial_sort<cl_uint>(rnd), N, iter);
    time_algorithm(parallel_sort<cl_uint>(rnd), N, iter);
    return 0;
}
