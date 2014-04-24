#include <parallel/numeric>
#include <parallel/algorithm>
#include <numeric>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include <vexcl/vexcl.hpp>
#include <vexcl/external/clogs.hpp>
#include <clogs/scan.h>
#include <cstdint>
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

template<typename T>
class my_parallel_scan
{
private:
    std::vector<T> a;
    std::vector<T> out;
public:
    my_parallel_scan(const std::vector<T> &h_a) : a(h_a), out(h_a.size()) {}

    std::string name() const { return "my parallel scan"; }
    void run()
    {
        std::size_t threads;
#pragma omp parallel
        {
#pragma omp single
            {
                threads = omp_get_num_threads();
            }
        }

        const std::size_t chunk = 4 * 1024 * 1024 / sizeof(T);
        T reduced[threads];
        T carry{};
#pragma omp parallel
        {
            std::size_t tid = omp_get_thread_num();
            auto begin = a.begin();
            for (std::size_t start = 0; start < a.size(); start += chunk)
            {
                std::size_t len = std::min(a.size() - start, chunk);
                auto p = begin + (start + tid * len / threads);
                auto q = begin + (start + (tid + 1) * len / threads);
                reduced[tid] = accumulate(p, q, T());
#pragma omp barrier
#pragma omp single
                {
                    T sum = carry;
                    for (std::size_t i = 0; i < threads; i++)
                    {
                        T next = sum + reduced[i];
                        reduced[i] = sum;
                        sum = next;
                    }
                    carry = sum;
                }

                T sum = reduced[tid];
                for (auto i = p; i != q; ++i)
                {
                    T tmp = sum;
                    sum += *p;
                    *p = tmp;
                }
            }
        }
    }

    void finish() {}
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
    std::vector<std::int32_t> h_a(N);
    for (std::size_t i = 0; i < h_a.size(); i++)
        h_a[i] = i;

#if USE_COMPUTE
    time_algorithm(compute_scan<std::int32_t>(h_a), N, iter);
#endif
#if USE_VEX
    time_algorithm(vex_scan<std::int32_t>(h_a), N, iter);
    time_algorithm(vex_clogs_scan<std::int32_t>(h_a), N, iter);
#endif
#if USE_CLOGS
    time_algorithm(clogs_scan<std::int32_t>(h_a), N, iter);
#endif
#if USE_CUDA
    time_algorithm(thrust_scan<std::int32_t>(h_a), N, iter);
#endif
    time_algorithm(serial_scan<std::int32_t>(h_a), N, iter);
    time_algorithm(parallel_scan<std::int32_t>(h_a), N, iter);
    time_algorithm(my_parallel_scan<cl_int>(h_a), N, iter);

    std::cout << "\n";

    std::vector<std::uint32_t> rnd(N);
    for (std::size_t i = 0; i < rnd.size(); i++)
        rnd[i] = (std::uint32_t) i * 0x9E3779B9;
#if USE_VEX
    time_algorithm(vex_sort<std::uint32_t>(rnd), N, iter);
#endif
#if USE_CLOGS
    time_algorithm(clogs_sort<std::uint32_t>(rnd), N, iter);
#endif
#if USE_CUDA
    time_algorithm(thrust_sort<std::uint32_t>(rnd), N, iter);
#endif
    time_algorithm(serial_sort<std::uint32_t>(rnd), N, iter);
    time_algorithm(parallel_sort<std::uint32_t>(rnd), N, iter);
    return 0;
}
