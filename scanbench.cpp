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
#include "scanbench_cpu.h"

typedef std::chrono::high_resolution_clock clock_type;

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
#if USE_CPU
    time_algorithm(serial_scan<std::int32_t>(h_a), N, iter);
    time_algorithm(parallel_scan<std::int32_t>(h_a), N, iter);
    time_algorithm(my_parallel_scan<cl_int>(h_a), N, iter);
#endif

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
#if USE_CPU
    time_algorithm(serial_sort<std::uint32_t>(rnd), N, iter);
    time_algorithm(parallel_sort<std::uint32_t>(rnd), N, iter);
#endif
    return 0;
}
