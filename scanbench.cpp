#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <numeric>
#include "scanbench_thrust.h"
#include "scanbench_vex.h"
#include "scanbench_compute.h"
#include "scanbench_clogs.h"
#include "scanbench_cpu.h"

typedef std::chrono::high_resolution_clock clock_type;

template<typename T>
static bool check(std::size_t idx, const T &a, const T &b)
{
    if (a != b)
    {
        std::cerr << idx << ": expected " << a << " but found " << b << '\n';
        return false;
    }
    return true;
}

template<typename T>
class validate_scan
{
private:
    std::vector<T> expected;

public:
    validate_scan(const std::vector<T> &in)
        : expected(in.size())
    {
        std::partial_sum(in.begin(), in.end() - 1, expected.begin() + 1);
        expected[0] = T();
    }

    void operator()(const std::vector<T> &out) const
    {
        assert(out.size() == expected.size());
        for (std::size_t i = 0; i < expected.size(); i++)
            if (!check(i, expected[i], out[i]))
                break;
    }
};

template<typename T>
class validate_sort
{
private:
    std::vector<T> expected;

public:
    validate_sort(std::vector<T> in)
        : expected(std::move(in))
    {
        std::sort(expected.begin(), expected.end());
    }

    void operator()(const std::vector<T> &out) const
    {
        assert(out.size() == expected.size());
        for (std::size_t i = 0; i < expected.size(); i++)
            if (!check(i, expected[i], out[i]))
                break;
    }
};

template<typename T, typename V>
static void time_algorithm(T &&alg, V &&validate, size_t N, int iter)
{
    // Warmup
    alg.run();
    alg.finish();
    auto start = clock_type::now();
    for (int i = 0; i < iter; i++)
        alg.run();
    alg.finish();
    auto stop = clock_type::now();

    validate(alg.get());

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
    std::vector<std::uint32_t> rnd(N);
    for (std::size_t i = 0; i < rnd.size(); i++)
        rnd[i] = (std::uint32_t) i * 0x9E3779B9;

    validate_scan<std::int32_t> vscan(h_a);
    validate_sort<std::uint32_t> vsort(rnd);

#if USE_CPU
    time_algorithm(serial_scan<std::int32_t>(h_a), vscan, N, iter);
    time_algorithm(parallel_scan<std::int32_t>(h_a), vscan, N, iter);
    time_algorithm(my_parallel_scan<std::int32_t>(h_a), vscan, N, iter);
    time_algorithm(serial_sort<std::uint32_t>(rnd), vsort, N, iter);
    time_algorithm(parallel_sort<std::uint32_t>(rnd), vsort, N, iter);
#endif

    std::cout << "\n";

#if USE_COMPUTE
    time_algorithm(compute_scan<std::int32_t>(h_a), vscan, N, iter);
#endif
#if USE_VEX
    time_algorithm(vex_scan<std::int32_t>(h_a), vscan, N, iter);
    time_algorithm(vex_clogs_scan<std::int32_t>(h_a), vscan, N, iter);
#endif
#if USE_CLOGS
    time_algorithm(clogs_scan<std::int32_t>(h_a), vscan, N, iter);
#endif
#if USE_THRUST
    time_algorithm(thrust_scan<std::int32_t>(h_a), vscan, N, iter);
#endif

    std::cout << "\n";

#if USE_COMPUTE
    time_algorithm(compute_sort<std::uint32_t>(rnd), vsort, N, iter);
#endif
#if USE_VEX
    time_algorithm(vex_sort<std::uint32_t>(rnd), vsort, N, iter);
#endif
#if USE_CLOGS
    time_algorithm(clogs_sort<std::uint32_t>(rnd), vsort, N, iter);
#endif
#if USE_THRUST
    time_algorithm(thrust_sort<std::uint32_t>(rnd), vsort, N, iter);
#endif
    return 0;
}
