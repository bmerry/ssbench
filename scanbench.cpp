#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <boost/program_options.hpp>
#include "scanbench_thrust.h"
#include "scanbench_vex.h"
#include "scanbench_compute.h"
#include "scanbench_clogs.h"
#include "scanbench.h"

namespace po = boost::program_options;

typedef std::chrono::high_resolution_clock clock_type;

static void time_algorithm(algorithm &alg, size_t N, int iter)
{
    // Warmup
    alg.run();
    alg.finish();

    // Real run
    auto start = clock_type::now();
    for (int i = 0; i < iter; i++)
        alg.run();
    alg.finish();
    auto stop = clock_type::now();

    alg.validate();

    std::chrono::duration<double> elapsed(stop - start);
    double time = elapsed.count();
    double rate = (double) N * iter / time / 1e6;
    std::cout << std::setw(20) << std::fixed << std::setprecision(1);
    std::cout << rate << " M/s\t";
    std::cout << std::setw(0) << std::setprecision(6);
    std::cout << time << "\t" << alg.name() << '\n';
}

static void usage(std::ostream &o, const po::options_description &opts)
{
    o << "Usage: scanbench [options]\n\n";
    o << opts;
}

static po::variables_map processOptions(int argc, char **argv)
{
    po::options_description opts;

    opts.add_options()
        ("help,h",        "show usage")
        ("items,N",       po::value<int>()->default_value(16777216), "Problem size")
        ("iterations,R",  po::value<int>()->default_value(16), "Number of repetitions")
#if USE_CPU
        ("no-cpu",        "disable CPU algorithms")
#endif
#if USE_VEX
        ("no-vex",        "disable VexCL algorithms")
#endif
#if USE_COMPUTE
        ("no-compute",    "disable boost::compute algorithms")
#endif
#if USE_CLOGS
        ("no-clogs",      "disable CLOGS algorithms")
#endif
#if USE_THRUST
        ("no-thrust",     "disable Thrust algorithms")
#endif
        ("no-sort",       "disable all sorting algorithms")
        ("no-scan",       "disable all scan algorithms");

    try
    {
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv)
                  .style(po::command_line_style::default_style & ~po::command_line_style::allow_guessing)
                  .options(opts)
                  .run(), vm);
        po::notify(vm);

        if (vm.count("help"))
        {
            usage(std::cout, opts);
            std::exit(0);
        }

        return vm;
    }
    catch (po::error &e)
    {
        std::cerr << e.what() << "\n\n";
        usage(std::cerr, opts);
        std::exit(1);
    }
}

int main(int argc, char **argv)
{
    po::variables_map vm = processOptions(argc, argv);
    const int iterations = vm["iterations"].as<int>();
    const int items = vm["items"].as<int>();

    std::vector<std::int32_t> h_a(items);
    for (std::size_t i = 0; i < h_a.size(); i++)
        h_a[i] = i;
    std::vector<std::uint32_t> rnd(items);
    for (std::size_t i = 0; i < rnd.size(); i++)
        rnd[i] = (std::uint32_t) i * 0x9E3779B9;

    for (const auto &factory : scan_registry<std::int32_t>::get())
        time_algorithm(*factory(h_a), items, iterations);

    std::cout << "\n";

#if USE_COMPUTE
    time_algorithm(compute_scan<std::int32_t>(h_a), vscan, items, iterations);
#endif
#if USE_VEX
    time_algorithm(vex_scan<std::int32_t>(h_a), vscan, items, iterations);
    time_algorithm(vex_clogs_scan<std::int32_t>(h_a), vscan, items, iterations);
#endif
#if USE_CLOGS
    time_algorithm(clogs_scan<std::int32_t>(h_a), vscan, items, iterations);
#endif
#if USE_THRUST
    time_algorithm(thrust_scan<std::int32_t>(h_a), vscan, items, iterations);
#endif

    std::cout << "\n";

#if USE_COMPUTE
    time_algorithm(compute_sort<std::uint32_t>(rnd), vsort, items, iterations);
#endif
#if USE_VEX
    time_algorithm(vex_sort<std::uint32_t>(rnd), vsort, items, iterations);
#endif
#if USE_CLOGS
    time_algorithm(clogs_sort<std::uint32_t>(rnd), vsort, items, iterations);
#endif
#if USE_THRUST
    time_algorithm(thrust_sort<std::uint32_t>(rnd), vsort, items, iterations);
#endif
    return 0;
}
