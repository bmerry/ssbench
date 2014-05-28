#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <set>
#include <boost/program_options.hpp>
#include "scanbench_algorithms.h"
#include "scanbench_register.h"

namespace po = boost::program_options;

typedef std::chrono::high_resolution_clock clock_type;

static bool csv = false;

static void output_header(const std::string &name)
{
    if (!csv)
    {
        std::cout << name << "\n\n";
    }
}

static void output_result(
    const std::string &algname, const std::string &api, const std::string &name,
    double time, std::size_t N, int iter)
{
    double rate = (double) N * iter / time;
    if (csv)
    {
        std::cout << algname << "," << api << "," << N << "," << iter << "," << time << "," << rate << '\n';
    }
    else
    {
        std::cout << std::setw(20) << std::fixed << std::setprecision(1);
        std::cout << rate * 1e-6 << " M/s\t";
        std::cout << std::setw(0) << std::setprecision(6);
        std::cout << time << "\t" << name << '\n';
    }
}

static void output_footer()
{
    if (!csv)
        std::cout << '\n';
}

static void time_algorithm(
    algorithm &alg,
    const std::string &algname, const std::string &api, const std::string &name,
    std::size_t N, int iter)
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
    output_result(algname, api, name, time, N, iter);
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
        ("csv",           "output results in CSV format")
        ("cpu",           "run algorithms on the CPU")
        ("no-sort",       "disable all sorting algorithms")
        ("no-scan",       "disable all scan algorithms");

    std::set<std::string> apis;
    for (const auto &entry : scan_registry<std::int32_t>::get())
        apis.insert(entry.api);
    for (const auto &entry : sort_registry<std::uint32_t, std::uint32_t>::get())
        apis.insert(entry.api);

    for (const std::string &api : apis)
    {
        std::string optname = "no-" + api;
        std::string optdesc = "disable " + api + " algorithms";
        opts.add_options()(optname.c_str(), optdesc.c_str());
    }

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

static bool enabled(const po::variables_map &vm, const std::string &api)
{
    return !vm.count("no-" + api);
}

int main(int argc, char **argv)
{
    po::variables_map vm = processOptions(argc, argv);
    const int iterations = vm["iterations"].as<int>();
    const int items = vm["items"].as<int>();

    device_type d = DEVICE_TYPE_GPU;
    if (vm.count("cpu"))
        d = DEVICE_TYPE_CPU;
    if (vm.count("csv"))
    {
        std::cout << "algorithm,api,N,iter,time,rate\n";
        csv = true;
    }

    if (!vm.count("no-scan"))
    {
        std::vector<std::int32_t> a(items);
        for (std::size_t i = 0; i < a.size(); i++)
            a[i] = i;

        output_header("Scan");
        for (const auto &entry : scan_registry<std::int32_t>::get())
        {
            if (enabled(vm, entry.api))
            {
                try
                {
                    auto ptr = entry.factory(d, a);
                    time_algorithm(*ptr, "scan", entry.api, entry.name, items, iterations);
                }
                catch (device_not_supported)
                {
                }
            }
        }
        output_footer();
    }

    if (!vm.count("no-sort"))
    {
        std::vector<std::uint32_t> keys(items);
        std::vector<std::uint32_t> values(items);
        for (std::size_t i = 0; i < keys.size(); i++)
        {
            keys[i] = (std::uint32_t) i * 0x9E3779B9;
            values[i] = i;
        }

        output_header("Sort");
        for (const auto &entry : sort_registry<std::uint32_t, void>::get())
        {
            if (enabled(vm, entry.api))
            {
                try
                {
                    auto ptr = entry.factory(d, keys, void_vector());
                    time_algorithm(*ptr, "sort", entry.api, entry.name, items, iterations);
                }
                catch (device_not_supported)
                {
                }
            }
        }
        output_footer();

        output_header("Sort by key");
        for (const auto &entry : sort_registry<std::uint32_t, std::uint32_t>::get())
        {
            if (enabled(vm, entry.api))
            {
                try
                {
                    auto ptr = entry.factory(d, keys, values);
                    time_algorithm(*ptr, "sort-by-key", entry.api, entry.name, items, iterations);
                }
                catch (device_not_supported)
                {
                }
            }
        }
        output_footer();
    }

    return 0;
}
