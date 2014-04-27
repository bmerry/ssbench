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

static void time_algorithm(algorithm &alg, const std::string &name, size_t N, int iter)
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
    std::cout << time << "\t" << name << '\n';
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
        ("no-sort",       "disable all sorting algorithms")
        ("no-scan",       "disable all scan algorithms");

    std::set<std::string> apis;
    for (const auto &entry : scan_registry<std::int32_t>::get())
        apis.insert(entry.api);
    for (const auto &entry : sort_registry<std::uint32_t>::get())
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

    if (!vm.count("no-scan"))
    {
        std::vector<std::int32_t> a(items);
        for (std::size_t i = 0; i < a.size(); i++)
            a[i] = i;

        std::cout << "Scan\n\n";
        for (const auto &entry : scan_registry<std::int32_t>::get())
        {
            if (enabled(vm, entry.api))
            {
                auto ptr = entry.factory(a);
                time_algorithm(*ptr, entry.name, items, iterations);
            }
        }
        std::cout << "\n";
    }

    if (!vm.count("no-sort"))
    {
        std::vector<std::uint32_t> rnd(items);
        for (std::size_t i = 0; i < rnd.size(); i++)
            rnd[i] = (std::uint32_t) i * 0x9E3779B9;

        std::cout << "Sort\n\n";
        for (const auto &entry : sort_registry<std::uint32_t>::get())
        {
            if (enabled(vm, entry.api))
            {
                auto ptr = entry.factory(rnd);
                time_algorithm(*ptr, entry.name, items, iterations);
            }
        }
        std::cout << "\n";
    }

    return 0;
}
