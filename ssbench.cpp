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
#include "algorithms.h"
#include "register.h"

namespace po = boost::program_options;

typedef std::chrono::high_resolution_clock clock_type;

static bool csv = false;
static std::set<std::string> enabled_apis;
static std::set<std::string> enabled_algorithms;

struct properties
{
    std::size_t N;
    int iterations;
    int warmup;
};

static void output_header(const std::string &name)
{
    if (!csv)
    {
        std::cout << name << "\n\n";
    }
}

static void output_result(
    const std::string &algname, const std::string &api,
    double time, const properties &props)
{
    double rate = 1e-6 * props.N * props.iterations / time;
    if (csv)
    {
        std::cout << algname << "," << api << "," << props.N << "," << props.iterations
            << "," << time << "," << rate << '\n';
    }
    else
    {
        std::cout << std::setw(20) << std::fixed << std::setprecision(1);
        std::cout << rate << " M/s\t";
        std::cout << std::setw(0) << std::setprecision(6);
        std::cout << time << "\t" << api << '\n';
    }
}

static void output_footer()
{
    if (!csv)
        std::cout << '\n';
}

static bool enabled_api(const std::string &api)
{
    return enabled_apis.empty() || enabled_apis.count(api);
}

static bool enabled_algorithm(const std::string &algorithm)
{
    return enabled_algorithms.empty() || enabled_algorithms.count(algorithm);
}

static void time_algorithm(
    algorithm &alg,
    const std::string &algname, const std::string &api,
    const properties &props)
{
    // Warmup
    for (int i = 0; i < props.warmup; i++)
        alg.run();
    alg.finish();

    // Real run
    auto start = clock_type::now();
    for (int i = 0; i < props.iterations; i++)
        alg.run();
    alg.finish();
    auto stop = clock_type::now();

    alg.validate();

    std::chrono::duration<double> elapsed(stop - start);
    double time = elapsed.count();
    output_result(algname, api, time, props);
}

template<typename T, typename... FactoryArgs>
static void time_entry(
    const T &entry, const std::string &algname, const properties &props,
    FactoryArgs &&... args)
{
    if (enabled_api(entry.api))
    {
        try
        {
            auto ptr = entry.factory(std::forward<FactoryArgs>(args)...);
            time_algorithm(*ptr, algname, entry.api, props);
        }
        catch (device_not_supported)
        {
        }
    }
}

static void usage(std::ostream &o, const po::options_description &opts)
{
    o << "Usage: ssbench [options]\n\n";
    o << opts;
}

static po::variables_map processOptions(int argc, char **argv)
{
    po::options_description opts;

    opts.add_options()
        ("help,h",        "show usage")
        ("items,N",       po::value<int>()->default_value(16777216), "Problem size")
        ("iterations,R",  po::value<int>()->default_value(10), "Number of repetitions")
        ("warmup",        po::value<int>()->default_value(3), "Number of warmup rounds")
        ("csv",           "output results in CSV format")
        ("cpu",           "run algorithms on the CPU")
        ("api,a",         po::value<std::vector<std::string> >()->composing(), "library to benchmark")
        ("algorithm,A",   po::value<std::vector<std::string> >()->composing(), "algorithm to benchmark");

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
    properties props;
    props.iterations = vm["iterations"].as<int>();
    props.warmup = vm["warmup"].as<int>();
    props.N = vm["items"].as<int>();
    device_type d = vm.count("cpu") ? DEVICE_TYPE_CPU : DEVICE_TYPE_GPU;
    csv = vm.count("csv");
    if (vm.count("api"))
        for (const std::string &a : vm["api"].as<std::vector<std::string> >())
            enabled_apis.insert(a);
    if (vm.count("algorithm"))
        for (const std::string &a : vm["algorithm"].as<std::vector<std::string> >())
            enabled_algorithms.insert(a);


    if (enabled_algorithm("scan"))
    {
        std::vector<std::int32_t> a(props.N);
        for (std::size_t i = 0; i < a.size(); i++)
            a[i] = i;

        output_header("Scan");
        for (const auto &entry : scan_registry<std::int32_t>::get())
            time_entry(entry, "scan", props, d, a);
        output_footer();
    }

    if (enabled_algorithm("sort") || enabled_algorithm("sort-by-key"))
    {
        std::vector<std::uint32_t> keys(props.N);
        std::vector<std::uint32_t> values(props.N);
        for (std::size_t i = 0; i < keys.size(); i++)
        {
            keys[i] = (std::uint32_t) i * 0x9E3779B9;
            values[i] = i;
        }

        if (enabled_algorithm("sort"))
        {
            output_header("Sort");
            for (const auto &entry : sort_registry<std::uint32_t>::get())
                time_entry(entry, "sort", props, d, keys);
            output_footer();
        }

        if (enabled_algorithm("sort-by-key"))
        {
            output_header("Sort by key");
            for (const auto &entry : sort_by_key_registry<std::uint32_t, std::uint32_t>::get())
                time_entry(entry, "sort-by-key", props, d, keys, values);
            output_footer();
        }
    }

    return 0;
}
