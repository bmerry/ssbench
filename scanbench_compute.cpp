#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>
#undef CL_VERSION_1_2
#include <boost/compute/container/vector.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/exclusive_scan.hpp>
#include <boost/compute/algorithm/sort.hpp>
#include <boost/compute/algorithm/sort_by_key.hpp>
#include "scanbench_algorithms.h"
#include "scanbench_register.h"
#include "clutils.h"

namespace compute = boost::compute;

template<typename T>
struct compute_traits
{
    typedef compute::vector<T> vector;

    static void copy(const vector &src, const vector &dst, compute::command_queue &queue)
    {
        compute::copy_async(src.begin(), src.end(), dst.begin(), queue);
    }

    static std::vector<T> get(const vector &v, compute::command_queue &queue)
    {
        std::vector<T> ans(v.size());
        compute::copy(v.begin(), v.end(), ans.begin(), queue);
        return ans;
    }

    template<typename K>
    static void sort(compute::vector<K> &keys, vector &values, compute::command_queue &queue)
    {
        compute::sort_by_key(keys.begin(), keys.end(), values.begin(), queue);
    }
};

template<>
struct compute_traits<void>
{
    struct vector
    {
        vector() {}
        vector(const void_vector &, const compute::command_queue &) {}
        vector(std::size_t size, const compute::context &) {}
        std::size_t size() const { return 0; }
    };

    static void copy(const vector &, const vector &, compute::command_queue &)
    {
    }

    static void_vector get(const vector &, compute::command_queue &)
    {
        return void_vector();
    }

    template<typename K>
    static void sort(compute::vector<K> &keys, vector &values, compute::command_queue &queue)
    {
        compute::sort(keys.begin(), keys.end(), queue);
    }
};

class compute_algorithm
{
private:
    static compute::device getDevice(device_type d)
    {
        for (const auto &device : compute::system::devices())
        {
            if (device.type() == type_to_cl_type(d))
                return device;
        }
        throw device_not_supported();
    }

protected:
    compute::device device;
    compute::context ctx;
    mutable compute::command_queue queue;

    explicit compute_algorithm(device_type d)
    : device(getDevice(d)),
    ctx(device),
    queue(ctx, device)
    {
    }
};

/************************************************************************/

template<typename T>
class compute_scan : public scan_algorithm<T>, compute_algorithm
{
private:
    compute::vector<T> d_a;
    compute::vector<T> d_scan;

public:
    compute_scan(device_type d, const std::vector<T> &h_a)
        : scan_algorithm<T>(h_a), compute_algorithm(d), d_a(h_a, queue), d_scan(h_a.size(), ctx)
    {
    }

    static std::string name() { return "compute::exclusive_scan"; }
    static std::string api() { return "compute"; }
    virtual void finish() override { queue.finish(); }

    virtual void run() override
    {
        compute::exclusive_scan(d_a.begin(), d_a.end(), d_scan.begin(), queue);
    }

    std::vector<T> get() const override
    {
        std::vector<T> ans(d_scan.size());
        compute::copy(d_scan.begin(), d_scan.end(), ans.begin(), queue);
        return ans;
    }
};

static register_scan_algorithm<compute_scan> register_compute_scan;

/************************************************************************/

template<typename K, typename V>
class compute_sort : public sort_algorithm<K, V>, public compute_algorithm
{
private:
    typedef typename vector_of<K>::type key_vector;
    typedef typename vector_of<V>::type value_vector;
    typedef typename compute_traits<K>::vector d_key_vector;
    typedef typename compute_traits<V>::vector d_value_vector;

    d_key_vector d_keys, d_sorted_keys;
    d_value_vector d_values, d_sorted_values;

public:
    compute_sort(device_type d, const key_vector &h_keys, const value_vector &h_values)
        : sort_algorithm<K, V>(h_keys, h_values),
        compute_algorithm(d),
        d_keys(h_keys, queue),
        d_sorted_keys(h_keys.size(), ctx),
        d_values(h_values, queue),
        d_sorted_values(h_values.size(), ctx)
    {
    }

    static std::string name() { return "compute::sort"; }
    static std::string api() { return "compute"; }
    virtual void finish() override { queue.finish(); }

    virtual void run() override
    {
        compute_traits<K>::copy(d_keys, d_sorted_keys, queue);
        compute_traits<V>::copy(d_values, d_sorted_values, queue);
        compute_traits<V>::sort(d_sorted_keys, d_sorted_values, queue);
    }

    virtual std::pair<key_vector, value_vector> get() const override
    {
        return std::make_pair(compute_traits<K>::get(d_sorted_keys, queue),
                              compute_traits<V>::get(d_sorted_values, queue));
    }
};

static register_sort_algorithm<compute_sort> register_compute_sort;
