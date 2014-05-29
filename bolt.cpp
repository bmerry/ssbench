#include <bolt/cl/scan.h>
#include <bolt/cl/sort.h>
#include <bolt/cl/sort_by_key.h>
#include <bolt/cl/copy.h>
#include <bolt/cl/device_vector.h>
#include <stdexcept>
#include "algorithms.h"
#include "register.h"
#include "clutils.h"

template<typename T>
struct bolt_traits
{
    typedef bolt::cl::device_vector<T> vector;

    static vector make_vector(std::size_t elements, const bolt::cl::control &control)
    {
        return vector(elements, T(), CL_MEM_READ_WRITE, false, control);
    }

    static vector make_vector(const std::vector<T> &v, const bolt::cl::control &control)
    {
        return vector(v.begin(), v.end(), CL_MEM_READ_WRITE, control);
    }

    static void copy(const bolt::cl::control &control, vector &src, vector &trg)
    {
        bolt::cl::copy(control, src.begin(), src.end(), trg.begin());
    }

    static std::vector<T> get(const vector &v)
    {
        auto ptr = v.data();
        return std::vector<T>(ptr.get(), ptr.get() + v.size());
    }

    template<typename K>
    static void sort(bolt::cl::control &control, bolt::cl::device_vector<T> &keys, vector &values)
    {
        bolt::cl::sort_by_key(control, keys.begin(), keys.end(), values.begin());
    }
};

template<>
struct bolt_traits<void>
{
    struct vector
    {
    };

    static vector make_vector(std::size_t, const bolt::cl::control &)
    {
        return vector();
    }

    static vector make_vector(const void_vector &, const bolt::cl::control &)
    {
        return vector();
    }

    static void copy(const bolt::cl::control &, const vector &, vector &)
    {
    }

    static void_vector get(const vector &v)
    {
        return void_vector();
    }

    template<typename K>
    static void sort(bolt::cl::control &control, bolt::cl::device_vector<K> &keys, vector &values)
    {
        bolt::cl::sort(control, keys.begin(), keys.end());
    }
};

class bolt_algorithm
{
protected:
    bolt::cl::control control;

    explicit bolt_algorithm(device_type d) : control()
    {
        cl::Device device = device_from_type(d);
        cl::Context ctx(device);
        cl::CommandQueue queue(ctx, device);
        control.setCommandQueue(queue);
        switch (d)
        {
        case DEVICE_TYPE_GPU:
            control.setUseHost(bolt::cl::control::NoUseHost);
            control.setForceRunMode(bolt::cl::control::OpenCL);
            break;
        case DEVICE_TYPE_CPU:
            control.setUseHost(bolt::cl::control::UseHost);
            // TODO: install TBB to enable multicore; also use host vectors
            control.setForceRunMode(bolt::cl::control::SerialCpu);
            break;
        }
    }
};

/************************************************************************/

template<typename T>
class bolt_scan : public scan_algorithm<T>, public bolt_algorithm
{
protected:
    bolt::cl::device_vector<T> d_a, d_scan;

public:
    bolt_scan(device_type d, const std::vector<T> &h_a)
        : scan_algorithm<T>(h_a), bolt_algorithm(d),
        d_a(h_a.begin(), h_a.end(), CL_MEM_READ_WRITE, control),
        d_scan(h_a.size(), T(), CL_MEM_READ_WRITE, false, control)
    {
    }

    static std::string name() { return "bolt::cl::exclusive_scan"; }
    static std::string api() { return "bolt"; }
    virtual void finish() override { control.getCommandQueue().finish(); }

    virtual void run() override
    {
        bolt::cl::exclusive_scan(control, d_a.begin(), d_a.end(), d_scan.begin());
    }

    virtual std::vector<T> get() const override
    {
        auto ptr = d_scan.data();
        return std::vector<T>(ptr.get(), ptr.get() + d_scan.size());
    }
};

static register_scan_algorithm<bolt_scan> register_bolt_scan;

/************************************************************************/

template<typename K, typename V>
class bolt_sort : public sort_algorithm<K, V>, public bolt_algorithm
{
protected:
    typedef typename vector_of<K>::type key_vector;
    typedef typename vector_of<V>::type value_vector;
    typedef typename bolt_traits<K>::vector d_key_vector;
    typedef typename bolt_traits<V>::vector d_value_vector;

    d_key_vector d_keys, d_sorted_keys;
    d_value_vector d_values, d_sorted_values;

public:
    bolt_sort(device_type d, const key_vector &h_keys, const value_vector &h_values)
        : sort_algorithm<K, V>(h_keys, h_values), bolt_algorithm(d),
        d_keys(bolt_traits<K>::make_vector(h_keys, control)),
        d_sorted_keys(bolt_traits<K>::make_vector(h_keys.size(), control)),
        d_values(bolt_traits<V>::make_vector(h_values, control)),
        d_sorted_values(bolt_traits<V>::make_vector(h_values.size(), control))
    {
    }

    static std::string name() { return "bolt::cl::sort"; }
    static std::string api() { return "bolt"; }
    virtual void finish() override { control.getCommandQueue().finish(); }

    virtual void run() override
    {
        bolt_traits<K>::copy(control, d_keys, d_sorted_keys);
        bolt_traits<V>::copy(control, d_values, d_sorted_values);
        bolt_traits<V>::template sort<K>(control, d_sorted_keys, d_sorted_values);
    }

    virtual std::pair<key_vector, value_vector> get() const override
    {
        return std::make_pair(
            bolt_traits<K>::get(d_sorted_keys),
            bolt_traits<V>::get(d_sorted_values));
    }
};

static register_sort_algorithm<bolt_sort> register_bolt_sort;
