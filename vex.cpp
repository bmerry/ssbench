#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>
#undef CL_VERSION_1_2
#include <vexcl/vexcl.hpp>
#include <vexcl/external/clogs.hpp>
#include <stdexcept>
#include "algorithms.h"
#include "register.h"
#include "clutils.h"

template<typename T>
struct vex_traits
{
    typedef vex::vector<T> vector;

    static std::vector<T> get(const vector &v)
    {
        std::vector<T> ans(v.size());
        vex::copy(v, ans);
        return ans;
    }

    template<typename K>
    static void sort(vex::vector<K> &keys, vector &values)
    {
        vex::sort_by_key(keys, values);
    }
};

template<>
struct vex_traits<void>
{
    struct vector
    {
        vector() {}
        vector(const vex::Context &ctx, const void_vector &) {}
        vector(const vex::Context &ctx, std::size_t size) {}
        std::size_t size() const { return 0; }
    };

    static void_vector get(const vector &v)
    {
        return void_vector();
    }

    template<typename K>
    static void sort(vex::vector<K> &keys, vector &values)
    {
        vex::sort(keys);
    }
};

class vex_algorithm
{
protected:
    vex::Context ctx;

    explicit vex_algorithm(device_type d) : ctx(vex::Filter::Type(type_to_cl_type(d)) && vex::Filter::Position(0))
    {
        if (!ctx)
            throw device_not_supported();
    }
};

/************************************************************************/

template<typename T>
class vex_scan : public scan_algorithm<T>, public vex_algorithm
{
protected:
    vex::vector<T> d_a, d_scan;

public:
    vex_scan(device_type d, const std::vector<T> &h_a)
        : scan_algorithm<T>(h_a), vex_algorithm(d), d_a(ctx, h_a), d_scan(ctx, h_a.size())
    {
    }

    static std::string name() { return "vex::exclusive_scan"; }
    static std::string api() { return "vex"; }
    virtual void finish() override { ctx.finish(); }

    virtual void run() override
    {
        vex::exclusive_scan(d_a, d_scan);
    }

    virtual std::vector<T> get() const override
    {
        std::vector<T> ans(d_scan.size());
        vex::copy(d_scan, ans);
        return ans;
    }
};


static register_scan_algorithm<vex_scan> register_vex_scan;

/************************************************************************/

template<typename K, typename V>
class vex_sort : public sort_algorithm<K, V>, public vex_algorithm
{
protected:
    typedef typename vector_of<K>::type key_vector;
    typedef typename vector_of<V>::type value_vector;
    typedef typename vex_traits<K>::vector d_key_vector;
    typedef typename vex_traits<V>::vector d_value_vector;

    d_key_vector d_keys, d_sorted_keys;
    d_value_vector d_values, d_sorted_values;

public:
    vex_sort(device_type d, const key_vector &h_keys, const value_vector &h_values)
        : sort_algorithm<K, V>(h_keys, h_values), vex_algorithm(d),
        d_keys(ctx, h_keys),
        d_sorted_keys(ctx, h_keys.size()),
        d_values(ctx, h_values),
        d_sorted_values(ctx, h_values.size())
    {
    }

    static std::string name() { return "vex::sort"; }
    static std::string api() { return "vex"; }
    virtual void finish() override { ctx.finish(); }

    virtual void run() override
    {
        d_sorted_keys = d_keys;
        d_sorted_values = d_values;
        vex_traits<V>::template sort<K>(d_sorted_keys, d_sorted_values);
    }

    virtual std::pair<key_vector, value_vector> get() const override
    {
        return std::make_pair(
            vex_traits<K>::get(d_sorted_keys),
            vex_traits<V>::get(d_sorted_values));
    }
};

static register_sort_algorithm<vex_sort> register_vex_sort;
