#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>
#undef CL_VERSION_1_2
#include <boost/compute/container/vector.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/exclusive_scan.hpp>
#include <boost/compute/algorithm/sort.hpp>
#include <boost/compute/algorithm/sort_by_key.hpp>
#include <memory>
#include "algorithms.h"
#include "register.h"
#include "clutils.h"

namespace compute = boost::compute;

class compute_algorithm
{
private:
    compute::device device;
    compute::context ctx;
    mutable compute::command_queue queue;

    static compute::device getDevice(device_info d)
    {
        int index = d.index;
        for (const auto &device : compute::system::devices())
        {
            if (device.type() == type_to_cl_type(d.type))
            {
                if (index == 0)
                    return device;
                index--;
            }
        }
        throw device_not_supported();
    }

public:
    template<typename T>
    struct types
    {
        // compute::vector<T> cannot be initialized after construction with a different context
        typedef std::unique_ptr<compute::vector<T> > vector;
        typedef vector scan_vector;
        typedef vector sort_vector;
    };

    template<typename T>
    void create(std::unique_ptr<compute::vector<T> > &out, std::size_t elements)
    {
        out.reset(new compute::vector<T>(elements, ctx));
    }

    template<typename T>
    void copy(const std::vector<T> &src, std::unique_ptr<compute::vector<T> > &dst) const
    {
        compute::copy(src.begin(), src.end(), dst->begin(), queue);
    }

    template<typename T>
    void copy(const std::unique_ptr<compute::vector<T> > &src, std::unique_ptr<compute::vector<T> > &dst) const
    {
        compute::copy_async(src->begin(), src->end(), dst->begin(), queue);
    }

    template<typename T>
    void copy(const std::unique_ptr<compute::vector<T> > &src, std::vector<T> &dst) const
    {
        compute::copy(src->begin(), src->end(), dst.begin(), queue);
    }

    template<typename T>
    void pre_scan(const std::unique_ptr<compute::vector<T> > &src, std::unique_ptr<compute::vector<T> > &dst)
    {
    }

    template<typename T>
    void scan(const std::unique_ptr<compute::vector<T> > &src, std::unique_ptr<compute::vector<T> > &dst) const
    {
        compute::exclusive_scan(src->begin(), src->end(), dst->begin(), queue);
    }

    template<typename K, typename V>
    void pre_sort_by_key(std::unique_ptr<compute::vector<K> > &keys, std::unique_ptr<compute::vector<V> > &values)
    {
    }

    template<typename K, typename V>
    void sort_by_key(std::unique_ptr<compute::vector<K> > &keys, std::unique_ptr<compute::vector<V> > &values) const
    {
        compute::sort_by_key(keys->begin(), keys->end(), values->begin(), queue);
    }

    template<typename K>
    void pre_sort(std::unique_ptr<compute::vector<K> > &keys)
    {
    }

    template<typename K>
    void sort(std::unique_ptr<compute::vector<K> > &keys) const
    {
        compute::sort(keys->begin(), keys->end(), queue);
    }

    void finish()
    {
        queue.finish();
    }

    static std::string api() { return "compute"; }

    explicit compute_algorithm(device_info d)
        : device(getDevice(d)),
        ctx(device),
        queue(ctx, device)
    {
    }
};

static register_algorithms<compute_algorithm> register_compute;
