#include <bolt/cl/scan.h>
#include <bolt/cl/sort.h>
#include <bolt/cl/sort_by_key.h>
#include <bolt/cl/copy.h>
#include <bolt/cl/device_vector.h>
#include <stdexcept>
#include "algorithms.h"
#include "register.h"
#include "clutils.h"

// Bolt has lots of constness bugs, so we wrap up a vector with mutable
template<typename T>
struct bolt_vector
{
    mutable bolt::cl::device_vector<T> data;
};

class bolt_algorithm
{
private:
    bolt::cl::control control;

public:
    template<typename T>
    struct types
    {
        typedef bolt_vector<T> vector;
        typedef vector scan_vector;
        typedef vector sort_vector;
    };

    template<typename T>
    void create(bolt_vector<T> &out, std::size_t elements) const
    {
        typename bolt::cl::device_vector<T>::device_vector(
            elements, T(), CL_MEM_READ_WRITE, false, control)
            .swap(out.data);
    }

    template<typename T>
    void copy(const std::vector<T> &src, bolt_vector<T> &dst) const
    {
        bolt::cl::copy(control, src.begin(), src.end(), dst.data.begin());
    }

    template<typename T>
    void copy(const bolt_vector<T> &src, bolt_vector<T> &dst) const
    {
        bolt::cl::copy(control, src.data.begin(), src.data.end(), dst.data.begin());
    }

    template<typename T>
    void copy(const bolt_vector<T> &src, std::vector<T> &dst) const
    {
        bolt::cl::copy(control, src.data.begin(), src.data.end(), dst.begin());
    }

    template<typename T>
    static void pre_scan(const bolt_vector<T> &src, bolt_vector<T> &dst) {}

    template<typename T>
    void scan(const bolt_vector<T> &src, bolt_vector<T> &dst)
    {
        bolt::cl::exclusive_scan(control, src.data.begin(), src.data.end(), dst.data.begin());
    }

    template<typename K, typename V>
    static void pre_sort_by_key(bolt_vector<K> &keys, bolt_vector<V> &values) {}

    template<typename K, typename V>
    void sort_by_key(bolt_vector<K> &keys, bolt_vector<V> &values)
    {
        bolt::cl::sort_by_key(control, keys.data.begin(), keys.data.end(), values.data.begin());
    }

    template<typename T>
    static void pre_sort(bolt_vector<T> &keys) {}

    template<typename T>
    void sort(bolt_vector<T> &keys)
    {
        bolt::cl::sort(control, keys.data.begin(), keys.data.end());
    }

    void finish()
    {
        control.getCommandQueue().finish();
    }

    static std::string api() { return "bolt"; }

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

static register_algorithms<bolt_algorithm> register_bolt;
