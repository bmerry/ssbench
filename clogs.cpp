#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>
#undef CL_VERSION_1_2
#include <clogs/scan.h>
#include <clogs/radixsort.h>
#include "algorithms.h"
#include "register.h"
#include "clutils.h"

template<typename T>
struct clogs_type
{
};

template<>
struct clogs_type<cl_int>
{
    static clogs::Type type() { return clogs::TYPE_INT; }
};

template<>
struct clogs_type<cl_uint>
{
    static clogs::Type type() { return clogs::TYPE_UINT; }
};

template<typename T>
struct clogs_vector : public cl::Buffer
{
    using cl::Buffer::Buffer;
    std::size_t size() const { return getInfo<CL_MEM_SIZE>() / sizeof(T); }
};

class clogs_algorithm
{
private:
    cl::Device device;
    cl::Context ctx;
    cl::CommandQueue queue;

    std::unique_ptr<clogs::Scan> scanner;
    std::unique_ptr<clogs::Radixsort> sorter;

public:
    template<typename T>
    struct types
    {
        typedef clogs_vector<T> vector;
        typedef clogs_vector<T> scan_vector;
        typedef clogs_vector<T> sort_vector;
    };

    template<typename T>
    void create(clogs_vector<T> &out, std::size_t elements)
    {
        out = clogs_vector<T>(ctx, CL_MEM_READ_WRITE, elements * sizeof(T));
    }

    template<typename T>
    void copy(const std::vector<T> &src, clogs_vector<T> &dst) const
    {
        queue.enqueueWriteBuffer(dst, CL_TRUE, 0, src.size() * sizeof(T), src.data());
    }

    template<typename T>
    void copy(const clogs_vector<T> &src, clogs_vector<T> &dst) const
    {
        queue.enqueueCopyBuffer(src, dst, 0, 0, src.template getInfo<CL_MEM_SIZE>());
    }

    template<typename T>
    void copy(const clogs_vector<T> &src, std::vector<T> &dst) const
    {
        queue.enqueueReadBuffer(src, CL_TRUE, 0, dst.size() * sizeof(T), dst.data());
    }

    template<typename T>
    void pre_scan(const clogs_vector<T> &src, clogs_vector<T> &dst)
    {
        scanner.reset(new clogs::Scan(ctx, device, clogs_type<T>::type()));
    }

    template<typename T>
    void scan(const clogs_vector<T> &src, clogs_vector<T> &dst) const
    {
        scanner->enqueue(queue, src, dst, src.size());
    }

    template<typename K, typename V>
    void pre_sort_by_key(clogs_vector<K> &keys, clogs_vector<V> &values)
    {
        sorter.reset(new clogs::Radixsort(ctx, device, clogs_type<K>::type(), clogs_type<V>::type()));
        clogs_vector<K> temp_keys;
        clogs_vector<V> temp_values;
        create(temp_keys, keys.size());
        create(temp_values, values.size());
        sorter->setTemporaryBuffers(temp_keys, temp_values);
    }

    template<typename K, typename V>
    void sort_by_key(clogs_vector<K> &keys, clogs_vector<V> &values) const
    {
        sorter->enqueue(queue, keys, values, keys.size());
    }

    template<typename K>
    void pre_sort(clogs_vector<K> &keys)
    {
        sorter.reset(new clogs::Radixsort(ctx, device, clogs_type<K>::type()));
        clogs_vector<K> temp_keys;
        create(temp_keys, keys.size());
        sorter->setTemporaryBuffers(temp_keys, cl::Buffer());
    }

    template<typename K>
    void sort(clogs_vector<K> &keys) const
    {
        sorter->enqueue(queue, keys, cl::Buffer(), keys.size());
    }

    void finish()
    {
        queue.finish();
    }

    static std::string api() { return "clogs"; }

    explicit clogs_algorithm(device_type d)
        : device(device_from_type(d)),
        ctx(device),
        queue(ctx, device)
    {
    }
};

static register_algorithms<clogs_algorithm> register_clogs;
