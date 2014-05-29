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

template<>
struct clogs_type<void>
{
    static clogs::Type type() { return clogs::Type(); }
};

template<typename T>
struct clogs_traits
{
    // Creates a buffer with a copy of data, or with uninitialized contents if flags does
    // not contain CL_MEM_COPY_HOST_PTR
    static cl::Buffer make_buffer(const cl::Context &ctx, cl_mem_flags flags, const std::vector<T> &data)
    {
        return cl::Buffer(ctx, flags, data.size() * sizeof(T),
                          (flags & CL_MEM_COPY_HOST_PTR) ? const_cast<T *>(data.data()) : nullptr);
    }

    static void copy_buffer(const cl::CommandQueue &queue, const cl::Buffer &src, const cl::Buffer &trg)
    {
        queue.enqueueCopyBuffer(src, trg, 0, 0, src.getInfo<CL_MEM_SIZE>());
    }

    static std::vector<T> get_buffer(const cl::CommandQueue &queue, const cl::Buffer &buffer)
    {
        std::size_t elements = buffer.getInfo<CL_MEM_SIZE>() / sizeof(T);
        std::vector<T> ans(elements);
        queue.enqueueReadBuffer(buffer, CL_TRUE, 0, elements * sizeof(T), ans.data());
        return ans;
    }
};

template<>
struct clogs_traits<void>
{
    static cl::Buffer make_buffer(const cl::Context &, cl_mem_flags, const void_vector &)
    {
        return cl::Buffer();
    }

    static void copy_buffer(const cl::CommandQueue &, const cl::Buffer &, const cl::Buffer &)
    {
    }

    static void_vector get_buffer(const cl::CommandQueue &, const cl::Buffer &)
    {
        return void_vector();
    }
};

/************************************************************************/

class clogs_algorithm
{
protected:
    cl::Device device;
    cl::Context ctx;
    cl::CommandQueue queue;

    explicit clogs_algorithm(device_type d)
        : device(device_from_type(d)),
        ctx(device),
        queue(ctx, device)
    {
    }
};

/************************************************************************/

template<typename T>
class clogs_scan : public scan_algorithm<T>, public clogs_algorithm
{
private:
    std::size_t elements;
    cl::Buffer d_a;
    cl::Buffer d_scan;
    clogs::Scan scan;

public:
    clogs_scan(device_type d, const std::vector<T> &h_a)
        : scan_algorithm<T>(h_a), clogs_algorithm(d),
        elements(h_a.size()),
        d_a(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            h_a.size() * sizeof(T),
            const_cast<T *>(h_a.data())),
        d_scan(ctx, CL_MEM_READ_WRITE, elements * sizeof(T)),
        scan(ctx, device, clogs_type<T>::type())
    {
    }

    static std::string name() { return "clogs::Scan"; }
    static std::string api() { return "clogs"; }
    virtual void finish() override { queue.finish(); }

    virtual void run() override
    {
        scan.enqueue(queue, d_a, d_scan, elements);
    }

    virtual std::vector<T> get() const override
    {
        std::vector<T> ans(elements);
        queue.enqueueReadBuffer(d_scan, CL_TRUE, 0, elements * sizeof(T), ans.data());
        return ans;
    }
};

static register_scan_algorithm<clogs_scan> register_clogs_scan;

/************************************************************************/

template<typename K, typename V>
class clogs_sort : public sort_algorithm<K, V>, public clogs_algorithm
{
private:
    std::size_t elements;
    cl::Buffer d_keys, d_values;
    cl::Buffer d_sorted_keys, d_sorted_values;
    cl::Buffer d_tmp_keys, d_tmp_values;
    clogs::Radixsort sort;

    typedef typename vector_of<K>::type key_vector;
    typedef typename vector_of<V>::type value_vector;

public:
    clogs_sort(device_type d, const key_vector &h_keys, const value_vector &h_values)
        : sort_algorithm<K, V>(h_keys, h_values), clogs_algorithm(d),
        elements(h_keys.size()),
        d_keys(clogs_traits<K>::make_buffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, h_keys)),
        d_values(clogs_traits<V>::make_buffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, h_values)),
        d_sorted_keys(clogs_traits<K>::make_buffer(ctx, CL_MEM_READ_WRITE, h_keys)),
        d_sorted_values(clogs_traits<V>::make_buffer(ctx, CL_MEM_READ_WRITE, h_values)),
        d_tmp_keys(clogs_traits<K>::make_buffer(ctx, CL_MEM_READ_WRITE, h_keys)),
        d_tmp_values(clogs_traits<V>::make_buffer(ctx, CL_MEM_READ_WRITE, h_values)),
        sort(ctx, device, clogs_type<K>::type(), clogs_type<V>::type())
    {
        sort.setTemporaryBuffers(d_tmp_keys, d_tmp_values);
    }

    static std::string name() { return "clogs::Radixsort"; }
    static std::string api() { return "clogs"; }
    virtual void finish() override { queue.finish(); }

    virtual void run() override
    {
        clogs_traits<K>::copy_buffer(queue, d_keys, d_sorted_keys);
        clogs_traits<V>::copy_buffer(queue, d_values, d_sorted_values);
        sort.enqueue(queue, d_sorted_keys, d_sorted_values, elements);
    }

    virtual std::pair<key_vector, value_vector> get() const override
    {
        return std::make_pair(clogs_traits<K>::get_buffer(queue, d_sorted_keys),
                              clogs_traits<V>::get_buffer(queue, d_sorted_values));
    }
};

static register_sort_algorithm<clogs_sort> register_clogs_sort;
