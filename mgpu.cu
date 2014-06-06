#include <cub/cub.cuh>
#include <vector>
#include <string>
#include <cstddef>
#include <boost/utility.hpp>
#include "algorithms.h"
#include "register.h"
#include "moderngpu.cuh"

class mgpu_algorithm
{
private:
    mgpu::ContextPtr ctx;

public:
    template<typename T>
    struct types
    {
        typedef MGPU_MEM(T) vector;
        typedef vector scan_vector;
        typedef vector sort_vector;
    };

    template<typename T>
    void create(MGPU_MEM(T) &out, std::size_t elements) const
    {
        ctx->Malloc<T>(elements).swap(out);
    }

    template<typename T>
    static void copy(const std::vector<T> &src, MGPU_MEM(T) &dst)
    {
        dst->FromHost(src);
    }

    template<typename T>
    static void copy(const MGPU_MEM(T) &src, MGPU_MEM(T) &dst)
    {
        src->ToDevice(dst->get(), dst->Size());
    }

    template<typename T>
    static void copy(const MGPU_MEM(T) &src, std::vector<T> &dst)
    {
        src->ToHost(dst);
    }

    template<typename T>
    static void pre_scan(const MGPU_MEM(T) &src, MGPU_MEM(T) &dst)
    {
    }

    template<typename T>
    void scan(const MGPU_MEM(T) &src, MGPU_MEM(T) &dst) const
    {
        mgpu::Scan<mgpu::MgpuScanTypeExc>(
            src->get(), src->Size(), T(0), mgpu::plus<T>(), (T *) NULL, (T *) NULL, dst->get(), *ctx);
    }

    template<typename K>
    static void pre_sort(MGPU_MEM(K) &keys)
    {
    }

    template<typename K>
    void sort(MGPU_MEM(K) &keys) const
    {
        mgpu::MergesortKeys(keys->get(), keys->Size(), mgpu::less<K>(), *ctx);
    }

    template<typename K, typename V>
    static void pre_sort_by_key(MGPU_MEM(K) &keys, MGPU_MEM(V) &values)
    {
    }

    template<typename K, typename V>
    void sort_by_key(MGPU_MEM(K) &keys, MGPU_MEM(V) &values) const
    {
        mgpu::MergesortPairs(keys->get(), values->get(), keys->Size(), mgpu::less<K>(), *ctx);
    }

    static void finish()
    {
        cudaDeviceSynchronize();
    }

    static std::string api() { return "mgpu"; }

    explicit mgpu_algorithm(device_type d)
    {
        if (d != DEVICE_TYPE_GPU)
            throw device_not_supported();
        ctx = mgpu::CreateCudaDevice(0);
    }
};

static register_algorithms<mgpu_algorithm> register_mgpu;
