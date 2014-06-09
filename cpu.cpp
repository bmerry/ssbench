#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <parallel/algorithm>
#include <parallel/numeric>
#include <omp.h>
#include "algorithms.h"
#include "register.h"
#include "hostutils.h"

class cpu_algorithm
{
public:
    template<typename T>
    struct types
    {
        typedef std::vector<T> vector;
        typedef std::vector<T> scan_vector;
        typedef std::vector<T> sort_vector;
    };

    template<typename T>
    static void create(std::vector<T> &out, std::size_t elements)
    {
        out.resize(elements);
    }

    template<typename T>
    static void copy(const std::vector<T> &src, std::vector<T> &dst)
    {
        dst = src;
    }

    template<typename T>
    static void pre_scan(std::vector<T> &src, std::vector<T> &dst) {}

    template<typename K, typename V>
    static void pre_sort_by_key(std::vector<K> &keys, std::vector<V> &values) {}

    template<typename T>
    static void pre_sort(std::vector<T> &keys) {}

    static void finish() {}

    explicit cpu_algorithm(device_info d)
    {
        if (d.type != DEVICE_TYPE_CPU)
            throw device_not_supported();
    }
};

class serial_algorithm : public cpu_algorithm
{
public:
    using cpu_algorithm::cpu_algorithm;

    static std::string api() { return "serial"; }

    template<typename T>
    static void scan(const std::vector<T> &src, std::vector<T> &dst)
    {
        std::partial_sum(src.begin(), src.end() - 1, dst.begin() + 1);
        dst[0] = T();
    }

    template<typename K>
    static void sort(std::vector<K> &keys)
    {
        std::sort(keys.begin(), keys.end());
    }

    template<typename K, typename V>
    static void sort_by_key(std::vector<K> &keys, std::vector<V> &values)
    {
        ::sort_by_key(keys, values);
    }
};

class parallel_algorithm : public cpu_algorithm
{
public:
    using cpu_algorithm::cpu_algorithm;

    static std::string api() { return "parallel"; }

    template<typename T>
    static void scan(const std::vector<T> &src, std::vector<T> &dst)
    {
        __gnu_parallel::partial_sum(src.begin(), src.end() - 1, dst.begin() + 1);
        dst[0] = T();
    }

    template<typename K>
    static void sort(std::vector<K> &keys)
    {
        __gnu_parallel::sort(keys.begin(), keys.end());
    }

    template<typename K, typename V>
    static void sort_by_key(std::vector<K> &keys, std::vector<V> &values)
    {
        ::parallel_sort_by_key(keys, values);
    }
};

class my_parallel_algorithm : public parallel_algorithm
{
public:
    using parallel_algorithm::parallel_algorithm;

    static std::string api() { return "my_parallel"; }

    template<typename T>
    static void scan(const std::vector<T> &src, std::vector<T> &dst)
    {
        std::size_t threads;
#pragma omp parallel
        {
#pragma omp single
            {
                threads = omp_get_num_threads();
            }
        }

        const std::size_t chunk = 2 * 1024 * 1024 / sizeof(T);
        T reduced[threads];
        T carry{};
#pragma omp parallel
        {
            std::size_t tid = omp_get_thread_num();
            auto in_begin = src.cbegin();
            for (std::size_t start = 0; start < src.size(); start += chunk)
            {
                std::size_t len = std::min(src.size() - start, chunk);
                std::size_t pofs = (start + tid * len / threads);
                std::size_t qofs = (start + (tid + 1) * len / threads);
                reduced[tid] = accumulate(in_begin + pofs, in_begin + qofs, T());
#pragma omp barrier
#pragma omp single
                {
                    T sum = carry;
                    for (std::size_t i = 0; i < threads; i++)
                    {
                        T next = sum + reduced[i];
                        reduced[i] = sum;
                        sum = next;
                    }
                    carry = sum;
                }

                T sum = reduced[tid];
                for (std::size_t i = pofs; i != qofs; ++i)
                {
                    T tmp = sum;
                    sum += src[i];
                    dst[i] = tmp;
                }
            }
        }
    }
};

static register_algorithms<serial_algorithm> register_serial;
static register_algorithms<parallel_algorithm> register_parallel;
static register_scan_algorithm<my_parallel_algorithm> register_my_parallel_scan;
