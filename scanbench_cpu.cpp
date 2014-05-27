#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <parallel/algorithm>
#include <parallel/numeric>
#include <omp.h>
#include "scanbench_algorithms.h"
#include "scanbench_register.h"
#include "hostutils.h"

template<typename T>
class cpu_scan : public scan_algorithm<T>
{
protected:
    std::vector<T> a;
    std::vector<T> out;
public:
    cpu_scan(device_type d, const std::vector<T> &h_a)
        : scan_algorithm<T>(h_a), a(h_a), out(h_a.size())
    {
        if (d != DEVICE_TYPE_CPU)
            throw device_not_supported();
    }
    static std::string api() { return "cpu"; }
    virtual void finish() override {}
    virtual std::vector<T> get() const override { return out; }
};

template<typename T>
class serial_scan : public cpu_scan<T>
{
public:
    using cpu_scan<T>::cpu_scan;
    static std::string name() { return "serial scan"; }
    virtual void run() override
    {
        std::partial_sum(this->a.begin(), this->a.end() - 1, this->out.begin() + 1);
        this->out[0] = T();
    }
};

template<typename T>
class parallel_scan : public cpu_scan<T>
{
public:
    using cpu_scan<T>::cpu_scan;

    static std::string name() { return "parallel scan"; }
    virtual void run() override
    {
        __gnu_parallel::partial_sum(this->a.begin(), this->a.end() - 1, this->out.begin() + 1);
        this->out[0] = T();
    }
};

template<typename T>
class my_parallel_scan : public cpu_scan<T>
{
public:
    using cpu_scan<T>::cpu_scan;

    static std::string name() { return "my parallel scan"; }
    virtual void run() override
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
            auto in_begin = this->a.cbegin();
            for (std::size_t start = 0; start < this->a.size(); start += chunk)
            {
                std::size_t len = std::min(this->a.size() - start, chunk);
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
                    sum += this->a[i];
                    this->out[i] = tmp;
                }
            }
        }
    }
};

static register_scan_algorithm<serial_scan> register_serial_scan;
static register_scan_algorithm<parallel_scan> register_parallel_scan;
static register_scan_algorithm<my_parallel_scan> register_my_parallel_scan;

/************************************************************************/

template<typename K, typename V>
class cpu_sort : public sort_algorithm<K, V>
{
protected:
    typedef typename vector_of<K>::type key_vector;
    typedef typename vector_of<V>::type value_vector;
    key_vector keys, sorted_keys;
    value_vector values, sorted_values;

public:
    cpu_sort(device_type d, const key_vector &h_keys, const value_vector &h_values)
        : sort_algorithm<K, V>(h_keys, h_values),
        keys(h_keys),
        sorted_keys(h_keys.size()),
        values(h_values),
        sorted_values(h_values.size())
    {
        if (d != DEVICE_TYPE_CPU)
            throw device_not_supported();
    }
    static std::string api() { return "cpu"; }
    virtual void finish() override {}
    virtual std::pair<key_vector, value_vector> get() const override
    {
        return std::make_pair(sorted_keys, sorted_values);
    }
};

template<typename K, typename V>
class serial_sort : public cpu_sort<K, V>
{
public:
    using cpu_sort<K, V>::cpu_sort;

    static std::string name() { return "serial sort"; }
    virtual void run() override
    {
        this->sorted_keys = this->keys;
        this->sorted_values = this->values;
        sort_by_key(this->sorted_keys, this->sorted_values);
    }
};

template<typename K, typename V>
class parallel_sort : public cpu_sort<K, V>
{
public:
    using cpu_sort<K, V>::cpu_sort;

    static std::string name() { return "parallel sort"; }
    virtual void run() override
    {
        this->sorted_keys = this->keys;
        this->sorted_values = this->values;

        parallel_sort_by_key(this->sorted_keys, this->sorted_values);
    }
};

static register_sort_algorithm<serial_sort> register_serial_sort;
static register_sort_algorithm<parallel_sort> register_parallel_sort;
