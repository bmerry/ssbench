#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <parallel/algorithm>
#include <parallel/numeric>
#include <omp.h>
#include "scanbench.h"

template<typename T>
class cpu_scan : public scan_algorithm<T>
{
protected:
    std::vector<T> a;
    std::vector<T> out;
public:
    cpu_scan(const std::vector<T> &h_a) : scan_algorithm<T>(h_a), a(h_a), out(h_a.size()) {}
    virtual std::string api() const override { return "cpu"; }
    virtual void finish() override {}
    virtual std::vector<T> get() const override { return out; }
};

template<typename T>
class serial_scan : public cpu_scan<T>
{
public:
    using cpu_scan<T>::cpu_scan;
    std::string name() const { return "serial scan"; }
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

    std::string name() const { return "parallel scan"; }
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

    std::string name() const { return "my parallel scan"; }
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

template<typename T>
class cpu_sort : public sort_algorithm<T>
{
protected:
    std::vector<T> a;
    std::vector<T> target;

public:
    cpu_sort(const std::vector<T> &h_a)
        : sort_algorithm<T>(h_a), a(h_a), target(h_a.size())
    {
    }

    virtual std::string api() const override { return "cpu"; }
    virtual void finish() override {}
    virtual std::vector<T> get() const override { return target; }
};

template<typename T>
class serial_sort : public cpu_sort<T>
{
public:
    using cpu_sort<T>::cpu_sort;

    virtual std::string name() const override { return "serial sort"; }
    virtual void run() override
    {
        this->target = this->a;
        std::sort(this->target.begin(), this->target.end());
    }
};

template<typename T>
class parallel_sort : public cpu_sort<T>
{
public:
    using cpu_sort<T>::cpu_sort;

    virtual std::string name() const override { return "parallel sort"; }
    virtual void run() override
    {
        this->target = this->a;
        __gnu_parallel::sort(this->target.begin(), this->target.end());
    }
};

static register_sort_algorithm<serial_sort> register_serial_sort;
static register_sort_algorithm<parallel_sort> register_parallel_sort;
