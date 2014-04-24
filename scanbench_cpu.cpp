#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <parallel/algorithm>
#include <parallel/numeric>
#include <omp.h>
#include "scanbench_cpu.h"

template<typename T>
void serial_scan<T>::run()
{
    std::partial_sum(this->a.begin(), this->a.end(), this->out.begin());
}

template<typename T>
void parallel_scan<T>::run()
{
    __gnu_parallel::partial_sum(this->a.begin(), this->a.end(), this->out.begin());
}

template<typename T>
void my_parallel_scan<T>::run()
{
    // TODO: should write to out, not in-place?

    std::size_t threads;
#pragma omp parallel
    {
#pragma omp single
        {
            threads = omp_get_num_threads();
        }
    }

    const std::size_t chunk = 4 * 1024 * 1024 / sizeof(T);
    T reduced[threads];
    T carry{};
#pragma omp parallel
    {
        std::size_t tid = omp_get_thread_num();
        auto begin = this->a.begin();
        for (std::size_t start = 0; start < this->a.size(); start += chunk)
        {
            std::size_t len = std::min(this->a.size() - start, chunk);
            auto p = begin + (start + tid * len / threads);
            auto q = begin + (start + (tid + 1) * len / threads);
            reduced[tid] = accumulate(p, q, T());
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
            for (auto i = p; i != q; ++i)
            {
                T tmp = sum;
                sum += *p;
                *p = tmp;
            }
        }
    }
}

template class serial_scan<std::int32_t>;
template class parallel_scan<std::int32_t>;
template class my_parallel_scan<std::int32_t>;

/************************************************************************/

template<typename T>
void serial_sort<T>::run()
{
    this->target = this->a;
    std::sort(this->target.begin(), this->target.end());
}

template<typename T>
void parallel_sort<T>::run()
{
    this->target = this->a;
    __gnu_parallel::sort(this->target.begin(), this->target.end());
}

template class serial_sort<std::uint32_t>;
template class parallel_sort<std::uint32_t>;