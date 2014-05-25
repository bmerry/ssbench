/* Note: this file must not use C++11 features, because CUDA does not support it.
 */

#ifndef SCANBENCH_ALGORITHMS_H
#define SCANBENCH_ALGORITHMS_H

#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <utility>
#include <cassert>
#include <iostream>

enum device_type
{
    DEVICE_TYPE_CPU,
    DEVICE_TYPE_GPU
};

/* Exception class thrown by constructors to indicate that the library does not
 * support the given device. This is a slight abuse of exceptions (it is not
 * really exceptional), but is used because constructors cannot just return a
 * failure code.
 */
class device_not_supported
{
};

class algorithm
{
public:
    virtual void run() = 0;
    virtual void finish() = 0;
    virtual void validate() const = 0;
    virtual ~algorithm() {}
};

template<typename t>
static inline bool check_equal(std::size_t idx, const t &a, const t &b)
{
    if (a != b)
    {
        std::cerr << idx << ": expected " << a << " but found " << b << '\n';
        return false;
    }
    return true;
}

template<typename T>
class scan_algorithm : public algorithm
{
private:
    std::vector<T> expected;

    virtual std::vector<T> get() const = 0;
public:
    typedef T value_type;

    scan_algorithm(const std::vector<T> &in)
        : expected(in.size())
    {
        std::partial_sum(in.begin(), in.end() - 1, expected.begin() + 1);
        expected[0] = T();
    }

    virtual void validate() const
    {
        std::vector<T> out = get();
        assert(out.size() == expected.size());
        for (std::size_t i = 0; i < expected.size(); i++)
            if (!check_equal(i, expected[i], out[i]))
                break;
    }
};

template<typename T>
class sort_algorithm : public algorithm
{
private:
    std::vector<T> expected;
    virtual std::vector<T> get() const = 0;
public:
    typedef T value_type;

    sort_algorithm(const std::vector<T> &in)
        : expected(in)
    {
        std::sort(expected.begin(), expected.end());
    }

    virtual void validate() const
    {
        std::vector<T> out = get();
        assert(out.size() == expected.size());
        for (std::size_t i = 0; i < expected.size(); i++)
            if (!check_equal(i, expected[i], out[i]))
                break;
    }
};

/* See scanbench_register for the definition. It is declared here so that
 * it can be specialised by CUDA code that cannot include C++11 code.
 */
template<typename A>
struct algorithm_factory;

#endif
