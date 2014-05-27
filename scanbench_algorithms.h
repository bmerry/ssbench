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
#include "hostutils.h"

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

struct void_vector
{
    void_vector() {}
    explicit void_vector(std::size_t size) {}

    int operator[](std::size_t idx) const
    {
        return 0;
    }

    std::size_t size() const
    {
        return 0;
    }
};

template<typename T>
struct vector_of
{
    typedef std::vector<T> type;
};

template<>
struct vector_of<void>
{
    typedef void_vector type;
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

template<typename K, typename V>
class sort_algorithm : public algorithm
{
public:
    typedef K key_type;
    typedef V value_type;
    typedef typename vector_of<K>::type key_vector;
    typedef typename vector_of<V>::type value_vector;

private:
    key_vector expected_keys;
    value_vector expected_values;
    virtual std::pair<key_vector, value_vector> get() const = 0;

public:
    sort_algorithm(const key_vector &keys, const value_vector &values)
        : expected_keys(keys), expected_values(values)
    {
        sort_by_key(expected_keys, expected_values);
    }

    virtual void validate() const
    {
        std::pair<key_vector, value_vector> out = get();
        const key_vector &keys = out.first;
        const value_vector &values = out.second;
        assert(keys.size() == expected_keys.size());
        assert(values.size() == expected_values.size());
        for (std::size_t i = 0; i < expected_keys.size(); i++)
        {
            if (!check_equal(i, expected_keys[i], keys[i]))
                break;
        }
        for (std::size_t i = 0; i < expected_values.size(); i++)
        {
            if (!check_equal(i, expected_values[i], values[i]))
                break;
        }
    }
};

/* See scanbench_register for the definition. It is declared here so that
 * it can be specialised by CUDA code that cannot include C++11 code.
 */
template<typename A>
struct algorithm_factory;

#endif
