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

template<typename T, typename A>
class scan_algorithm : public algorithm
{
private:
    std::vector<T> expected;
    A impl;
    typename A::template types<T>::vector d_a;
    typename A::template types<T>::scan_vector d_scan;

public:
    typedef T value_type;

    scan_algorithm(device_type d, const std::vector<T> &in)
        : expected(in.size()), impl(d)
    {
        impl.create(d_a, in.size());
        impl.create(d_scan, in.size());
        impl.copy(in, d_a);
        impl.pre_scan(d_a, d_scan);

        std::partial_sum(in.begin(), in.end() - 1, expected.begin() + 1);
        expected[0] = T();
    }

    virtual void run()
    {
        impl.scan(d_a, d_scan);
    }

    virtual void finish() { impl.finish(); }
    static std::string api() { return A::api(); }

    virtual void validate() const
    {
        std::vector<T> out(expected.size());
        impl.copy(d_scan, out);
        assert(out.size() == expected.size());
        for (std::size_t i = 0; i < expected.size(); i++)
            if (!check_equal(i, expected[i], out[i]))
                break;
    }
};

template<typename K, typename V, typename A>
class sort_by_key_algorithm : public algorithm
{
public:
    typedef K key_type;
    typedef V value_type;

private:
    std::vector<K> expected_keys;
    std::vector<V> expected_values;
    A impl;
    typename A::template types<K>::vector d_keys;
    typename A::template types<V>::vector d_values;
    typename A::template types<K>::sort_vector d_sorted_keys;
    typename A::template types<V>::sort_vector d_sorted_values;

public:
    sort_by_key_algorithm(device_type d, const std::vector<K> &keys, const std::vector<V> &values)
        : expected_keys(keys), expected_values(values), impl(d)
    {
        assert(keys.size() == values.size());
        std::size_t N = keys.size();
        impl.create(d_keys, N);
        impl.create(d_values, N);
        impl.create(d_sorted_keys, N);
        impl.create(d_sorted_values, N);
        impl.copy(keys, d_keys);
        impl.copy(values, d_values);
        impl.pre_sort_by_key(d_sorted_keys, d_sorted_values);

        sort_by_key(expected_keys, expected_values);
    }

    virtual void run()
    {
        impl.copy(d_keys, d_sorted_keys);
        impl.copy(d_values, d_sorted_values);
        impl.sort_by_key(d_sorted_keys, d_sorted_values);
    }

    virtual void finish() { impl.finish(); }
    static std::string api() { return A::api(); }

    virtual void validate() const
    {
        std::vector<K> keys(expected_keys.size());
        std::vector<V> values(expected_values.size());
        impl.copy(d_sorted_keys, keys);
        impl.copy(d_sorted_values, values);
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

template<typename K, typename A>
class sort_algorithm : public algorithm
{
public:
    typedef K key_type;

private:
    std::vector<K> expected_keys;
    A impl;
    typename A::template types<K>::vector d_keys;
    typename A::template types<K>::sort_vector d_sorted_keys;

public:
    sort_algorithm(device_type d, const std::vector<K> &keys)
        : expected_keys(keys), impl(d)
    {
        std::size_t N = keys.size();
        impl.create(d_keys, N);
        impl.create(d_sorted_keys, N);
        impl.copy(keys, d_keys);
        impl.pre_sort(d_sorted_keys);

        std::sort(expected_keys.begin(), expected_keys.end());
    }

    virtual void run()
    {
        impl.copy(d_keys, d_sorted_keys);
        impl.sort(d_sorted_keys);
    }

    virtual void finish() { impl.finish(); }
    static std::string api() { return A::api(); }

    virtual void validate() const
    {
        std::vector<K> keys(expected_keys.size());
        impl.copy(d_sorted_keys, keys);
        assert(keys.size() == expected_keys.size());

        for (std::size_t i = 0; i < expected_keys.size(); i++)
        {
            if (!check_equal(i, expected_keys[i], keys[i]))
                break;
        }
    }
};

/* See register for the definition. It is declared here so that
 * it can be specialised by CUDA code that cannot include C++11 code.
 */
template<typename A>
struct algorithm_factory;

#endif
