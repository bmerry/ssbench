#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <vector>
#include <string>
#include "scanbench_algorithms.h"
#include "scanbench_thrust.h"

template<typename T>
struct thrust_traits
{
    typedef thrust::device_vector<T> vector;

    static std::vector<T> get(const vector &v)
    {
        std::vector<T> ans(v.size());
        thrust::copy(v.begin(), v.end(), ans.begin());
        return ans;
    }

    template<typename K>
    static void sort(thrust::device_vector<K> &keys, vector &values)
    {
        thrust::sort_by_key(keys.begin(), keys.end(), values.begin());
    }
};

template<>
struct thrust_traits<void>
{
    struct vector
    {
        vector() {}
        explicit vector(const void_vector &) {}
        explicit vector(std::size_t) {}
        std::size_t size() const { return 0; }
    };

    static void_vector get(const vector &v)
    {
        return void_vector();
    }

    template<typename K>
    static void sort(thrust::device_vector<K> &keys, vector &values)
    {
        thrust::sort(keys.begin(), keys.end());
    }
};

/********************************************************************/

template<typename T>
class thrust_scan : public scan_algorithm<T>
{
private:
    thrust::device_vector<T> d_a;
    thrust::device_vector<T> d_scan;

public:
    thrust_scan(device_type d, const std::vector<T> &h_a)
        : scan_algorithm<T>(h_a), d_a(h_a), d_scan(h_a.size())
    {
        if (d != DEVICE_TYPE_GPU)
            throw device_not_supported();
    }

    static std::string name() { return "thrust::exclusive_scan"; }
    static std::string api() { return "thrust"; }
    virtual void finish() { cudaDeviceSynchronize(); }

    virtual void run()
    {
        thrust::exclusive_scan(d_a.begin(), d_a.end(), d_scan.begin());
    }

    virtual std::vector<T> get() const
    {
        std::vector<T> ans(d_scan.size());
        thrust::copy(d_scan.begin(), d_scan.end(), ans.begin());
        return ans;
    }
};

template<typename T>
scan_algorithm<T> *algorithm_factory<thrust_scan<T> >::create(device_type d, const std::vector<T> &h_a)
{
    return new thrust_scan<T>(d, h_a);
}

template<typename T>
std::string algorithm_factory<thrust_scan<T> >::name() { return thrust_scan<T>::name(); }

template<typename T>
std::string algorithm_factory<thrust_scan<T> >::api() { return thrust_scan<T>::api(); }

template struct algorithm_factory<thrust_scan<int> >;

/********************************************************************/

template<typename K, typename V>
class thrust_sort : public sort_algorithm<K, V>
{
private:
    typedef typename vector_of<K>::type key_vector;
    typedef typename vector_of<V>::type value_vector;
    typedef typename thrust_traits<K>::vector d_key_vector;
    typedef typename thrust_traits<V>::vector d_value_vector;

    d_key_vector d_keys, d_sorted_keys;
    d_value_vector d_values, d_sorted_values;

public:
    thrust_sort(device_type d, const key_vector &h_keys, const value_vector &h_values)
        : sort_algorithm<K, V>(h_keys, h_values),
        d_keys(h_keys), d_sorted_keys(h_keys.size()),
        d_values(h_values), d_sorted_values(h_values.size())
    {
        if (d != DEVICE_TYPE_GPU)
            throw device_not_supported();
    }

    static std::string name() { return "thrust::sort"; }
    static std::string api() { return "thrust"; }
    virtual void finish() { cudaDeviceSynchronize(); }

    virtual void run()
    {
        d_sorted_keys = d_keys;
        d_sorted_values = d_values;
        thrust_traits<V>::template sort<K>(d_sorted_keys, d_sorted_values);
    }

    virtual std::pair<key_vector, value_vector> get() const
    {
        return std::make_pair(
            thrust_traits<K>::get(d_sorted_keys),
            thrust_traits<V>::get(d_sorted_values));
    }
};

template<typename K, typename V>
sort_algorithm<K, V> *algorithm_factory<thrust_sort<K, V> >::create(
    device_type d,
    const typename vector_of<K>::type h_keys,
    const typename vector_of<V>::type h_values)
{
    return new thrust_sort<K, V>(d, h_keys, h_values);
}

template<typename K, typename V>
std::string algorithm_factory<thrust_sort<K, V> >::name() { return thrust_sort<K, V>::name(); }

template<typename K, typename V>
std::string algorithm_factory<thrust_sort<K, V> >::api() { return thrust_sort<K, V>::api(); }

template struct algorithm_factory<thrust_sort<unsigned int, void> >;
template struct algorithm_factory<thrust_sort<unsigned int, unsigned int> >;
