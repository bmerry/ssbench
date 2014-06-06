#ifndef SCANBENCH_REGISTER_H
#define SCANBENCH_REGISTER_H

#include <memory>
#include <functional>
#include <utility>
#include "algorithms.h"

/* Creates a new instance of class A. Note that CUDA-based APIs provide
 * specialisations because CUDA does not support C++11.
 */
template<typename A>
struct algorithm_factory
{
    // Throws
    template<typename... Args>
    static A *create(device_type d, Args&&... args)
    {
        return new A(d, std::forward<Args>(args)...);
    }

    static std::string api()  { return A::api(); }
};

template<typename Tag, typename... Args>
class registry
{
public:
    struct entry
    {
        std::function<std::unique_ptr<algorithm>(device_type, Args...)> factory;
        std::string api;
    };

    template<typename S>
    static void add_class()
    {
        entry e;
        e.factory = [](device_type d, Args... in) -> std::unique_ptr<algorithm>
        {
            return std::unique_ptr<algorithm>(algorithm_factory<S>::create(d, in...));
        };
        e.api = algorithm_factory<S>::api();
        entries.push_back(std::move(e));
    }

    static const std::vector<entry> &get() { return entries; }

private:
    static std::vector<entry> entries;
};

template<typename Tag, typename... Args>
std::vector<typename registry<Tag, Args...>::entry> registry<Tag, Args...>::entries;

struct scan_tag {};
struct sort_tag {};
struct sort_by_key_tag {};

template<typename T>
using scan_registry = registry<scan_tag, const std::vector<T> &>;
template<typename K>
using sort_registry = registry<sort_tag, const std::vector<K> &>;
template<typename K, typename V>
using sort_by_key_registry = registry<sort_by_key_tag,
      const std::vector<K> &,
      const std::vector<V> &>;

template<typename A>
class register_scan_algorithm
{
public:
    register_scan_algorithm()
    {
        scan_registry<std::int32_t>::add_class<scan_algorithm<std::int32_t, A> >();
    }
};

template<typename A>
class register_sort_algorithm
{
public:
    register_sort_algorithm()
    {
        sort_registry<std::uint32_t>::add_class<sort_algorithm<std::uint32_t, A> >();
    }
};

template<typename A>
class register_sort_by_key_algorithm
{
public:
    register_sort_by_key_algorithm()
    {
        sort_by_key_registry<std::uint32_t, std::uint32_t>::add_class<
            sort_by_key_algorithm<std::uint32_t, std::uint32_t, A> >();
    }
};

template<typename A>
class register_algorithms
{
private:
    register_scan_algorithm<A> scan;
    register_sort_algorithm<A> sort;
    register_sort_by_key_algorithm<A> sort_by_key;
};

#endif
