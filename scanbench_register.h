#ifndef SCANBENCH_REGISTER_H
#define SCANBENCH_REGISTER_H

#include <memory>
#include <functional>
#include <utility>
#include "scanbench_algorithms.h"

/* Creates a new instance of class A. Note that CUDA-based APIs provide
 * specialisations because CUDA does not support C++11.
 */
template<typename A>
struct algorithm_factory
{
    // Throws
    template<typename... Args>
    static A *create(Args&&... args)
    {
        return new A(std::forward<Args>(args)...);
    }

    static std::string name() { return A::name(); }
    static std::string api()  { return A::api(); }
};

template<typename A, typename... Args>
class registry
{
public:
    struct entry
    {
        std::function<std::unique_ptr<A>(device_type, Args...)> factory;
        std::string name;
        std::string api;
    };

    template<typename S>
    static void add_class()
    {
        entry e;
        e.factory = [](device_type d, Args... in) -> std::unique_ptr<A>
        {
            return std::unique_ptr<A>(algorithm_factory<S>::create(d, in...));
        };
        e.name = algorithm_factory<S>::name();
        e.api = algorithm_factory<S>::api();
        entries.push_back(std::move(e));
    }

    static const std::vector<entry> &get() { return entries; }

private:
    static std::vector<entry> entries;
};

template<typename A, typename... Args>
std::vector<typename registry<A, Args...>::entry> registry<A, Args...>::entries;

template<typename T>
using scan_registry = registry<scan_algorithm<T>, const std::vector<T> &>;
template<typename K, typename V>
using sort_registry = registry<sort_algorithm<K, V>,
      const typename vector_of<K>::type &,
      const typename vector_of<V>::type &>;

template<template<typename T> class A>
class register_scan_algorithm
{
public:
    register_scan_algorithm()
    {
        scan_registry<std::int32_t>::add_class<A<std::int32_t>>();
    }
};

template<template<typename K, typename V> class A>
class register_sort_algorithm
{
public:
    register_sort_algorithm()
    {
        sort_registry<std::uint32_t, void>::add_class<A<std::uint32_t, void>>();
        sort_registry<std::uint32_t, std::uint32_t>::add_class<A<std::uint32_t, std::uint32_t>>();
    }
};

#endif
