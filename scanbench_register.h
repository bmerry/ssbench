#ifndef SCANBENCH_REGISTER_H
#define SCANBENCH_REGISTER_H

#include <memory>
#include <functional>
#include <utility>
#include "scanbench_algorithms.h"

/* Creates a new instance of class A. It is specialised for Thrust because
 * the thrust algorithm classes are incomplete at the point of instantiation.
 */
template<typename A>
struct algorithm_factory
{
    template<typename... Args>
    static A *create(Args&&... args)
    {
        return new A(std::forward<Args>(args)...);
    }

    static std::string name() { return A::name(); }
    static std::string api()  { return A::api(); }
};

template<typename T, typename A, typename... Args>
class registry
{
public:
    struct entry
    {
        std::function<std::unique_ptr<A>(Args...)> factory;
        std::string name;
        std::string api;
    };

    template<typename S>
    static void add_class()
    {
        entry e;
        e.factory = [](Args... in) -> std::unique_ptr<A>
        {
            return std::unique_ptr<A>(algorithm_factory<S>::create(in...));
        };
        e.name = algorithm_factory<S>::name();
        e.api = algorithm_factory<S>::api();
        entries.push_back(std::move(e));
    }

    static const std::vector<entry> &get() { return entries; }

private:
    static std::vector<entry> entries;
};

template<typename T, typename A, typename... Args>
std::vector<typename registry<T, A, Args...>::entry> registry<T, A, Args...>::entries;

template<typename T>
using scan_registry = registry<T, scan_algorithm<T>, const std::vector<T> &>;
template<typename T>
using sort_registry = registry<T, sort_algorithm<T>, const std::vector<T> &>;

template<template<typename T> class A>
class register_scan_algorithm
{
public:
    register_scan_algorithm()
    {
        scan_registry<std::int32_t>::add_class<A<std::int32_t>>();
    }
};

template<template<typename T> class A>
class register_sort_algorithm
{
public:
    register_sort_algorithm()
    {
        sort_registry<std::uint32_t>::add_class<A<std::uint32_t>>();
    }
};

#endif
