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
};

template<typename T, typename A, typename... Args>
class registry
{
public:
    typedef std::function<A *(Args...)> factory;

    template<typename S>
    static void add_class()
    {
        auto f = [](Args... in) -> A*
        {
            return algorithm_factory<S>::create(in...);
        };
        factories.push_back(std::move(f));
    }

    static const std::vector<factory> &get() { return factories; }

private:
    static std::vector<factory> factories;
};

template<typename T, typename A, typename... Args>
std::vector<typename registry<T, A, Args...>::factory> registry<T, A, Args...>::factories;

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
