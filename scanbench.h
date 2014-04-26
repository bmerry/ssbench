#ifndef SCANBENCH_H
#define SCANBENCH_H

#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <utility>
#include <cassert>
#include <iostream>

class algorithm
{
public:
    virtual std::string api() const = 0;
    virtual std::string algorithm_name() const = 0;
    virtual std::string name() const = 0;
    virtual void run() = 0;
    virtual void finish() = 0;
    virtual void validate() const = 0;
    ~algorithm() {}
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
class scan_algorithm : public virtual algorithm
{
private:
    std::vector<T> expected;

    virtual std::vector<T> get() const = 0;
public:
    typedef T value_type;

    scan_algorithm(const std::vector<T> &in)
    {
        std::vector<T> expected(in.size());
        std::partial_sum(in.begin(), in.end() - 1, expected.begin() + 1);
        expected[0] = T();
    }

    virtual std::string algorithm_name() const override { return "scan"; }

    virtual void validate() const override
    {
        std::vector<T> out = get();
        assert(out.size() == expected.size());
        for (std::size_t i = 0; i < expected.size(); i++)
            if (!check_equal(i, expected[i], out[i]))
                break;
    }
};

template<typename T>
class sort_algorithm : public virtual algorithm
{
private:
    std::vector<T> expected;
    virtual std::vector<T> get() const = 0;
public:
    typedef T value_type;

    sort_algorithm(std::vector<T> in)
        : expected(std::move(in))
    {
        std::sort(expected.begin(), expected.end());
    }

    virtual std::string algorithm_name() const override { return "sort"; }

    virtual void validate() const override
    {
        std::vector<T> out = get();
        assert(out.size() == expected.size());
        for (std::size_t i = 0; i < expected.size(); i++)
            if (!check_equal(i, expected[i], out[i]))
                break;
    }
};

template<typename T, typename A, typename... Args>
class registry
{
public:
    typedef std::function<std::unique_ptr<A>(Args...)> factory;

    static void add_factory(factory f) { factories.push_back(std::move(f)); }

    template<typename S>
    static void add_class()
    {
        auto f = [](Args... in) -> std::unique_ptr<A>
        {
            return new S(in...);
        };
        add_factory(f);
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
