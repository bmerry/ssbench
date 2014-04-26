#ifndef SCANBENCH_CPU_H
#define SCANBENCH_CPU_H

#include <vector>
#include <string>
#include "scanbench.h"

template<typename T>
class cpu_scan : public scan_algorithm<T>
{
protected:
    std::vector<T> a;
    std::vector<T> out;
public:
    cpu_scan(const std::vector<T> &h_a) : scan_algorithm<T>(h_a), a(h_a), out(h_a.size()) {}
    virtual std::string api() const override { return "cpu"; }
    virtual void finish() override {}
    virtual std::vector<T> get() const override { return out; }
};

template<typename T>
class serial_scan : public cpu_scan<T>
{
public:
    using cpu_scan<T>::cpu_scan;
    std::string name() const { return "serial scan"; }
    virtual void run() override;
};

template<typename T>
class parallel_scan : public cpu_scan<T>
{
public:
    using cpu_scan<T>::cpu_scan;

    std::string name() const { return "parallel scan"; }
    virtual void run() override;
};

template<typename T>
class my_parallel_scan : public cpu_scan<T>
{
public:
    using cpu_scan<T>::cpu_scan;

    std::string name() const { return "my parallel scan"; }
    virtual void run() override;
};

/************************************************************************/

template<typename T>
class cpu_sort
{
protected:
    std::vector<T> a;
    std::vector<T> target;

public:
    cpu_sort(const std::vector<T> &h_a)
        : a(h_a), target(h_a.size())
    {
    }

    void finish() {}
    std::vector<T> get() const { return target; }
};

template<typename T>
class serial_sort : public cpu_sort<T>
{
public:
    using cpu_sort<T>::cpu_sort;

    std::string name() const { return "serial sort"; }
    void run();
};

template<typename T>
class parallel_sort : public cpu_sort<T>
{
public:
    using cpu_sort<T>::cpu_sort;

    std::string name() const { return "parallel sort"; }
    void run();
};


#endif
