#ifndef SCANBENCH_VEX_H
#define SCANBENCH_VEX_H

#include <vector>
#include <string>
#include <memory>

namespace vex
{
    class Context;
}

class vex_algorithm
{
protected:
    std::unique_ptr<vex::Context> ctx;

    vex_algorithm();
    ~vex_algorithm();
public:
    void finish();
};

/************************************************************************/

template<typename T>
class vex_scan : public vex_algorithm
{
protected:
    struct data_t;
    std::unique_ptr<data_t> data;

public:
    vex_scan(const std::vector<T> &h_a);
    ~vex_scan();
    std::string name() const { return "vex::exclusive_scan"; }
    void run();
};

template<typename T>
class vex_clogs_scan : public vex_scan<T>
{
public:
    vex_clogs_scan(const std::vector<T> &h_a) : vex_scan<T>(h_a) {}

    std::string name() const { return "vex::clogs::exclusive_scan"; }
    void run();
};

/************************************************************************/

template<typename T>
class vex_sort : public vex_algorithm
{
protected:
    struct data_t;
    std::unique_ptr<data_t> data;

public:
    vex_sort(const std::vector<T> &h_a);
    ~vex_sort();

    std::string name() const { return "vex::sort"; }
    void run();
};

#endif
