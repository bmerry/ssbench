#ifndef SCANBENCH_COMPUTE_H
#define SCANBENCH_COMPUTE_H

#include <vector>
#include <string>
#include <memory>

namespace boost
{
namespace compute
{
    class command_queue;
}
}

class compute_algorithm
{
protected:
    std::unique_ptr<boost::compute::command_queue> queue;

    compute_algorithm();
    ~compute_algorithm();
public:
    void finish();
};

template<typename T>
class compute_scan : public compute_algorithm
{
private:
    struct data_t;
    std::unique_ptr<data_t> data;
public:
    compute_scan(const std::vector<T> &h_a);
    ~compute_scan();
    std::string name() const { return "compute::exclusive_scan"; }
    void run();
    std::vector<T> get() const;
};

template<typename T>
class compute_sort : public compute_algorithm
{
private:
    struct data_t;
    std::unique_ptr<data_t> data;
public:
    compute_sort(const std::vector<T> &h_a);
    ~compute_sort();
    std::string name() const { return "compute::sort"; }
    void run();
    std::vector<T> get() const;
};
#endif
