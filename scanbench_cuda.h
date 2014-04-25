#ifndef SCANBENCH_CUDA_H
#define SCANBENCH_CUDA_H

#include <vector>
#include <string>

template<typename T>
class thrust_scan
{
private:
    struct data_t;

    data_t *data;
public:
    thrust_scan(const std::vector<T> &h_a);

    std::string name() const { return "thrust::exclusive_scan"; }
    void run();
    void finish();
    std::vector<T> get() const;
    ~thrust_scan();
};

template<typename T>
class thrust_sort
{
private:
    struct data_t;

    data_t *data;
public:
    thrust_sort(const std::vector<T> &h_a);

    std::string name() const { return "thrust::sort"; }
    void run();
    void finish();
    std::vector<T> get() const;
    ~thrust_sort();
};

#endif
