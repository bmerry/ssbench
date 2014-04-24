#ifndef SCANBENCH_CLOGS_H
#define SCANBENCH_CLOGS_H

#include <vector>
#include <string>
#include <memory>

class clogs_algorithm
{
protected:
    struct resources_t;
    std::unique_ptr<resources_t> resources;

    clogs_algorithm();
    ~clogs_algorithm();
public:
    void finish();
};

template<typename T>
class clogs_scan : public clogs_algorithm
{
private:
    struct data_t;
    std::unique_ptr<data_t> data;

public:
    clogs_scan(const std::vector<T> &h_a);
    ~clogs_scan();
    std::string name() const { return "clogs::Scan"; }
    void run();
};

template<typename T>
class clogs_sort : public clogs_algorithm
{
private:
    struct data_t;
    std::unique_ptr<data_t> data;

public:
    clogs_sort(const std::vector<T> &h_a);
    ~clogs_sort();
    std::string name() const { return "clogs::Radixsort"; }
    void run();
};

#endif
