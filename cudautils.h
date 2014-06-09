#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#include <sstream>
#include <stdexcept>

#define CUDA_CHECK(expr) (cuda_check(expr, __FILE__, __LINE__))

struct cuda_error : public std::runtime_error
{
public:
    cuda_error(const std::string &msg) : std::runtime_error(msg) {}
};

static inline void cuda_check(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        std::ostringstream msg;
        msg << file << ":" << line << ": " << cudaGetErrorString(err);
        throw cuda_error(msg.str());
    }
}

static inline void cuda_set_device(device_info d)
{
    int devices;
    CUDA_CHECK( cudaGetDeviceCount(&devices) );
    if (d.type != DEVICE_TYPE_GPU || d.index >= devices)
        throw device_not_supported();
    CUDA_CHECK( cudaSetDevice(d.index) );
}

#endif /* CUDAUTILS_H */
