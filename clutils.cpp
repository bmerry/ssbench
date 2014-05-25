#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>
#undef CL_VERSION_1_2
#include <CL/cl.hpp>
#include "clutils.h"

cl_device_type type_to_cl_type(device_type d)
{
    switch (d)
    {
    case DEVICE_TYPE_CPU: return CL_DEVICE_TYPE_CPU;
    case DEVICE_TYPE_GPU: return CL_DEVICE_TYPE_GPU;
    default:
        // should never be reached
        throw std::runtime_error("Illegal device type");
    }
}

cl::Device device_from_type(device_type d)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (const cl::Platform &platform : platforms)
    {
        std::vector<cl::Device> devices;
        try
        {
            platform.getDevices(type_to_cl_type(d), &devices);
        }
        catch (cl::Error &e)
        {
            if (e.err() != CL_DEVICE_NOT_FOUND)
                throw;
        }
        if (!devices.empty())
            return devices[0];
    }
    throw device_not_supported();
}
