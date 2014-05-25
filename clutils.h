#ifndef CLUTILS_H
#define CLUTILS_H

#include <CL/cl.hpp>
#include <vector>
#include "scanbench_algorithms.h"

cl_device_type type_to_cl_type(device_type d);
cl::Device device_from_type(device_type d);

#endif
