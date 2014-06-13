/* ssbench: benchmarking of sort and scan libraries
 * Copyright (C) 2014  Bruce Merry
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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

cl::Device device_from_info(device_info d)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    int index = d.index;
    for (const cl::Platform &platform : platforms)
    {
        std::vector<cl::Device> devices;
        try
        {
            platform.getDevices(type_to_cl_type(d.type), &devices);
        }
        catch (cl::Error &e)
        {
            if (e.err() != CL_DEVICE_NOT_FOUND)
                throw;
        }
        if (index < (int) devices.size())
            return devices[index];
        else
            index -= devices.size();
    }
    throw device_not_supported();
}
