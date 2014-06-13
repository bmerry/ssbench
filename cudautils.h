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

#ifndef SSBENCH_CUDAUTILS_H
#define SSBENCH_CUDAUTILS_H

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

#endif
