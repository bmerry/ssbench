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

#include <vector>
#include <string>
#include <cstddef>
#include "algorithms.h"
#include "register.h"
#include "moderngpu.cuh"
#include "cudautils.h"

class mgpu_algorithm
{
private:
    mgpu::ContextPtr ctx;

public:
    template<typename T>
    struct types
    {
        typedef MGPU_MEM(T) vector;
        typedef vector scan_vector;
        typedef vector sort_vector;
    };

    template<typename T>
    void create(MGPU_MEM(T) &out, std::size_t elements) const
    {
        ctx->Malloc<T>(elements).swap(out);
    }

    template<typename T>
    static void copy(const std::vector<T> &src, MGPU_MEM(T) &dst)
    {
        dst->FromHost(src);
    }

    template<typename T>
    static void copy(const MGPU_MEM(T) &src, MGPU_MEM(T) &dst)
    {
        src->ToDevice(dst->get(), dst->Size());
    }

    template<typename T>
    static void copy(const MGPU_MEM(T) &src, std::vector<T> &dst)
    {
        src->ToHost(dst);
    }

    template<typename T>
    static void pre_scan(const MGPU_MEM(T) &src, MGPU_MEM(T) &dst)
    {
    }

    template<typename T>
    void scan(const MGPU_MEM(T) &src, MGPU_MEM(T) &dst) const
    {
        mgpu::Scan<mgpu::MgpuScanTypeExc>(
            src->get(), src->Size(), T(0), mgpu::plus<T>(), (T *) NULL, (T *) NULL, dst->get(), *ctx);
    }

    template<typename K>
    static void pre_sort(MGPU_MEM(K) &keys)
    {
    }

    template<typename K>
    void sort(MGPU_MEM(K) &keys) const
    {
        mgpu::MergesortKeys(keys->get(), keys->Size(), mgpu::less<K>(), *ctx);
    }

    template<typename K, typename V>
    static void pre_sort_by_key(MGPU_MEM(K) &keys, MGPU_MEM(V) &values)
    {
    }

    template<typename K, typename V>
    void sort_by_key(MGPU_MEM(K) &keys, MGPU_MEM(V) &values) const
    {
        mgpu::MergesortPairs(keys->get(), values->get(), keys->Size(), mgpu::less<K>(), *ctx);
    }

    static void finish()
    {
        CUDA_CHECK( cudaDeviceSynchronize() );
    }

    static std::string api() { return "mgpu"; }

    explicit mgpu_algorithm(device_info d)
    {
        if (d.type != DEVICE_TYPE_GPU || d.index >= mgpu::CudaDevice::DeviceCount())
            throw device_not_supported();
        ctx = mgpu::CreateCudaDevice(d.index);
    }
};

static register_algorithms<mgpu_algorithm> register_mgpu;
