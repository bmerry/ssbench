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

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <vector>
#include <string>
#include "algorithms.h"
#include "register.h"
#include "cudautils.h"

class thrust_algorithm
{
public:
    template<typename T>
    struct types
    {
        typedef thrust::device_vector<T> vector;
        typedef vector scan_vector;
        typedef vector sort_vector;
    };

    template<typename T>
    static void create(thrust::device_vector<T> &out, std::size_t elements)
    {
        out.resize(elements);
    }

    template<typename T>
    static void copy(const std::vector<T> &src, thrust::device_vector<T> &dst)
    {
        thrust::copy(src.begin(), src.end(), dst.begin());
    }

    template<typename T>
    static void copy(const thrust::device_vector<T> &src, thrust::device_vector<T> &dst)
    {
        thrust::copy(src.begin(), src.end(), dst.begin());
    }

    template<typename T>
    static void copy(const thrust::device_vector<T> &src, std::vector<T> &dst)
    {
        thrust::copy(src.begin(), src.end(), dst.begin());
    }

    template<typename T>
    static void pre_scan(const thrust::device_vector<T> &src, thrust::device_vector<T> &dst)
    {
    }

    template<typename T>
    static void scan(const thrust::device_vector<T> &src, thrust::device_vector<T> &dst)
    {
        thrust::exclusive_scan(src.begin(), src.end(), dst.begin());
    }

    template<typename K, typename V>
    static void pre_sort_by_key(thrust::device_vector<K> &keys, thrust::device_vector<V> &values) {}

    template<typename K, typename V>
    static void sort_by_key(thrust::device_vector<K> &keys, thrust::device_vector<V> &values)
    {
        thrust::sort_by_key(keys.begin(), keys.end(), values.begin());
    }

    template<typename T>
    static void pre_sort(thrust::device_vector<T> &keys) {}

    template<typename T>
    static void sort(thrust::device_vector<T> &keys)
    {
        thrust::sort(keys.begin(), keys.end());
    }

    static void finish()
    {
        CUDA_CHECK( cudaDeviceSynchronize() );
    }

    static std::string api() { return "thrust"; }

    explicit thrust_algorithm(device_info d)
    {
        cuda_set_device(d);
    }
};

static register_algorithms<thrust_algorithm> register_thrust;
