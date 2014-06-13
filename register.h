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

/* Note: this file must not use C++11 features, because CUDA does not support it.
 */

#ifndef SSBENCH_REGISTER_H
#define SSBENCH_REGISTER_H

#include <memory>
#include <functional>
#include <utility>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/cstdint.hpp>
#include "algorithms.h"

class entry_base
{
public:
    virtual ~entry_base() {}
    virtual std::string api() const = 0;
};

template<typename T>
struct scan_entry
{
    class base_type : public entry_base
    {
    public:
        virtual algorithm *create(device_info d, const std::vector<T> &values) const = 0;
    };

    template<typename A>
    class type : public base_type
    {
    public:
        virtual algorithm *create(device_info d, const std::vector<T> &values) const
        {
            return new scan_algorithm<T, A>(d, values);
        }

        virtual std::string api() const
        {
            return A::api();
        }
    };
};

template<typename K>
struct sort_entry
{
    class base_type : public entry_base
    {
    public:
        virtual algorithm *create(device_info d, const std::vector<K> &keys) const = 0;
    };

    template<typename A>
    class type : public base_type
    {
    public:
        virtual algorithm *create(device_info d, const std::vector<K> &keys) const
        {
            return new sort_algorithm<K, A>(d, keys);
        }

        virtual std::string api() const
        {
            return A::api();
        }
    };
};

template<typename K, typename V>
struct sort_by_key_entry
{

    class base_type : public entry_base
    {
    public:
        virtual algorithm *create(device_info d, const std::vector<K> &keys, const std::vector<V> &values) const = 0;
    };

    template<typename A>
    class type : public base_type
    {
    public:
        virtual algorithm *create(device_info d, const std::vector<K> &keys, const std::vector<V> &values) const
        {
            return new sort_by_key_algorithm<K, V, A>(d, keys, values);
        }

        virtual std::string api() const
        {
            return A::api();
        }
    };
};

template<typename Entry>
class registry
{
public:
    template<typename S>
    static void add_class()
    {
        typedef typename Entry::template type<S> T;
        entries.push_back(new T);
    }

    static const boost::ptr_vector<typename Entry::base_type> &get() { return entries; }

private:
    static boost::ptr_vector<typename Entry::base_type> entries;
};

template<typename Entry>
boost::ptr_vector<typename Entry::base_type> registry<Entry>::entries;

template<typename A>
class register_scan_algorithm
{
public:
    register_scan_algorithm()
    {
        registry<scan_entry<boost::int32_t> >::add_class<A>();
    }
};

template<typename A>
class register_sort_algorithm
{
public:
    register_sort_algorithm()
    {
        registry<sort_entry<boost::uint32_t> >::add_class<A>();
    }
};

template<typename A>
class register_sort_by_key_algorithm
{
public:
    register_sort_by_key_algorithm()
    {
        registry<sort_by_key_entry<boost::uint32_t, boost::uint32_t> >::add_class<A>();
    }
};

template<typename A>
class register_algorithms
{
private:
    register_scan_algorithm<A> scan;
    register_sort_algorithm<A> sort;
    register_sort_by_key_algorithm<A> sort_by_key;
};

#endif
