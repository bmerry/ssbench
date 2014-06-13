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

#include <utility>
#include <iterator>
#include <algorithm>
#include <cstdint>
#include <vector>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/cstdint.hpp>
#include "hostutils.h"

template<typename I1, typename I2>
class pair_iterator : public boost::iterator_facade<
    pair_iterator<I1, I2>,
    std::pair<typename std::iterator_traits<I1>::value_type, typename std::iterator_traits<I2>::value_type>,
    std::random_access_iterator_tag,
    std::pair<typename std::iterator_traits<I1>::reference, typename std::iterator_traits<I2>::reference>,
    typename std::iterator_traits<I1>::difference_type>
{
public:
    pair_iterator() {}
    pair_iterator(I1 i1, I2 i2) : i1(i1), i2(i2) {}

private:
    void increment() { ++i1; ++i2; }
    void decrement() { --i1; --i2; }
    bool equal(const pair_iterator &other) const { return i1 == other.i1 && i2 == other.i2; }

    std::pair<typename std::iterator_traits<I1>::reference, typename std::iterator_traits<I2>::reference> dereference() const
    {
        return std::pair<typename std::iterator_traits<I1>::reference, typename std::iterator_traits<I2>::reference>(
            *i1, *i2);
    }

    typename std::iterator_traits<I1>::difference_type distance_to(const pair_iterator &other) const
    {
        return other.i1 - i1;
    }

    void advance(typename std::iterator_traits<I1>::difference_type n)
    {
        i1 += n;
        i2 += n;
    }

    I1 i1;
    I2 i2;
    friend class boost::iterator_core_access;
};

namespace std
{

template<typename A1, typename A2, typename B1, typename B2>
void swap(std::pair<A1 &, A2 &> a, std::pair<B1 &, B2 &> b)
{
    swap(a.first, b.first);
    swap(a.second, b.second);
}

}

template<typename T1, typename T2, typename Cmp>
class PairCmp
{
private:
    Cmp c;
public:
    typedef bool result_type;

    explicit PairCmp(const Cmp &c = Cmp()) : c(c) {}

    bool operator()(const std::pair<T1, T2> &a, const std::pair<T1, T2> &b)
    {
        return c(a.first, b.first);
    }
};

template<typename I1, typename I2, typename Cmp = std::less<typename std::iterator_traits<I1>::value_type> >
static void sort_by_key(I1 key_first, I1 key_last, I2 value_first, Cmp cmp = Cmp())
{
    pair_iterator<I1, I2> first(key_first, value_first);
    pair_iterator<I1, I2> last(key_last, value_first + (key_last - key_first));
    std::stable_sort(first, last,
        PairCmp<typename std::iterator_traits<I1>::value_type,
                typename std::iterator_traits<I2>::value_type,
                Cmp>(cmp));
}

template<typename K, typename V>
void sort_by_key(K &keys, V &values)
{
    sort_by_key(keys.begin(), keys.end(), values.begin());
}

#include <parallel/algorithm>

template<typename I1, typename I2, typename Cmp = std::less<typename std::iterator_traits<I1>::value_type> >
static void parallel_sort_by_key(I1 key_first, I1 key_last, I2 value_first, Cmp cmp = Cmp())
{
    pair_iterator<I1, I2> first(key_first, value_first);
    pair_iterator<I1, I2> last(key_last, value_first + (key_last - key_first));
    __gnu_parallel::sort(first, last,
        PairCmp<typename std::iterator_traits<I1>::value_type,
                typename std::iterator_traits<I2>::value_type,
                Cmp>(cmp));
}

template<typename K, typename V>
void parallel_sort_by_key(K &keys, V &values)
{
    parallel_sort_by_key(keys.begin(), keys.end(), values.begin());
}

template void sort_by_key<std::vector<boost::uint32_t>, std::vector<boost::uint32_t> >(std::vector<boost::uint32_t> &, std::vector<boost::uint32_t> &);
template void parallel_sort_by_key<std::vector<boost::uint32_t>, std::vector<boost::uint32_t> >(std::vector<boost::uint32_t> &, std::vector<boost::uint32_t> &);
