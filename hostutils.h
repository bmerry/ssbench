#ifndef HOSTUTILS_H
#define HOSTUTILS_H

struct void_vector;

template<typename K, typename V>
void sort_by_key(K &keys, V &values);

template<typename K>
void sort_by_key(K &keys, void_vector &values);

template<typename K, typename V>
void parallel_sort_by_key(K &keys, V &values);

template<typename K>
void parallel_sort_by_key(K &keys, void_vector &values);

#endif /* HOSTUTILS_H */
