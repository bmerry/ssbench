#ifndef HOSTUTILS_H
#define HOSTUTILS_H

template<typename K, typename V>
void sort_by_key(K &keys, V &values);

template<typename K, typename V>
void parallel_sort_by_key(K &keys, V &values);

#endif /* HOSTUTILS_H */
