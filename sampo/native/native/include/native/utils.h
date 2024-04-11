#ifndef SAMPO_UTILS_H
#define SAMPO_UTILS_H

#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

#include "native/schemas/chromosome.h"

using namespace std;

// TODO Think about make own more cache-friendly shuffle using areas around
// target element
vector<int> sample_ind(int n, float prob, random_device &rd);

template <typename T>
vector<T *> sample(vector<T *> &src, float prob, random_device &rd, bool copy = true) {
    auto indexes = sample_ind(src.size(), prob, rd);

    vector<T *> result;
    result.resize(indexes.size());

    for (int i = 0; i < result.size(); i++) {
        result[i] = src[indexes[i]];
        if (copy) {
            result[i] = new T(result[i]);
        }
    }
    return result;
}

int randInt(int min, int max);

std::vector<size_t> argsort(const std::vector<Chromosome *> &array);

template <typename T>
T identity(const T &t) {
    return t;
}

template <typename T, typename K, typename V>
std::unordered_map<K, V> build_index(vector<T> items,
                                     K (*get_key)(const T&),
                                     V (*get_value)(const T&)) {
    std::unordered_map<K, V> diction;

    for (const auto &item : items) {
        diction.insert({ get_key(item), get_value(item) });
    }

    return diction;
}

template <typename T, typename K>
std::unordered_map<K, T> build_index(vector<T> items,
                                     K (*get_key)(const T&)) {
    std::unordered_map<K, T> diction;

    for (const auto &item : items) {
        diction.insert({ get_key(item), item });
    }

    return diction;
}

#endif //SAMPO_UTILS_H
