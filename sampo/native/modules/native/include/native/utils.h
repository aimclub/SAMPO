#ifndef SAMPO_UTILS_H
#define SAMPO_UTILS_H

#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

#include "native/schemas/evaluator_types.h"

using namespace std;

// TODO Think about make own more cache-friendly shuffle using areas around
// target element
vector<int> sample_ind(int n, float prob, random_device &rd);

template <typename T>
vector<T *> sample(vector<T *> &src, float prob, random_device &rd, bool copy = true);

int randInt(int min, int max);

inline std::vector<size_t> argsort(const std::vector<Chromosome *> &array);

#endif //SAMPO_UTILS_H
