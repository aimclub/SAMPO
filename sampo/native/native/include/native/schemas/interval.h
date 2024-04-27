#pragma once

#include <random>
#include <utility>
#include <vector>

#include "basic_types.h"

#define EPS 1e5f

using namespace std;

class IntervalGaussian {
private:
    random_device rd {};
    mt19937 gen { rd() };
    normal_distribution<float> d;

    float min_val;
    float max_val;

public:
    explicit IntervalGaussian(float mean = 1, float sigma = EPS, float min_val = 0, float max_val = 0);

    IntervalGaussian(const IntervalGaussian &other);

    double mean();

    float randFloat();

    int randInt();
};
