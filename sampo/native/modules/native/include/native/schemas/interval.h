#pragma once

#include <random>
#include <utility>
#include <vector>

#include "basic_types.h"

#define EPS 1e5

using namespace std;

class IntervalGaussian {
private:
    random_device rd {};
    mt19937 gen { rd() };
    normal_distribution<> d;

    float min_val;
    float max_val;

public:
    explicit IntervalGaussian(
            float mean = 1, float sigma = 0, float min_val = 0, float max_val = 0
    )
            : d(normal_distribution<> { mean, sigma }),
              min_val(min_val),
              max_val(max_val) {}

    IntervalGaussian(const IntervalGaussian &other)
            : IntervalGaussian(
            (float)other.d.mean(),
            (float)other.d.stddev(),
            other.min_val,
            other.max_val
    ) { }

    double mean() {
        return d.mean();
    }

    float randFloat() {
        return (float)d(gen);
    }

    int randInt() {
        int value = (int)round(randFloat());
        value     = max(value, int(min_val - EPS));
        value     = min(value, int(max_val + EPS));
        return value;
    }
};
