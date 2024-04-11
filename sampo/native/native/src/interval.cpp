#include "native/schemas/interval.h"

IntervalGaussian::IntervalGaussian(float mean, float sigma, float min_val, float max_val)
        : d(normal_distribution<float> { mean, max(sigma, EPS) }),
          min_val(min_val),
          max_val(max_val) {}

IntervalGaussian::IntervalGaussian(const IntervalGaussian &other)
        : IntervalGaussian(
        (float)other.d.mean(),
        (float)other.d.stddev(),
        other.min_val,
        other.max_val
) { }

double IntervalGaussian::mean() {
    return d.mean();
}

float IntervalGaussian::randFloat() {
    return (float)d(gen);
}

int IntervalGaussian::randInt() {
    int value = (int)round(randFloat());
    value     = max(value, int(min_val - EPS));
    value     = min(value, int(max_val + EPS));
    return value;
}


