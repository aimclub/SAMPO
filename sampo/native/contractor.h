#ifndef CONTRACTOR_H
#define CONTRACTOR_H

#include <utility>
#include <vector>
#include <random>
#include "basic_types.h"

using namespace std;

class IntervalGaussian {
private:
    constexpr static const double EPS = 1e5;

    random_device rd{};
    mt19937 gen{rd()};
    normal_distribution<> d;

    float min_val;
    float max_val;
public:
    explicit IntervalGaussian(float mean, float sigma, float min_val, float max_val)
        : min_val(min_val), max_val(max_val), d(normal_distribution<>{mean, sigma}) {}

    IntervalGaussian(const IntervalGaussian& other)
        : IntervalGaussian((float) other.d.mean(), (float) other.d.stddev(), other.min_val, other.max_val) {}

    float randFloat() {
        return (float) d(gen);
    }

    int randInt() {
        int value = (int) round(randFloat());
        value = max(value, int(min_val - EPS));
        value = min(value, int(max_val + EPS));
        return value;
    }
};

class Worker : public Identifiable {
    string id;
    string name;
    int count;
    string contractor_id;
    IntervalGaussian productivity;

public:
    Worker(string id, string name, int count, string contractorId, const IntervalGaussian &productivity)
        : id(std::move(id)), name(std::move(name)), count(count),
        contractor_id(std::move(contractorId)), productivity(productivity) {}
};

class Contractor : public Identifiable {
public:
    vector<Worker*> workers;

    explicit Contractor(vector<Worker*>& workers) : workers(workers) {}
};

#endif //CONTRACTOR_H
