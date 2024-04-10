#ifndef CONTRACTOR_H
#define CONTRACTOR_H

#include <random>
#include <utility>
#include <vector>

#include "basic_types.h"
#include "interval.h"

class Worker : public Identifiable {
public:
    string name;
    int count;
    int cost;
    string contractor_id;
    IntervalGaussian productivity;

    explicit Worker(
        string id = "",
        string name = "",
        int count = 0,
        int cost = 0,
        string contractor_id = "",
        const IntervalGaussian& productivity = IntervalGaussian()
    )
        : Identifiable(std::move(id)),
          name(std::move(name)),
          count(count),
          cost(cost),
          contractor_id(std::move(contractor_id)),
          productivity(productivity) {}

    Worker(const Worker& other) = default;

    Worker& with_count(int count) {
        this->count = count;
        return *this;
    }

    inline Worker copy() const {
        return Worker(id, name, count, cost, contractor_id, productivity);
    }
};

class Contractor : public Identifiable {
public:
    string name;
    vector<Worker> workers;

    explicit Contractor(const string &id, string name, const vector<Worker> &workers)
        : Identifiable(id),
          name(std::move(name)),
          workers(workers) { }
};

#endif    // CONTRACTOR_H
