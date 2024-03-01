#ifndef CONTRACTOR_H
#define CONTRACTOR_H

#include <random>
#include <utility>
#include <vector>

#include "basic_types.h"
#include "interval.h"

class Worker : public Identifiable {

public:
    string id;
    string name;
    int count;
    int cost;
    string contractor_id;
    IntervalGaussian productivity;

    Worker(
        string id = "",
        string name = "",
        int count = 0,
        int cost = 0,
        string contractorId = "",
        const IntervalGaussian& productivity = IntervalGaussian()
    )
        : id(std::move(id)),
          name(std::move(name)),
          count(count),
          cost(cost),
          contractor_id(std::move(contractorId)),
          productivity(productivity) { }

    Worker& with_count(int count) {
        this->count = count;
        return *this;
    }

    inline Worker copy() const {
        return { id, name, count, cost, contractor_id, productivity };
    }
};

class Contractor : public Identifiable {
public:
    vector<Worker *> workers;

    explicit Contractor(vector<Worker *> &workers) : workers(workers) { }
};

#endif    // CONTRACTOR_H
