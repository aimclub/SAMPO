#pragma once

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "native/schemas/dtime.h"
#include "native/schemas/scheduled_work.h"
#include "native/schemas/spec.h"
#include "native/schemas/array2d.h"

using namespace std;

class Chromosome {
private:
    int worksCount;
    int resourcesCount;
    int contractorsCount;

    int *data;    // packed one-after-another chromosome parts
    Array2D<int> order;
    Array2D<int> resources;
    Array2D<int> contractors;
    ScheduleSpec spec;
    size_t DATA_SIZE;

public:
    float fitness = TIME_INF;    // infinity

    Chromosome(int worksCount, int resourcesCount, int contractorsCount, ScheduleSpec spec = ScheduleSpec());

    explicit Chromosome(Chromosome *other);

    ~Chromosome() {
        free(data);
    }

    // ---------------
    // Getters/Setters
    // ---------------

    Array2D<int>& getOrder() {
        return order;
    }

    Array2D<int>& getResources() {
        return resources;
    }

    Array2D<int>& getContractors() {
        return contractors;
    }

    ScheduleSpec& getSpec() {
        return spec;
    }

    int &getContractor(int work) {
        return getResources()[work][resourcesCount];
    }

    int *getContractorBorder(int contractor) {
        return getContractors()[contractor];
    }

    int *getWorkResourceBorder(int work) {
        return getContractorBorder(getContractor(work));
    }

    int numWorks() {
        return getOrder().size();
    }

    int numResources() {
        return getResources().width() - 1;
    }

    int numContractors() {
        return getContractors().height();
    }

    static Chromosome* from_schedule(unordered_map<string, int> &work_id2index,
                                     unordered_map<string, int> &worker_name2index,
                                     unordered_map<string, int> &contractor2index,
                                     Array2D<int> &contractor_borders,
                                     unordered_map<string, ScheduledWork> &schedule,
                                     ScheduleSpec &spec,
                                     LandscapeConfiguration &landscape,
                                     vector<string> order = vector<string>());
};
