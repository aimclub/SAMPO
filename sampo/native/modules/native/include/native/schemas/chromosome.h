#pragma once

#define PY_SSIZE_T_CLEAN
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "native/schemas/dtime.h"
#include "native/schemas/scheduled_work.h"
#include "native/schemas/time_estimator.h"
#include "native/schemas/spec.h"

using namespace std;

template <typename T>
class Array2D {
private:
    size_t length = 0;
    size_t stride = 0;
    T *data       = nullptr;
    bool shallow  = true;

public:
    Array2D() = default;

    Array2D(size_t length, size_t stride, T *data)
            : length(length), stride(stride), data(data), shallow(true) { }

    //    Array2D(size_t length, size_t stride)
    //        : Array2D(length, stride, (T*) malloc(length * sizeof(T))),
    //        shallow(true) {}

    Array2D(const Array2D &other)
            : Array2D(other.length, other.stride, other.shallow) {
        //        memcpy(this->data, other.data, length * sizeof(T));
        cout << "Copy" << endl;
    }

    ~Array2D() {
        if (!shallow) {
            cout << "Free array" << endl;
            //        free(this->data);
        }
    }

    // int* to use this operator as 2D array. To use as 1D array, follow this
    // call with '*'.
    int *operator[](size_t i) {
        return this->data + i * stride;
    }

    // shallow copy
    Array2D<T> &operator=(const Array2D &other) {
        this->length = other.length;
        this->stride = other.stride;
        this->data   = other.data;
        return *this;
    }

    int width() {
        return stride;
    }

    int height() {
        return length / stride;
    }

    int size() {
        return length;
    }
};

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

    Chromosome(int worksCount, int resourcesCount, int contractorsCount, ScheduleSpec spec = ScheduleSpec())
            : worksCount(worksCount),
              resourcesCount(resourcesCount),
              contractorsCount(contractorsCount),
              spec(std::move(spec)) {
        //        cout << worksCount << " " << resourcesCount << " " <<
        //        contractorsCount << endl;
        size_t ORDER_SHIFT     = 0;
        size_t RESOURCES_SHIFT = ORDER_SHIFT + worksCount;
        size_t CONTRACTORS_SHIFT =
                RESOURCES_SHIFT + worksCount * (resourcesCount + 1);
        this->DATA_SIZE =
                (CONTRACTORS_SHIFT + contractorsCount * resourcesCount)
                * sizeof(int);
        //        cout << ORDER_SHIFT << " " << RESOURCES_SHIFT << " " <<
        //        CONTRACTORS_SHIFT << endl; cout << DATA_SIZE << endl;
        this->data = (int *)malloc(DATA_SIZE);
        if (data == nullptr) {
            cout << "Not enough memory" << endl;
            return;
        }
        this->order     = Array2D<int>(worksCount, 1, this->data);
        this->resources = Array2D<int>(
                worksCount * (resourcesCount + 1),
                resourcesCount + 1,
                this->data + RESOURCES_SHIFT
        );
        this->contractors = Array2D<int>(
                contractorsCount * resourcesCount,
                resourcesCount,
                this->data + CONTRACTORS_SHIFT
        );

        //        cout << order.size()<< endl;
        //        cout << resources.width() << " " << resources.height() <<
        //        endl; cout << contractors.width() << " " <<
        //        contractors.height() << endl;
    }

    Chromosome(Chromosome *other)
            : Chromosome(
            other->worksCount, other->resourcesCount, other->contractorsCount
    ) {
        // copy all packed data
        memcpy(this->data, other->data, DATA_SIZE);
        this->fitness = other->fitness;
    }

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
                                     vector<string> order = vector<string>()) {
        auto* chromosome = new Chromosome(work_id2index.size(),
                                          worker_name2index.size(),
                                          contractor2index.size());

        if (order.size() == 0) {
            // if order not specified, create
            for (auto& entry : work_id2index) {
                order.emplace_back(entry.first);
            }
        }

        for (size_t i = 0; i < order.size(); i++) {
            auto& node = order[i];
            int index = work_id2index[node];
            *chromosome->getOrder()[i] = index;
            for (auto& resource : schedule[node].workers) {
                int res_index = worker_name2index[resource.name];
                chromosome->getResources()[index][res_index] = resource.count;
                chromosome->getContractor(index) = contractor2index[resource.contractor_id];
            }
        }

        // TODO Implement this using memcpy
        auto& contractor_chromosome = chromosome->getContractors();
        for (int contractor = 0; contractor < contractor_borders.height(); contractor++) {
            for (int resource = 0; resource < contractor_borders.width(); resource++) {
                contractor_chromosome[contractor][resource] = contractor_borders[contractor][resource];
            }
        }

        return chromosome;
    }
};
