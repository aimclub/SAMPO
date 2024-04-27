#pragma once

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "native/schemas/dtime.h"
#include "native/schemas/scheduled_work.h"
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
