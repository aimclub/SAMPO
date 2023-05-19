#ifndef NATIVE_EVALUATOR_TYPES_H
#define NATIVE_EVALUATOR_TYPES_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include <vector>
#include <cstdlib>
#include <cstring>

using namespace std;

template<typename T>
class Array2D {
private:
    size_t length;
    size_t stride;
    T* data;
    bool shallow;
public:
    explicit Array2D(size_t length, size_t stride, T* data, bool shallow = true)
        : length(length), stride(stride), data(data), shallow(shallow) {}

    explicit Array2D(size_t length = 1, size_t stride = 1)
        : Array2D(length, stride, (T*) malloc(length * sizeof(T)), false) {}

    Array2D(const Array2D& other) : Array2D(other.length, other.stride) {
        memcpy(this->data, other.data, length * sizeof(T));
    }

    ~Array2D() {
        if (!shallow) {
            free(this->data);
        }
    }

    // int* to use this operator as 2D array. To use as 1D array, follow this call with '*'.
    int* operator[](int i) {
        return this->data + i * stride;
    }

    // shallow copy
    Array2D<T>& operator=(const Array2D& other) {
        this->length = other.length;
        this->stride = other.stride;
        this->data = other.data;
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

    int* data;  // packed one-after-another chromosome parts
    Array2D<int> order;
    Array2D<int> resources;
    Array2D<int> contractors;
    size_t DATA_SIZE;
public:
    explicit Chromosome(int worksCount, int resourcesCount, int contractorsCount)
        : worksCount(worksCount), resourcesCount(resourcesCount), contractorsCount(contractorsCount) {
        size_t ORDER_SHIFT = 0;
        size_t RESOURCES_SHIFT = ORDER_SHIFT + worksCount;
        size_t CONTRACTORS_SHIFT = RESOURCES_SHIFT + worksCount * (resourcesCount + 1);
        this->DATA_SIZE = (CONTRACTORS_SHIFT + contractorsCount * resourcesCount) * sizeof(int);
        this->data = (int*) malloc(DATA_SIZE);
        this->order       = Array2D<int>(worksCount, 1, this->data);
        this->resources   = Array2D<int>(worksCount * (resourcesCount + 1),
                                         resourcesCount + 1, this->data + RESOURCES_SHIFT);
        this->contractors = Array2D<int>(contractorsCount * resourcesCount,
                                         resourcesCount, this->data + CONTRACTORS_SHIFT);
    }

    explicit Chromosome(const Chromosome* other)
        : Chromosome(other->worksCount, other->resourcesCount, other->contractorsCount) {
        // copy all packed data
        memcpy(this->data, other->data, DATA_SIZE);
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
};

typedef struct {
    PyObject* pythonWrapper;
    vector<vector<int>> parents;
    vector<vector<int>> inseparables;
    vector<vector<int>> workers;
    vector<double> volume;
    vector<vector<int>> minReq;
    vector<vector<int>> maxReq;
    int totalWorksCount;
    bool useExternalWorkEstimator;
} EvaluateInfo;

#endif //NATIVE_EVALUATOR_TYPES_H
