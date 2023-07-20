#ifndef NATIVE_EVALUATOR_TYPES_H
#define NATIVE_EVALUATOR_TYPES_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include <vector>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

template<typename T>
class Array2D {
private:
    size_t length = 0;
    size_t stride = 0;
    T* data = nullptr;
    bool shallow = true;
public:
    Array2D() = default;

    Array2D(size_t length, size_t stride, T* data)
        : length(length), stride(stride), data(data), shallow(true) {}

//    Array2D(size_t length, size_t stride)
//        : Array2D(length, stride, (T*) malloc(length * sizeof(T))), shallow(true) {}

    Array2D(const Array2D& other) : Array2D(other.length, other.stride, other.shallow) {
//        memcpy(this->data, other.data, length * sizeof(T));
        cout << "Copy" << endl;
    }

    ~Array2D() {
        if (!shallow) {
            cout << "Free array" << endl;
//            free(this->data);
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
    int fitness = INT_MAX;  // infinity

    Chromosome(int worksCount, int resourcesCount, int contractorsCount)
        : worksCount(worksCount), resourcesCount(resourcesCount), contractorsCount(contractorsCount) {
//        cout << worksCount << " " << resourcesCount << " " << contractorsCount << endl;
        size_t ORDER_SHIFT = 0;
        size_t RESOURCES_SHIFT = ORDER_SHIFT + worksCount;
        size_t CONTRACTORS_SHIFT = RESOURCES_SHIFT + worksCount * (resourcesCount + 1);
        this->DATA_SIZE = (CONTRACTORS_SHIFT + contractorsCount * resourcesCount) * sizeof(int);
//        cout << ORDER_SHIFT << " " << RESOURCES_SHIFT << " " << CONTRACTORS_SHIFT << endl;
//        cout << DATA_SIZE << endl;
        this->data = (int*) malloc(DATA_SIZE);
        if (data == nullptr) {
            cout << "Not enough memory" << endl;
            return;
        }
        this->order       = Array2D<int>(worksCount, 1, this->data);
        this->resources   = Array2D<int>(worksCount * (resourcesCount + 1),
                                         resourcesCount + 1, this->data + RESOURCES_SHIFT);
        this->contractors = Array2D<int>(contractorsCount * resourcesCount,
                                         resourcesCount, this->data + CONTRACTORS_SHIFT);

//        cout << order.size()<< endl;
//        cout << resources.width() << " " << resources.height() << endl;
//        cout << contractors.width() << " " << contractors.height() << endl;
    }

    Chromosome(Chromosome* other)
        : Chromosome(other->worksCount, other->resourcesCount, other->contractorsCount) {
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

    int& getContractor(int work) {
        return getResources()[work][resourcesCount];
    }

    int* getContractorBorder(int contractor) {
        return getContractors()[contractor];
    }

    int* getWorkResourceBorder(int work) {
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
};

typedef struct {
    PyObject* pythonWrapper;
    vector<vector<int>> parents;
    vector<vector<int>> headParents;
    vector<vector<int>> inseparables;
    vector<vector<int>> workers;
    vector<float> volume;
    vector<vector<int>> minReq;
    vector<vector<int>> maxReq;
    vector<string> id2work;
    vector<string> id2res;
    string timeEstimatorPath;
    int totalWorksCount;
    bool usePythonWorkEstimator;
    bool useExternalWorkEstimator;
} EvaluateInfo;

#endif //NATIVE_EVALUATOR_TYPES_H
