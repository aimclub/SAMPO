#ifndef NATIVE_EVALUATOR_TYPES_H
#define NATIVE_EVALUATOR_TYPES_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include <vector>

using namespace std;

typedef struct {
    vector<int> order;
    vector<vector<int>> resources;
    vector<vector<int>> contractors;
} Chromosome;

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
