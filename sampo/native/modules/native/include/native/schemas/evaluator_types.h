#pragma once

#define PY_SSIZE_T_CLEAN
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "Python.h"
#include "native/schemas/dtime.h"
#include "native/schemas/scheduled_work.h"
#include "native/schemas/time_estimator.h"
#include "native/schemas/spec.h"
#include "native/schemas/workgraph.h"

using namespace std;

typedef struct {
    PyObject *pythonWrapper;
    WorkGraph *wg;
    vector<Contractor*> contractors;
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
    LandscapeConfiguration landscape;
    WorkTimeEstimator *work_estimator;
    int totalWorksCount;
    bool usePythonWorkEstimator;
    bool useExternalWorkEstimator;
} EvaluateInfo;

using swork_dict_t = unordered_map<string, ScheduledWork>;
using exec_times_t = unordered_map<string, pair<Time, Time>>;
using worker_pool_t = unordered_map<string, unordered_map<string, Worker>>;
