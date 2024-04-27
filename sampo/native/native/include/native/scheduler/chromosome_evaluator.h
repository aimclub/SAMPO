#ifndef NATIVE_CHROMOSOME_EVALUATOR_H
#define NATIVE_CHROMOSOME_EVALUATOR_H

// #pragma GCC optimize("Ofast")
// #pragma GCC optimize("no-stack-protector")
// #pragma GCC optimize("unroll-loops")
// #pragma GCC target("sse,sse2,sse3,ssse3,popcnt,abm,mmx,tune=native")
// #pragma GCC optimize("fast-math")

// #pragma optimize( "O2", on )

#include <iostream>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include <omp.h>

#include "DLLoader.h"
#include "native/schemas/evaluator_types.h"
#include "native/schemas/external.h"
#include "native/schemas/time_estimator.h"
#include "native/schemas/chromosome.h"
#include "native/pycodec.h"

#include "native/scheduler/timeline/just_in_time.h"
#include "native/scheduler/sgs.h"
#include "native/scheduler/fitness.h"

class ChromosomeEvaluator {
private:
    const WorkGraph *wg;

    vector<vector<int>> headParents;     // vertices' parents without inseparables
    vector<vector<int>> minReqs;         // work -> worker -> WorkUnit.min_req
    vector<vector<int>> maxReqs;         // work -> worker -> WorkUnit.max_req
    vector<string> id2work;   // unused
    vector<string> id2res;    // unused

    worker_pool_t worker_pool;
    LandscapeConfiguration landscape;
    const WorkTimeEstimator *work_estimator;

    vector<vector<Worker*>> worker_pool_indices;
    vector<GraphNode*> index2node;
    vector<Contractor*> contractors;
    vector<int> index2zone;  // TODO
    unordered_map<string, int> worker_name2index;
    unordered_map<string, int> contractor2index;

    ScheduleSpec spec;
    ScheduleGenerationScheme sgs;

//    const py::object &python_wrapper;

    // TODO (?) Make interop with Python work estimators like in old NativeWrapper was
//    WorkTimeEstimator *timeEstimator;
//    dlloader::DLLoader<ITimeEstimatorLibrary> loader {
//        External::timeEstimatorLibPath
//    };

public:
    int num_threads;

    explicit ChromosomeEvaluator(const WorkGraph *wg,
                                 vector<Contractor*> contractors,
                                 ScheduleSpec spec,
                                 const WorkTimeEstimator *work_estimator);

    // TODO Research why deleting timeEstimator causes Head Corruption crash
    //    ~ChromosomeEvaluator() {
    //        delete timeEstimator;
    //    }
    //    ~ChromosomeEvaluator() {
    //        loader.DLCloseLib();
    //    }
    ~ChromosomeEvaluator();

    bool is_valid(Chromosome *chromosome);

    void set_sgs(ScheduleGenerationScheme sgs);

    void evaluate(vector<Chromosome *> &chromosomes);
};


#endif    // NATIVE_CHROMOSOME_EVALUATOR_H
