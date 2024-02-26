#ifndef NATIVE_CHROMOSOME_EVALUATOR_H
#define NATIVE_CHROMOSOME_EVALUATOR_H

// #pragma GCC optimize("Ofast")
// #pragma GCC optimize("no-stack-protector")
// #pragma GCC optimize("unroll-loops")
// #pragma GCC target("sse,sse2,sse3,ssse3,popcnt,abm,mmx,tune=native")
// #pragma GCC optimize("fast-math")

// #pragma optimize( "O2", on )

#define PY_SSIZE_T_CLEAN
#include <iostream>
#include <set>
#include <unordered_map>
#include <vector>

#include <omp.h>

#include "DLLoader.h"
#include "Python.h"
#include "native/schemas/evaluator_types.h"
#include "native/schemas/external.h"
#include "native/schemas/time_estimator.h"
#include "native/pycodec.h"

#include "numpy/arrayobject.h"
#include "sgs.h"
#include "native/scheduler/timeline/just_in_time.h"
#include "native/scheduler/fitness.h"

class ChromosomeEvaluator {
private:
    WorkGraph *wg;

    const vector<vector<int>> &parents;         // vertices' parents
    const vector<vector<int>> &headParents;     // vertices' parents without
                                                // inseparables
    const vector<vector<int>> &inseparables;    // inseparable chains with self
    const vector<vector<int>> &workers;         // contractor -> worker -> count
    const vector<float> &volumes;               // work -> worker -> WorkUnit.min_req
    const vector<vector<int>> &minReqs;         // work -> worker -> WorkUnit.max_req
    const vector<vector<int>> &maxReqs;         // work -> WorkUnit.volume
    const vector<string> &id2work;
    const vector<string> &id2res;

    worker_pool_t worker_pool;
    const LandscapeConfiguration &landscape;
    const WorkTimeEstimator &work_estimator;

    vector<vector<Worker*>> worker_pool_indices;
    vector<GraphNode*> index2node;
    const vector<Contractor*> &index2contractor;
    vector<int> index2zone;  // TODO
    unordered_map<string, int> worker_name2index;
    unordered_map<string, int> contractor2index;

    unordered_map<string, unordered_map<string, int>> minReqNames;
    unordered_map<string, unordered_map<string, int>> maxReqNames;

    int totalWorksCount;
    PyObject *pythonWrapper;
    bool usePythonWorkEstimator;

    WorkTimeEstimator *timeEstimator;
    dlloader::DLLoader<ITimeEstimatorLibrary> loader {
        External::timeEstimatorLibPath
    };

    int calculate_working_time(
        int chromosome_ind,
        int work,
        int team_target,
        const int *resources,
        size_t teamSize
    ) {
        if (usePythonWorkEstimator) {
            auto res = PyObject_CallMethod(
                pythonWrapper,
                "calculate_working_time_ind",
                "(iii)",
                chromosome_ind,
                team_target,
                work
            );
            if (res == nullptr) {
                cerr << "Result is NULL" << endl << flush;
                return 0;
            }
            Py_DECREF(res);
            return (int)PyLong_AsLong(res);
        }
        else {
            // map resources from indices to names
            vector<pair<string, int>> resourcesWithNames;
            for (int i = 0; i < teamSize; i++) {
                if (resources[i] != 0) {
                    resourcesWithNames.emplace_back(id2res[i], resources[i]);
                }
            }
            //        cout << "Called WorkTimeEstimator!" << endl;
            //        return 1;
            return calculate_working_time(
                chromosome_ind,
                id2work[work],
                id2work[team_target],
                volumes[work],
                resourcesWithNames
            );
        }
    }

    // TODO Rework with OOP
    int calculate_working_time(
        int chromosome_ind,
        const string &work,
        const string &team_target,
        float volume,
        vector<pair<string, int>> &resources
    ) {
        if (usePythonWorkEstimator) {
            auto res = PyObject_CallMethod(
                pythonWrapper,
                "calculate_working_time",
                "(iss)",
                chromosome_ind,
                team_target.data(),
                work.data()
            );
            if (res == nullptr) {
                cerr << "Result is NULL" << endl << flush;
                return 0;
            }
            Py_DECREF(res);
            return (int)PyLong_AsLong(res);
        } else {
            return timeEstimator->estimateTime(work, volume, resources);
        }
    }

public:
    int numThreads;

    explicit ChromosomeEvaluator(EvaluateInfo *info)
        : wg(info->wg),
          parents(info->parents),
          headParents(info->headParents),
          inseparables(info->inseparables),
          workers(info->workers),
          volumes(info->volume),
          minReqs(info->minReq),
          maxReqs(info->maxReq),
          id2work(info->id2work),
          id2res(info->id2res),
          landscape(info->landscape),
          work_estimator(*info->work_estimator),
          index2contractor(info->contractors),
          totalWorksCount(info->totalWorksCount),
          pythonWrapper(info->pythonWrapper) {

        for (size_t i = 0; i < minReqs.size(); i++) {
            const auto &work = id2work[i];
            // TODO I think we can remove this
            minReqNames[work] = unordered_map<string, int>();
            maxReqNames[work] = unordered_map<string, int>();
            for (size_t j = 0; j < minReqs[i].size(); j++) {
                if (minReqs[i][j] != 0) {
                    const auto &res = id2res[j];
                    minReqNames[work][res] = minReqs[i][j];
                    maxReqNames[work][res] = maxReqs[i][j];
                }
            }
        }

        // construct the outer numeration
        // start with inseparable heads
        for (auto* node : wg->nodes) {
            if (!node->is_inseparable_son()) {
                index2node.push_back(node);
            }
        }
        // continue with others
        // this should match the numeration on the Python-size backend,
        // otherwise everything will break
        for (auto* node : wg->nodes) {
            if (node->is_inseparable_son()) {
                index2node.push_back(node);
            }
        }
        // TODO
//        worker_pool_indices.resize(index2contractor.size());
//        for (size_t contractor_ind = 0; contractor_ind < index2contractor.size(); contractor_ind++) {
//            for (size_t worker_ind = 0; worker_ind < index2contractor[contractor_ind]->workers.size(); worker_ind++) {
//                worker_pool_indices[worker_ind].resize(index2contractor.size());
//                worker_pool_indices[worker_ind][contractor_ind] = index2contractor[contractor_ind]->workers[worker_ind];
//            }
//        }

        for (auto* contractor : index2contractor) {
            for (auto* worker : contractor->workers) {
                worker_pool[worker->id][contractor->id] = worker;
            }
        }

        int ind = 0;
        for (auto& it : worker_pool) {
            worker_name2index[it.first] = ind;
            ind++;
        }

        for (int index = 0; index < index2contractor.size(); index++) {
            contractor2index[index2contractor[index]->id] = index;
        }

        this->usePythonWorkEstimator = info->usePythonWorkEstimator;
        this->numThreads = this->usePythonWorkEstimator ? 1 : omp_get_num_procs();
        printf("Genetic running threads: %i\n", this->numThreads);

        if (info->useExternalWorkEstimator) {
            loader.DLOpenLib();
            auto library        = loader.DLGetInstance();
            this->timeEstimator = library->create(info->timeEstimatorPath);
        }
//        else if (!usePythonWorkEstimator) {
//            this->timeEstimator = new DefaultWorkTimeEstimator(minReqNames, maxReqNames);
//        }
    }

    // TODO Research why deleting timeEstimator causes Head Corruption crash
    //    ~ChromosomeEvaluator() {
    //        delete timeEstimator;
    //    }
    //    ~ChromosomeEvaluator() {
    //        loader.DLCloseLib();
    //    }
    ~ChromosomeEvaluator() = default;

    bool isValid(Chromosome *chromosome) {
        bool *visited = new bool[chromosome->numWorks()] { false };

        // check edges
        for (int i = 0; i < chromosome->numWorks(); i++) {
            int node      = *chromosome->getOrder()[i];
            visited[node] = true;
            for (int parent : headParents[node]) {
                if (!visited[parent]) {
                    return false;
                }
            }
        }

        delete[] visited;
        // check resources
        for (int node = 0; node < chromosome->numWorks(); node++) {
            int contractor = chromosome->getContractor(node);
            for (int res = 0; res < chromosome->numResources(); res++) {
                int count = chromosome->getResources()[node][res];
                if (count < minReqs[node][res]
                    || count > chromosome->getContractors()[contractor][res]) {
                    return false;
                }
            }
        }

        return true;
    }

    void evaluate(vector<Chromosome *> &chromosomes) {
        auto fitness = TimeFitness();
#pragma omp parallel for shared(chromosomes, fitness) default(none) num_threads(this->numThreads)
        for (size_t i = 0; i < chromosomes.size(); i++) {
            if (isValid(chromosomes[i])) {
                // TODO Add sgs_type parametrization
                JustInTimeTimeline timeline(worker_pool, landscape);
                Time assigned_parent_time;
                swork_dict_t schedule = SGS::serial(chromosomes[i],
                                                    worker_pool,
                                                    worker_pool_indices,
                                                    index2node,
                                                    index2contractor,
                                                    index2zone,
                                                    worker_name2index,
                                                    contractor2index,
                                                    landscape,
                                                    assigned_parent_time,
                                                    timeline,
                                                    work_estimator);
                chromosomes[i]->fitness = fitness.evaluate(schedule);
            }
            else {
                chromosomes[i]->fitness = INT_MAX;
            }
        }
    }
};


#endif    // NATIVE_CHROMOSOME_EVALUATOR_H
