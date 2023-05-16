#ifndef NATIVE_CHROMOSOME_EVALUATOR_H
#define NATIVE_CHROMOSOME_EVALUATOR_H

//#pragma GCC optimize("Ofast")
//#pragma GCC optimize("no-stack-protector")
//#pragma GCC optimize("unroll-loops")
//#pragma GCC target("sse,sse2,sse3,ssse3,popcnt,abm,mmx,tune=native")
//#pragma GCC optimize("fast-math")

//#pragma optimize( "O2", on )

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"
#include "pycodec.h"
#include "evaluator_types.h"

#include <vector>
#include <iostream>
#include <unordered_map>
#include <omp.h>

using namespace std;

// worker -> contractor -> vector<time, count> in descending order
typedef vector<vector<vector<pair<int, int>>>> Timeline;

#define TIME_INF 2000000000

class ChromosomeEvaluator {
private:
    const vector<vector<int>>& parents;      // vertices' parents
    const vector<vector<int>>& inseparables; // inseparable chains with self
    const vector<vector<int>>& workers;      // contractor -> worker -> count
    vector<double> volume;                   // work -> worker -> WorkUnit.min_req
    vector<vector<int>> minReqs;             // work -> worker -> WorkUnit.max_req
    vector<vector<int>> maxReqs;             // work -> WorkUnit.volume

    int totalWorksCount;
    PyObject* pythonWrapper;
    bool useExternalWorkEstimator;
    int numThreads;

    inline static float get_productivity(size_t workerType, int worker_count) {
        // TODO
        return 1.0F * (float) worker_count;
    }

    inline static float communication_coefficient(int workerCount, int maxWorkerCount) {
        int n = workerCount;
        int m = maxWorkerCount;
        return 1 / (float) (6 * m * m) * (float) (-2 * n * n * n + 3 * n * n + (6 * m * m - 1) * n);
    }

    inline static int get_worker(const PyObject* resources, size_t work, size_t worker) {
        return * (int*) PyArray_GETPTR2(resources, work, worker);
    }

    int calculate_working_time(int chromosome_ind, int work, int team_target, const PyObject* resources, size_t teamSize) {
        if (useExternalWorkEstimator) {
            auto res = PyObject_CallMethod(pythonWrapper, "calculate_working_time", "(iii)", chromosome_ind,
                                           team_target, work);
            if (res == nullptr) {
                cerr << "Result is NULL" << endl << flush;
                return 0;
            }
            Py_DECREF(res);
            return (int) PyLong_AsLong(res);
        } else {
            // the _abstract_estimate from WorkUnit
            int time = 0;

            for (size_t i = 0; i < teamSize; i++) {
                int minReq = this->minReqs[work][i];
                if (minReq == 0)
                    continue;
                if (get_worker(resources, team_target, i) < minReq) {
//                    cout << "Not conforms to min_req: " << get_worker(resources, team_target, i) << " < " << minReq << " on work " << work
//                         << " and worker " << i << ", chromosome " << chromosome_ind << ", teamSize=" << teamSize << endl;
//                    cout << "Team: ";
//                    for (size_t j = 0; j < teamSize; j++) {
//                        cout << get_worker(resources, team_target, i) << " ";
//                    }
//                    cout << endl;
                    return TIME_INF;
                }
                int maxReq = this->maxReqs[work][i];

                float productivity = get_productivity(i, get_worker(resources, team_target, i));
                productivity *= communication_coefficient(get_worker(resources, team_target, i), maxReq);

//                if (productivity < 0.000001) {
//                    return TIME_INF;
//                }
//                productivity = 0.1;
                int newTime = ceil((float) volume[work] / productivity);
                if (newTime > time) {
                    time = newTime;
                }
            }

            return time;
        }
    }

    int findMinStartTime(int nodeIndex, int contractor, const PyObject* resources, size_t teamSize,
                         vector<int>& completed, Timeline& timeline) {
        int maxParentTime = 0;
        // find min start time
        for (int parent : parents[nodeIndex]) {
            maxParentTime = max(maxParentTime, completed[parent]);
        }

        int maxAgentTime = 0;

        for (int worker = 0; worker < teamSize; worker++) {
            int worker_count = get_worker(resources, nodeIndex, worker);
            int need_count = worker_count;
            if (need_count == 0) continue;

            // Traverse list while not enough resources and grab it
            auto &worker_timeline = timeline[contractor][worker];
            size_t ind = worker_timeline.size() - 1;
            while (need_count > 0) {
//                cout << "ind: " << ind << endl;
                int offer_count = worker_timeline[ind].second;
                maxAgentTime = max(maxAgentTime, worker_timeline[ind].first);

                if (need_count < offer_count) {
                    offer_count = need_count;
                }
                need_count -= offer_count;
                if (ind == 0 && need_count > 0) {
                    cerr << "Not enough workers" << endl;
                    return TIME_INF;
                }
                ind--;
            }
        }

        return max(maxParentTime, maxAgentTime);
    }

    static void updateTimeline(int work, int finishTime, int contractor, const PyObject* resources, size_t teamSize, Timeline& timeline) {
        for (int worker = 0; worker < teamSize; worker++) {
            int worker_count = get_worker(resources, work, worker);
            int need_count = worker_count;
            if (need_count == 0) continue;

            // Consume need workers
            auto& worker_timeline = timeline[contractor][worker];
            while (need_count > 0) {
                int next_count = worker_timeline[worker_timeline.size() - 1].second;
                if (next_count > need_count) {
                    worker_timeline[worker_timeline.size() - 1].second -= need_count;
                    break;
                }
                need_count -= next_count;
                if (worker_timeline.size() == 1 && need_count > 0) {
                    cerr << "---- Empty worker_timeline for worker " << worker << " and contractor " << contractor << endl << flush;
                    return;
                }
                worker_timeline.pop_back();
            }

            // Add to the right place using bubble-sort iterations
            worker_timeline.emplace_back(finishTime, worker_count);
            size_t ind = worker_timeline.size() - 1;
            while (ind > 0 && worker_timeline[ind].first > worker_timeline[ind - 1].first) {
                auto tmp = worker_timeline[ind];
                worker_timeline[ind] = worker_timeline[ind - 1];
                worker_timeline[ind - 1] = tmp;
                ind--;
            }
        }
    }

    int schedule(int chromosome_ind, int nodeIndex, int startTime, int contractor, const PyObject* resources,
                 size_t teamSize, vector<int>& completed, Timeline& timeline) {
        int finishTime = startTime;

        for (int dep_node : inseparables[nodeIndex]) {
            int maxParentTime = 0;
            // find min start time
            for (int parent : parents[dep_node]) {
                maxParentTime = max(maxParentTime, completed[parent]);
            }
            startTime = max(startTime, maxParentTime);

            int workingTime = calculate_working_time(chromosome_ind, dep_node, nodeIndex, resources, teamSize);
            finishTime = startTime + workingTime;

            // cache finish time of scheduled work
            completed[dep_node] = finishTime;
        }

        updateTimeline(nodeIndex, finishTime, contractor, resources, teamSize, timeline);

        return finishTime;
    }

    inline Timeline createTimeline() {
        Timeline timeline;

        timeline.resize(workers.size());
        for (int contractor = 0; contractor < workers.size(); contractor++) {
            timeline[contractor].resize(workers[0].size());
            for (int worker = 0; worker < workers[0].size(); worker++) {
                timeline[contractor][worker].emplace_back(0, workers[contractor][worker]);
            }
        }

        return timeline;
    }

public:
    explicit ChromosomeEvaluator(EvaluateInfo* info)
        : parents(info->parents), inseparables(info->inseparables), workers(info->workers) {
        this->totalWorksCount = info->totalWorksCount;
        this->pythonWrapper = info->pythonWrapper;
        this->useExternalWorkEstimator = info->useExternalWorkEstimator;
//         this->numThreads = this->useExternalWorkEstimator ? 1 : omp_get_num_procs();
        this->numThreads = 1;
        this->volume = info->volume;
        this->minReqs = info->minReq;
        this->maxReqs = info->maxReq;

//        cout << volume.size() << endl;
//        cout << minReqs[5].size() << " " << minReqs.size() << endl;
//        cout << maxReqs[5].size() << " " << maxReqs.size() << endl;

//         printf("Threads: %i\n", this->numThreads);
    }

    ~ChromosomeEvaluator() = default;

    vector<int> evaluate(vector<PyObject*>& chromosomes) {
        auto results = vector<int>();
        results.resize(chromosomes.size());

        #pragma omp parallel for firstprivate(chromosomes) shared(results) default (none) num_threads(this->numThreads)
        for (int i = 0; i < chromosomes.size(); i++) {
            results[i] = evaluate(i, chromosomes[i]);
        }
        return results;
    }

    int evaluate(int chromosome_ind, PyObject* chromosome) {
        PyObject* pyOrder; //= PyList_GetItem(chromosome, 0);
        PyObject* pyResources; //= PyList_GetItem(chromosome, 1);
        PyObject* pyContractors; //= PyList_GetItem(chromosome, 2);

        if (!PyArg_ParseTuple(chromosome, "OOO", &pyOrder, &pyResources, &pyContractors)) {
            cerr << "Can't parse chromosome!!!!" << endl;
            return -1;
        }

        int* order = (int*) PyArray_DATA((PyArrayObject*) pyOrder);
        int* resources = (int*) PyArray_DATA((PyArrayObject*) pyResources);

        int worksCount = PyArray_DIM(pyOrder, 0);  // without inseparables
        int resourcesCount = PyArray_DIM(pyResources, 1) - 1;
//        cout << worksCount << " works, " << resourcesCount << " resources" << endl;

        int stride_works = PyArray_STRIDE(pyResources, 0);
//        cout << "Stride: " << stride_works << endl;

        Timeline timeline = createTimeline();

        auto completed = vector<int>();
        completed.resize(totalWorksCount);

        int finishTime = 0;

//        cout << "Resources: ";
        // scheduling works one-by-one
        for (int i = 0; i < worksCount; i++) {
            int workIndex = order[i];
            auto* team = (int*) PyArray_GETPTR1(pyResources, workIndex);
            // = resources + workIndex * (resourcesCount + 1); // go to the start of 'i' row in 2D array
            int contractor = team[resourcesCount];

//            for (size_t j = 0; j < resourcesCount; j++) {
//                int* worker_count = (int*) PyArray_GETPTR2(pyResources, workIndex, j);
//                cout << *worker_count << " ";
//            }
//            cout << endl;

            int st = findMinStartTime(workIndex, contractor, pyResources,
                                      resourcesCount, completed, timeline);
            if (st == TIME_INF) {
                return TIME_INF;
            }
            int c_ft = schedule(chromosome_ind, workIndex, st, contractor, pyResources,
                                resourcesCount, completed, timeline);
            finishTime = max(finishTime, c_ft);
        }

        return finishTime;
    }

//    int testEvaluate(vector<int>& order, vector<vector<int>>& resources) {
//        size_t worksCount = order.size();
//        size_t resourcesCount = resources[0].size() - 1;
//
//        Timeline timeline = createTimeline();
//
//        auto completed = vector<int>();
//        completed.resize(totalWorksCount);
//
//        int finishTime = 0;
//
//        // scheduling works one-by-one
//        for (int i = 0; i < worksCount; i++) {
//            int workIndex = order[i];
//            int contractor = resources[i][resourcesCount];
//            auto* team = resources[i].begin().operator->();
//
//            int st = findMinStartTime(workIndex, contractor, team,
//                                      resourcesCount, completed, timeline);
//            int c_ft = schedule(0, workIndex, st, contractor, team,
//                                resourcesCount, completed, timeline);
////            int c_ft = 0 + 5;
//            finishTime = max(finishTime, c_ft);
//        }
//
//        return finishTime;
//    }
};


#endif //NATIVE_CHROMOSOME_EVALUATOR_H
