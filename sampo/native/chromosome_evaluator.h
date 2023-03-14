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

#include <vector>
#include <iostream>
#include <unordered_map>

using namespace std;

// worker -> contractor -> vector<time, count> in descending order
typedef vector<vector<vector<pair<int, int>>>> Timeline;

#define TIME_INF 2000000000

class ChromosomeEvaluator {
private:
    const vector<vector<int>>& parents;      // vertices' parents
    const vector<vector<int>>& inseparables; // inseparable chains with self
    const vector<vector<int>>& workers;      // contractor -> worker -> count
    int totalWorksCount;
    PyObject* pythonWrapper;

    int calculate_working_time(int chromosome_ind, int work, int team_target) {
        auto res = PyObject_CallMethod(pythonWrapper, "calculate_working_time", "(iii)", chromosome_ind, team_target, work);
        if (res == nullptr) {
            cerr << "Result is NULL" << endl << flush;
            return 0;
        }
        Py_DECREF(res);
        return (int) PyLong_AsLong(res);
    }

    int findMinStartTime(int nodeIndex, int contractor, const int* team, size_t teamSize,
                         vector<int>& completed, Timeline& timeline) {
        int maxParentTime = 0;
        // find min start time
        for (int parent : parents[nodeIndex]) {
            maxParentTime = max(maxParentTime, completed[parent]);
        }

        int maxAgentTime = 0;

        const int* worker_team = team;
        for (int worker = 0; worker < teamSize; worker++) {
            int worker_count = worker_team[worker];
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

    static void updateTimeline(int finishTime, int contractor, const int* team, size_t teamSize, Timeline& timeline) {
        for (int worker = 0; worker < teamSize; worker++) {
            int worker_count = team[worker];
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

    int schedule(int chromosome_ind, int nodeIndex, int startTime, int contractor, const int* team,
                 size_t teamSize, vector<int>& completed, Timeline& timeline) {
        int finishTime = startTime;

        for (int dep_node : inseparables[nodeIndex]) {
            int maxParentTime = 0;
            // find min start time
            for (int parent : parents[dep_node]) {
                maxParentTime = max(maxParentTime, completed[parent]);
            }
            startTime = max(startTime, maxParentTime);

            int workingTime = calculate_working_time(chromosome_ind, dep_node, nodeIndex);
            finishTime = startTime + workingTime;

            // cache finish time of scheduled work
            completed[dep_node] = finishTime;
        }

        updateTimeline(finishTime, contractor, team, teamSize, timeline);

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
    explicit ChromosomeEvaluator(const vector<vector<int>>& parents,
                                 const vector<vector<int>>& inseparables,
                                 const vector<vector<int>>& workers,
                                 int totalWorksCount,
                                 PyObject* pythonWrapper) : parents(parents), inseparables(inseparables), workers(workers) {
        this->totalWorksCount = totalWorksCount;
        this->pythonWrapper = pythonWrapper;
    }

    ~ChromosomeEvaluator() = default;

    vector<int> evaluate(vector<PyObject*>& chromosomes) {
        auto results = vector<int>();
        int i = 0;
        for (auto* chromosome : chromosomes) {
            results.push_back(evaluate(i++, chromosome));
        }
        return results;
    }

    int evaluate(int chromosome_ind, PyObject* chromosome) {
        PyObject* pyOrder; //= PyList_GetItem(chromosome, 0);
        PyObject* pyResources; //= PyList_GetItem(chromosome, 1);
        PyObject* pyContractors; //= PyList_GetItem(chromosome, 1);

        if (!PyArg_ParseTuple(chromosome, "OOO", &pyOrder, &pyResources, &pyContractors)) {
            cerr << "Can't parse chromosome!!!!" << endl;
            return -1;
        }

        int* order = (int*) PyArray_DATA((PyArrayObject*) pyOrder);
        int* resources = (int*) PyArray_DATA((PyArrayObject*) pyResources);

        int worksCount = PyArray_DIM(pyOrder, 0);  // without inseparables
        int resourcesCount = PyArray_DIM(pyResources, 1) - 1;

        Timeline timeline = createTimeline();

        auto completed = vector<int>();
        completed.resize(totalWorksCount);

        int finishTime = 0;

        // scheduling works one-by-one
        for (int i = 0; i < worksCount; i++) {
            int workIndex = order[i];
            auto* team = resources + i * (resourcesCount + 1); // go to the start of 'i' row in 2D array
            int contractor = team[resourcesCount];

            int st = findMinStartTime(workIndex, contractor, team,
                                      resourcesCount, completed, timeline);
            if (st == TIME_INF) {
                return TIME_INF;
            }
            int c_ft = schedule(chromosome_ind, workIndex, st, contractor, team,
                                resourcesCount, completed, timeline);
            finishTime = max(finishTime, c_ft);
        }

        return finishTime;
    }

    int testEvaluate(vector<int>& order, vector<vector<int>>& resources) {
        size_t worksCount = order.size();
        size_t resourcesCount = resources[0].size() - 1;

        Timeline timeline = createTimeline();

        auto completed = vector<int>();
        completed.resize(totalWorksCount);

        int finishTime = 0;

        // scheduling works one-by-one
        for (int i = 0; i < worksCount; i++) {
            int workIndex = order[i];
            int contractor = resources[i][resourcesCount];
            auto* team = resources[i].begin().operator->();

            int st = findMinStartTime(workIndex, contractor, team,
                                      resourcesCount, completed, timeline);
            int c_ft = schedule(0, workIndex, st, contractor, team,
                                resourcesCount, completed, timeline);
//            int c_ft = 0 + 5;
            finishTime = max(finishTime, c_ft);
        }

        return finishTime;
    }
};


#endif //NATIVE_CHROMOSOME_EVALUATOR_H
