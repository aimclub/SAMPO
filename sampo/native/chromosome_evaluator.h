#ifndef NATIVE_CHROMOSOME_EVALUATOR_H
#define NATIVE_CHROMOSOME_EVALUATOR_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"
#include "pycodec.h"

#include <vector>
#include <iostream>

using namespace std;

// worker -> contractor -> vector<time, count> in descending order
typedef vector<vector<vector<pair<int, int>>>> Timeline;

class ChromosomeEvaluator {
private:
    const vector<vector<int>>& parents;      // vertices' parents
    const vector<vector<int>>& inseparables; // inseparable chains with self
    const vector<vector<int>>& workers;      // contractor -> worker -> count
    int totalWorksCount;
    PyObject* pythonWrapper;

    int calculate_working_time(int work, int contractor, PyArrayObject* team) {
        // TODO Make full hash-based cache

        PyObject* my_args = PyTuple_Pack(
                3,
                PyLong_FromLong(work),
                PyLong_FromLong(contractor),
                team
        );
        if (!my_args) {
            cout << "Problems with tuple allocating; probably end of RAM" << endl;
            return 0;
        }
        auto res = PyObject_CallMethodObjArgs(pythonWrapper,
                                              PyUnicode_FromString("calculate_working_time"),
                                              my_args, NULL);
        return (int) PyLong_AsLong(res);
    }

    static inline int get(PyArrayObject* arr, int i) {
        // very hacky, but simply getting element from numpy array
        return *(int*) PyArray_GETPTR1(arr, i);
    }

    static inline int get(PyArrayObject* arr, int i, int j) {
        return *(int*) PyArray_GETPTR2(arr, i, j);
    }

    int findMinStartTime(int nodeIndex, int contractor, const int* team, int teamSize,
                         vector<int>& completed, Timeline& timeline) {
        int maxParentTime = 0;
        // find min start time
//        cout << "nodeIndex: " << nodeIndex << endl;
        for (int parent : parents[nodeIndex]) {
            maxParentTime = max(maxParentTime, completed[parent]);
        }

        int maxAgentTime = 0;

        const int* worker_team = team;
        for (int worker = 0; worker < teamSize; worker++) {
            int worker_count = worker_team[worker];
            int need_count = worker_count;
            if (need_count == 0) continue;

//            cout << "contractor: " << contractor << " worker: " << worker << endl;
            // Traverse list while not enough resources and grab it
            auto &worker_timeline = timeline[contractor][worker];
            int ind = worker_timeline.size() - 1;
            while (need_count > 0) {
//                cout << "ind: " << ind << endl;
                int offer_count = worker_timeline[ind].second;
                maxAgentTime = max(maxAgentTime, worker_timeline[ind].first);

                if (offer_count < need_count) {
                    offer_count = need_count;
                }
                need_count -= offer_count;
                ind--;
            }
        }

        return max(maxParentTime, maxAgentTime);
    }

    static void updateTimeline(int finishTime, int contractor, const int* team, int teamSize, Timeline& timeline) {
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
                worker_timeline.pop_back();
            }

            // Add to the right place using bubble-sort iterations
            worker_timeline.emplace_back(finishTime, worker_count);
            int ind = worker_timeline.size() - 1;
            while (ind > 0 && worker_timeline[ind].first > worker_timeline[ind - 1].first) {
                auto tmp = worker_timeline[ind];
                worker_timeline[ind] = worker_timeline[ind - 1];
                worker_timeline[ind - 1] = tmp;
                ind--;
            }
        }
    }

    int schedule(int nodeIndex, int startTime, int contractor, const int* team,
                 int teamSize, vector<int>& completed, Timeline& timeline) {
        int finishTime = startTime;

        for (int dep_node : inseparables[nodeIndex]) {
            int maxParentTime = 0;
            // find min start time
            for (int parent : parents[dep_node]) {
                maxParentTime = max(maxParentTime, completed[parent]);
            }
            startTime = max(startTime, maxParentTime);

            int workingTime = 1;//calculate_working_time(dep_node, contractor, team);
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

    static PyObject* pyObjectIdentity(PyObject* object) {
        return object;
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

    ~ChromosomeEvaluator() {}

    vector<int> evaluate(vector<PyObject*>& chromosomes) {
        auto results = vector<int>();
        for (auto* chromosome : chromosomes) {
            results.push_back(evaluate(chromosome));
        }
        return results;
    }

    int evaluate(PyObject* chromosome) {
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
//        cout << "Scheduling" << endl;


        // scheduling works one-by-one
        for (int i = 0; i < worksCount; i++) {
//        int i = 0;
            int workIndex = order[i];
            auto* team = resources + i * (resourcesCount + 1); // go to the start of 'i' row in 2D array
            int contractor = team[resourcesCount];
//
//            cout << "Work index: " << workIndex << endl << flush;
//            cout << i << "," << resourcesCount << " Contractor: " << contractor << endl << flush;

//            for (int worker = 0; worker < resourcesCount; worker++) {
//                int worker_count = team[worker];
//                cout << "contractor: " << contractor << " worker: " << worker << ", worker_count: " << worker_count << endl;
//
//                int need_count = worker_count;
//                if (need_count == 0) continue;
//            }

            int st = findMinStartTime(workIndex, contractor, team,
                                      resourcesCount, completed, timeline);
//            int c_ft = 0 + 5;
            int c_ft = schedule(workIndex, st, contractor, team,
                                resourcesCount, completed, timeline);
            finishTime = max(finishTime, c_ft);
        }
//        cout << "Finish time: " << finishTime << endl;

        return finishTime;
    }

    int testEvaluate(vector<int>& order, vector<vector<int>>& resources) {
        int worksCount = order.size();
        int resourcesCount = resources[0].size() - 1;

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
            int c_ft = schedule(workIndex, st, contractor, team,
                                resourcesCount, completed, timeline);
//            int c_ft = 0 + 5;
            finishTime = max(finishTime, c_ft);
        }

        return finishTime;
    }
};


#endif //NATIVE_CHROMOSOME_EVALUATOR_H
