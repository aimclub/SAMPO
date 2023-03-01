#ifndef NATIVE_CHROMOSOME_EVALUATOR_H
#define NATIVE_CHROMOSOME_EVALUATOR_H

#include <vector>
#include <iostream>

#include "Python.h"
#include "numpy/arrayobject.h"

using namespace std;

// worker -> contractor -> vector<time, count> in descending order
typedef vector<vector<vector<pair<int, int>>>> Timeline;

class ChromosomeEvaluator {
private:
    vector<vector<int>> parents;      // vertices' parents
    vector<vector<int>> inseparables; // inseparable chains with self
    vector<vector<int>> workers;      // contractor -> worker -> count
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

    int findMinStartTime(int nodeIndex, int contractor, PyArrayObject* team, int teamSize,
                         vector<int>& completed, Timeline& timeline) {
        int maxParentTime = 0;
        // find min start time
        for (int parent : parents[nodeIndex]) {
            maxParentTime = max(maxParentTime, completed[parent]);
        }

        int maxAgentTime = 0;

        for (int worker = 0; worker < teamSize; worker++) {
            int worker_count = get(team, worker);
            int need_count = worker_count;
            if (need_count == 0) continue;

            // Traverse list while not enough resources and grab it
            auto &worker_timeline = timeline[worker][contractor];
            size_t ind = worker_timeline.size() - 1;
            while (need_count > 0) {
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

    static void updateTimeline(int finishTime, int contractor, PyArrayObject* team, int teamSize, Timeline& timeline) {
        for (int worker = 0; worker < teamSize; worker++) {
            int worker_count = get(team, worker);
            int need_count = worker_count;
            if (need_count == 0) continue;

            // Consume need workers
            auto& worker_timeline = timeline[worker][contractor];
            while (need_count > 0) {
                int next_count = worker_timeline.end()->second;
                if (next_count > need_count) {
                    worker_timeline.end()->second -= need_count;
                    break;
                }
                need_count -= next_count;
                worker_timeline.pop_back();
            }

            // Add to the right place using bubble-sort iterations
            worker_timeline.emplace_back(finishTime, worker_count);
            size_t ind = worker_timeline.size() - 1;
            while (ind > 0 && worker_timeline[ind].first > worker_timeline[ind - 1].first) {
                auto& tmp = worker_timeline[ind];
                worker_timeline[ind] = worker_timeline[ind - 1];
                worker_timeline[ind - 1] = tmp;
                ind--;
            }
        }
    }

    int schedule(int nodeIndex, int startTime, int contractor, PyArrayObject* team,
                  int teamSize, vector<int>& completed, Timeline& timeline) {
        int finishTime = startTime;

        for (int dep_node : inseparables[nodeIndex]) {
            int maxParentTime = 0;
            // find min start time
            for (int parent : parents[dep_node]) {
                maxParentTime = max(maxParentTime, completed[parent]);
            }
            startTime = max(startTime, maxParentTime);

            int workingTime = calculate_working_time(dep_node, contractor, team);
            finishTime = startTime + workingTime;

            // cache finish time of scheduled work
            completed[dep_node] = finishTime;
        }

        updateTimeline(finishTime, contractor, team, teamSize, timeline);

        return finishTime;
    }

    inline Timeline createTimeline() {
        Timeline timeline = Timeline();

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
                                 PyObject* pythonWrapper) {
        this->parents = parents;
        this->inseparables = inseparables;
        this->workers = workers;
        this->totalWorksCount = totalWorksCount;
        this->pythonWrapper = pythonWrapper;
    }

    vector<int> evaluate(vector<PyObject*>& chromosomes) {
        auto results = vector<int>();
        for (auto* chromosome : chromosomes) {
            results.push_back(evaluate(chromosome));
        }
        return results;
    }

    int evaluate(PyObject* chromosome) {
        PyObject* it = PyObject_GetIter(chromosome);
        auto* order = (PyArrayObject*) (it = PyIter_Next(it));
        auto* resources = (PyArrayObject*) (it = PyIter_Next(it));
        Py_DECREF(it);

        int worksCount = PyArray_SIZE(order);  // without inseparables
        int resourcesCount = PyArray_SIZE(resources) - 1;

        Timeline timeline = createTimeline();

        auto completed = vector<int>();
        completed.resize(totalWorksCount);

        int finishTime = 0;

        // scheduling works one-by-one
        for (int i = 0; i < worksCount; i++) {
            int workIndex = get(order, i);
            int contractor = get(resources, resourcesCount, i);
            auto* team = (PyArrayObject*) PyArray_GETPTR1(chromosome, i);

            int st = findMinStartTime(workIndex, contractor, team,
                                      resourcesCount, completed, timeline);
            int c_ft = schedule(workIndex, st, contractor, team,
                                resourcesCount, completed, timeline);
            finishTime = max(finishTime, c_ft);
        }

        return finishTime;
    }
};


#endif //NATIVE_CHROMOSOME_EVALUATOR_H
