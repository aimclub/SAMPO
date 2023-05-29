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

    int calculate_working_time(int chromosome_ind, int work, int team_target, const int* resources, size_t teamSize) {
        if (useExternalWorkEstimator) {
            auto res = PyObject_CallMethod(pythonWrapper, "calculate_working_time", "(iii)",
                                           chromosome_ind, team_target, work);
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
                if (resources[i] < minReq) {
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

                float productivity = get_productivity(i, resources[i]);
                productivity *= communication_coefficient(resources[i], maxReq);

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

    int findMinStartTime(int nodeIndex, int contractor, const int* resources, size_t teamSize,
                         vector<int>& completed, Timeline& timeline) {
        int maxParentTime = 0;
        // find min start time
        for (int parent : parents[nodeIndex]) {
            maxParentTime = max(maxParentTime, completed[parent]);
        }

        int maxAgentTime = 0;

        for (int worker = 0; worker < teamSize; worker++) {
            int worker_count = 1;//resources[worker];
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

    static void updateTimeline(int finishTime, int contractor, const int* resources, size_t teamSize, Timeline& timeline) {
        for (int worker = 0; worker < teamSize; worker++) {
            int worker_count = resources[worker];
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

    int schedule(int chromosome_ind, int nodeIndex, int startTime, int contractor, const int* resources,
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

        updateTimeline(finishTime, contractor, resources, teamSize, timeline);

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

    vector<int> evaluate(vector<Chromosome*>& chromosomes) {
        auto results = vector<int>();
        results.resize(chromosomes.size());

//        #pragma omp parallel for firstprivate(chromosomes) shared(results) default (none) num_threads(this->numThreads)
        for (int i = 0; i < chromosomes.size(); i++) {
            results[i] = evaluate(i, chromosomes[i]);
        }
        return results;
    }

    int evaluate(int chromosome_ind, Chromosome* chromosome) {
        Timeline timeline = createTimeline();

        auto completed = vector<int>();
        completed.resize(totalWorksCount);

        int finishTime = 0;

        // scheduling works one-by-one
        for (int i = 0; i < chromosome->numWorks(); i++) {
            int workIndex = *chromosome->getOrder()[i];
            int* team = chromosome->getResources()[workIndex];
            int contractor = chromosome->getContractor(workIndex);

            int st = findMinStartTime(workIndex, contractor, team,
                                      chromosome->numResources(), completed, timeline);
            if (st == TIME_INF) {
                return TIME_INF;
            }
            int c_ft = schedule(chromosome_ind, workIndex, st, contractor, team,
                                chromosome->numResources(), completed, timeline);
            finishTime = max(finishTime, c_ft);
        }

        return finishTime;
    }
};


#endif //NATIVE_CHROMOSOME_EVALUATOR_H
