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

#include <vector>
#include <iostream>
#include <unordered_map>
#include <omp.h>
#include <set>

#include "pycodec.h"
#include "evaluator_types.h"
#include "time_estimator.h"
#include "DLLoader.h"
#include "external.h"

// worker -> contractor -> vector<time, count> in descending order
typedef vector<vector<vector<pair<int, int>>>> Timeline;

#define TIME_INF 2000000000

class ChromosomeEvaluator {
private:
    const vector<vector<int>>& parents;      // vertices' parents
    const vector<vector<int>>& headParents;  // vertices' parents without inseparables
    const vector<vector<int>>& inseparables; // inseparable chains with self
    const vector<vector<int>>& workers;      // contractor -> worker -> count
    const vector<float>& volumes;           // work -> worker -> WorkUnit.min_req
    const vector<vector<int>>& minReqs;      // work -> worker -> WorkUnit.max_req
    const vector<vector<int>>& maxReqs;      // work -> WorkUnit.volume
    const vector<string>& id2work;
    const vector<string>& id2res;

    unordered_map<string, unordered_map<string, int>> minReqNames;
    unordered_map<string, unordered_map<string, int>> maxReqNames;

    int totalWorksCount;
    PyObject* pythonWrapper;
    bool usePythonWorkEstimator;

    WorkTimeEstimator* timeEstimator;
    dlloader::DLLoader<ITimeEstimatorLibrary> loader { External::timeEstimatorLibPath };

    int calculate_working_time(int chromosome_ind, int work, int team_target, const int* resources, size_t teamSize) {
        if (usePythonWorkEstimator) {
            auto res = PyObject_CallMethod(pythonWrapper, "calculate_working_time_ind", "(iii)",
                                           chromosome_ind, team_target, work);
            if (res == nullptr) {
                cerr << "Result is NULL" << endl << flush;
                return 0;
            }
            Py_DECREF(res);
            return (int) PyLong_AsLong(res);
        } else {
            // map resources from indices to names
            vector<pair<string, int>> resourcesWithNames;
            for (int i = 0; i < teamSize; i++) {
                if (resources[i] != 0) {
                    resourcesWithNames.emplace_back(id2res[i], resources[i]);
                }
            }
//            cout << "Called WorkTimeEstimator!" << endl;
//            return 1;
            return calculate_working_time(chromosome_ind, id2work[work], id2work[team_target], volumes[work], resourcesWithNames);
        }
    }

    int calculate_working_time(int chromosome_ind, const string& work, const string& team_target, float volume, vector<pair<string, int>> &resources) {
        if (usePythonWorkEstimator) {
            auto res = PyObject_CallMethod(pythonWrapper, "calculate_working_time", "(iss)",
                                           chromosome_ind, team_target.data(), work.data());
            if (res == nullptr) {
                cerr << "Result is NULL" << endl << flush;
                return 0;
            }
            Py_DECREF(res);
            return (int) PyLong_AsLong(res);
        } else {
            return timeEstimator->estimateTime(work, volume, resources);
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
            int worker_count = resources[worker];
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
//                    cerr << "Not enough workers" << endl;
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
    int numThreads;

    explicit ChromosomeEvaluator(EvaluateInfo* info)
        : parents(info->parents), headParents(info->headParents), inseparables(info->inseparables), workers(info->workers),
          minReqs(info->minReq), maxReqs(info->maxReq), volumes(info->volume), id2work(info->id2work), id2res(info->id2res) {
        this->totalWorksCount = info->totalWorksCount;
        this->pythonWrapper = info->pythonWrapper;

        for (int i = 0; i < minReqs.size(); i++) {
            auto& workName = id2work[i];
            minReqNames[workName] = unordered_map<string, int>();
            maxReqNames[workName] = unordered_map<string, int>();
            for (int j = 0; j < minReqs[i].size(); j++) {
                if (minReqs[i][j] != 0) {
                    auto& resName = id2res[j];
                    minReqNames[workName][resName] = minReqs[i][j];
                    maxReqNames[workName][resName] = maxReqs[i][j];
                }
            }
        }

        this->usePythonWorkEstimator = info->usePythonWorkEstimator;
        this->numThreads = this->usePythonWorkEstimator ? 1 : omp_get_num_procs();
        printf("Genetic running threads: %i\n", this->numThreads);

        if (info->useExternalWorkEstimator) {
            loader.DLOpenLib();
            auto library = loader.DLGetInstance();
            this->timeEstimator = library->create(info->timeEstimatorPath);
        } else if (!usePythonWorkEstimator) {
            this->timeEstimator = new DefaultWorkTimeEstimator(minReqNames, maxReqNames);
        }
    }

    // TODO Research why deleting timeEstimator causes Head Corruption crash
//    ~ChromosomeEvaluator() {
//        delete timeEstimator;
//    }
//    ~ChromosomeEvaluator() {
//        loader.DLCloseLib();
//    }
    ~ChromosomeEvaluator() = default;

    bool isValid(Chromosome* chromosome) {
        bool* visited = new bool[chromosome->numWorks()] { false };

        // check edges
        for (int i = 0; i < chromosome->numWorks(); i++) {
            int node = *chromosome->getOrder()[i];
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
                if (count < minReqs[node][res] || count > chromosome->getContractors()[contractor][res]) {
                    return false;
                }
            }
        }

        return true;
    }

    void evaluate(vector<Chromosome*>& chromosomes) {
        #pragma omp parallel for shared(chromosomes) default (none) num_threads(this->numThreads)
        for (int i = 0; i < chromosomes.size(); i++) {
            if (isValid(chromosomes[i])) {
                chromosomes[i]->fitness = evaluate(i, chromosomes[i]);
            } else {
                chromosomes[i]->fitness = INT_MAX;
            }
        }
    }

    int evaluate(int chromosome_ind, Chromosome* chromosome) {
        Timeline timeline = createTimeline();

        auto completed = vector<int>();
        completed.resize(totalWorksCount);

//        cout << "Evaluated" << endl;

        int finishTime = 0;

//        for (int w = 0; w < chromosome->numWorks(); w++) {
//            cout << *chromosome->getOrder()[w] << " ";
//        }
//        cout << endl;

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
