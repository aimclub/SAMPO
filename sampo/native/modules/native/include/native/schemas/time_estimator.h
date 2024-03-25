#ifndef NATIVE_TIME_ESTIMATOR_H
#define NATIVE_TIME_ESTIMATOR_H

#include <cmath>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "workgraph.h"

#define TIME_INF 2000000000

using namespace std;

class WorkTimeEstimator {
private:
    int mode = 0;    // 0 - 10%, 1 - 50%, 2 - 90%
    string path;

public:
    explicit WorkTimeEstimator(string path) : path(std::move(path)) { }

    void setMode(int mode) {
        this->mode = mode;
    }

    string &getPath() {
        return this->path;
    }

    virtual Time estimateTime(const WorkUnit &work, const vector<Worker> &workers) const = 0;

    virtual ~WorkTimeEstimator() = default;
};

class DefaultWorkTimeEstimator : public WorkTimeEstimator {
private:
    inline static float get_productivity(int worker_count) {
        // TODO
        return 1.0F * (float)worker_count;
    }

    inline static float communication_coefficient(int workerCount, int maxWorkerCount) {
        int n = workerCount;
        int m = maxWorkerCount;
        return 1 / (float)(6 * m * m) * (float)(-2 * n * n * n + 3 * n * n + (6 * m * m - 1) * n);
    }

public:
    DefaultWorkTimeEstimator() : WorkTimeEstimator("") { }

    //    ~DefaultWorkTimeEstimator() override = default;

    Time estimateTime(const WorkUnit &work, const vector<Worker> &workers) const override {
        // the _abstract_estimate from WorkUnit
        int time = 0;

        // TODO Rework with OOP

        // build worker_req index
        unordered_map<string, WorkerReq> worker_reqs;
        for (const auto& worker_req : work.worker_reqs) {
            worker_reqs[worker_req.kind] = worker_req;
        }

        for (const auto& worker : workers) {
            const auto& worker_req = worker_reqs[worker.name];
            if (worker_req.min_count == 0)
                continue;
            int actual_count = worker.count;
            if (actual_count < worker_req.min_count) {
                cout << "Not conforms to min_req: " << worker.name << " count: "
                     << actual_count << ", required: " << worker_req.min_count << endl;
                return TIME_INF;
            }
            int max_req = worker_req.max_count;

            float productivity = get_productivity(actual_count);
            productivity *= communication_coefficient(actual_count, max_req);

            //        if (productivity < 0.000001) {
            //            return TIME_INF;
            //        }
            //        productivity = 0.1;
            int new_time = ceil((float) worker_req.volume.val() / productivity);
            if (new_time > time) {
                time = new_time;
            }
        }

        return Time(time);
    }
};

// class ExternalWorkTimeEstimator : public WorkTimeEstimator {
// public:
//     explicit ExternalWorkTimeEstimator(const string& path) :
//     WorkTimeEstimator(path) {}
//
//     int estimateTime(const string &work, float volume, vector<pair<string,
//     int>> &resources) override {
//
//     };
// };

// DLL compat
class ITimeEstimatorLibrary {
public:
    virtual WorkTimeEstimator *create(const string &path) = 0;
};

#endif    // NATIVE_TIME_ESTIMATOR_H
