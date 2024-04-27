#ifndef NATIVE_TIME_ESTIMATOR_H
#define NATIVE_TIME_ESTIMATOR_H

#include <cmath>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "native/schemas/workgraph.h"
#include "native/utils.h"

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

    Time estimateTime(const WorkUnit &work, const vector<Worker> &workers) const override;
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
