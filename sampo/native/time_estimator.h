//
// Created by stasb on 15.06.2023.
//

#ifndef NATIVE_TIME_ESTIMATOR_H
#define NATIVE_TIME_ESTIMATOR_H

#include <string>
#include <map>
#include <vector>
#include <cmath>
#include <unordered_map>

#define TIME_INF 2000000000

using namespace std;

class WorkTimeEstimator {
private:
    int mode = 0; // 0 - 10%, 1 - 50%, 2 - 90%
    string path;

public:
    explicit WorkTimeEstimator(string path) : path(std::move(path)) {}

    void setMode(int mode) {
        this->mode = mode;
    }

    string& getPath() {
        return this->path;
    }

    virtual int estimateTime(const string &work, float volume, vector<pair<string, int>> &resources) = 0;

    virtual ~WorkTimeEstimator() = default;
};

class DefaultWorkTimeEstimator : public WorkTimeEstimator {
private:
    unordered_map<string, unordered_map<string, int>>& minReqs;  // work -> worker -> WorkUnit.max_req
    unordered_map<string, unordered_map<string, int>>& maxReqs;  // work -> worker -> WorkUnit.min_req

    inline static float get_productivity(int worker_count) {
        // TODO
        return 1.0F * (float) worker_count;
    }

    inline static float communication_coefficient(int workerCount, int maxWorkerCount) {
        int n = workerCount;
        int m = maxWorkerCount;
        return 1 / (float) (6 * m * m) * (float) (-2 * n * n * n + 3 * n * n + (6 * m * m - 1) * n);
    }

public:
    DefaultWorkTimeEstimator(unordered_map<string, unordered_map<string, int>>& minReqs,
                             unordered_map<string, unordered_map<string, int>>& maxReqs)
            : WorkTimeEstimator(""), minReqs(minReqs), maxReqs(maxReqs) {

    }

//    ~DefaultWorkTimeEstimator() override = default;

    int estimateTime(const string &work, float volume, vector<pair<string, int>> &resources) override {
        // the _abstract_estimate from WorkUnit
        int time = 0;

        for (auto& resource : resources) {
            int minReq = this->minReqs[work][resource.first];
            if (minReq == 0)
                continue;
            int actualCount = resource.second;
            if (actualCount < minReq) {
//                    cout << "Not conforms to min_req: " << get_worker(resources, team_target, i) << " < " << minReq << " on work " << work
//                         << " and worker " << i << ", chromosome " << chromosome_ind << ", teamSize=" << teamSize << endl;
//                    cout << "Team: ";
//                    for (size_t j = 0; j < teamSize; j++) {
//                        cout << get_worker(resources, team_target, i) << " ";
//                    }
//                    cout << endl;
                return TIME_INF;
            }
            int maxReq = this->maxReqs[work][resource.first];

            float productivity = get_productivity(actualCount);
            productivity *= communication_coefficient(actualCount, maxReq);

//                if (productivity < 0.000001) {
//                    return TIME_INF;
//                }
//                productivity = 0.1;
            int newTime = ceil(volume / productivity);
            if (newTime > time) {
                time = newTime;
            }
        }

        return time;
    }
};

//class ExternalWorkTimeEstimator : public WorkTimeEstimator {
//public:
//    explicit ExternalWorkTimeEstimator(const string& path) : WorkTimeEstimator(path) {}
//
//    int estimateTime(const string &work, float volume, vector<pair<string, int>> &resources) override {
//
//    };
//};

// DLL compat
class ITimeEstimatorLibrary {
public:
    virtual WorkTimeEstimator* create(const string& path) = 0;
};

#endif //NATIVE_TIME_ESTIMATOR_H
