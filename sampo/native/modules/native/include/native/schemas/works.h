#ifndef SAMPO_WORKS_H
#define SAMPO_WORKS_H

#include <string>
#include <vector>

#include "dtime.h"
#include "basic_types.h"

using namespace std;

class WorkerReq {
public:
    string kind;
    Time volume;
    int min_count;
    int max_count;

    WorkerReq(string kind = "", const Time &volume = Time(0), int min_count = 0, int max_count = 0)
            : kind(std::move(kind)),
              volume(volume),
              min_count(min_count),
              max_count(max_count) { }
};

class WorkUnit : public Identifiable {
public:
    std::vector<WorkerReq> worker_reqs;
    string name;
    float volume;
    bool isServiceUnit;

    // TODO Add id
    explicit WorkUnit(
            string name = "",
            const std::vector<WorkerReq> &worker_reqs = std::vector<WorkerReq>(),
            float volume                              = 1,
            bool isServiceUnit                        = false
    )
            : worker_reqs(worker_reqs),
              name(std::move(name)),
              volume(volume),
              isServiceUnit(isServiceUnit) { }
};

#endif //SAMPO_WORKS_H
