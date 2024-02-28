#ifndef SAMPO_SPEC_H
#define SAMPO_SPEC_H

#include <unordered_map>
#include <string>
#include <utility>

#include "workgraph.h"
#include "dtime.h"

using namespace std;

class WorkSpec {
public:
    unordered_map<string, int> assigned_workers;
    Time assigned_time;
    bool is_independent;

    explicit WorkSpec(unordered_map<string, int> assigned_workers = unordered_map<string, int>(),
                      Time assigned_time = Time::unassigned(),
                      bool is_independent = false)
        : assigned_workers(std::move(assigned_workers)),
          assigned_time(assigned_time),
          is_independent(is_independent) {}
};

class ScheduleSpec {
public:
    unordered_map<string, WorkSpec> work2spec;

    explicit ScheduleSpec(unordered_map<string, WorkSpec> work2spec = unordered_map<string, WorkSpec>())
        : work2spec(std::move(work2spec)) {}
};

#endif //SAMPO_SPEC_H
