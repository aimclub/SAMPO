#ifndef SAMPO_JUST_IN_TIME_H
#define SAMPO_JUST_IN_TIME_H

#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>

#include "native/scheduler/timeline/timeline.h"
#include "native/schemas/landscape.h"
#include "native/schemas/evaluator_types.h"
#include "native/schemas/workgraph.h"
#include "native/schemas/spec.h"
#include "native/schemas/scheduled_work.h"
#include "native/schemas/time_estimator.h"

using namespace std;

class JustInTimeTimeline : public Timeline {
private:
    unordered_map<string, unordered_map<string, vector<pair<Time, int>>>> timeline;
public:
    explicit JustInTimeTimeline(const worker_pool_t &worker_pool, const LandscapeConfiguration &landscape);

    JustInTimeTimeline(const JustInTimeTimeline &other);

    virtual ~JustInTimeTimeline() = default;

    tuple<Time, Time, exec_times_t> find_min_start_time_with_additional(GraphNode *node,
                                                                        vector<Worker>& worker_team,
                                                                        swork_dict_t &node2swork,
                                                                        WorkSpec &spec,
                                                                        Time assigned_start_time,
                                                                        Time assigned_parent_time,
                                                                        const WorkTimeEstimator &work_estimator) const override;

    bool can_schedule_at_the_moment(GraphNode *node,
                                    vector<Worker>& worker_team,
                                    swork_dict_t &node2swork,
                                    WorkSpec &spec,
                                    Time start_time,
                                    Time exec_time) const override;

    void update_timeline(GraphNode *node,
                         vector<Worker>& worker_team,
                         WorkSpec &spec,
                         Time finish_time,
                         Time exec_time) override;

    Time schedule(GraphNode *node,
                  vector<Worker>& worker_team,
                  swork_dict_t &node2swork,
                  WorkSpec &spec,
                  Contractor *contractor,
                  Time assigned_start_time,
                  Time assigned_time,
                  Time assigned_parent_time,
                  WorkTimeEstimator &work_estimator) override;
};

#endif //SAMPO_JUST_IN_TIME_H
