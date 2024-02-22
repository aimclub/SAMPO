#ifndef SAMPO_TIMELINE_H
#define SAMPO_TIMELINE_H

#include "native/schemas/dtime.h"

class Timeline {
public:
    Time find_min_start_time(GraphNode *node,
                             vector<Worker>& worker_team,
                             swork_dict_t &node2swork,
                             ScheduleSpec &spec,
                             Time &parent_time,
                             WorkTimeEstimator &work_estimator) {
        auto t = this->find_min_start_time_with_additional(node, worker_team, node2swork,
                                                           spec, Time.unassigned(),
                                                           parent_time, work_estimator);
        return get<0>(t);
    }

    virtual tuple<Time, Time, exec_times_t> find_min_start_time_with_additional(GraphNode *node,
                                                                                vector<Worker>& worker_team,
                                                                                swork_dict_t &node2swork,
                                                                                ScheduleSpec &spec,
                                                                                Time &assigned_start_time,
                                                                                Time &assigned_parent_time,
                                                                                WorkTimeEstimator &work_estimator) = 0;

    virtual bool can_schedule_at_the_moment(GraphNode *node,
                                            vector<Worker>& worker_team,
                                            swork_dict_t &node2swork,
                                            ScheduleSpec &spec,
                                            Time &start_time,
                                            Time &exec_time) = 0;

    virtual void update_timeline(GraphNode *node,
                                 vector<Worker>& worker_team,
                                 ScheduleSpec &spec,
                                 Time &finish_time,
                                 Time &exec_time) = 0;

    virtual Time schedule(GraphNode *node,
                          vector<Worker>& worker_team,
                          swork_dict_t &node2swork,
                          ScheduleSpec &spec,
                          Contractor *contractor,
                          Time &assigned_start_time,
                          Time &assigned_time,
                          Time &assigned_parent_time,
                          WorkTimeEstimator &work_estimator) = 0;
};

#endif //SAMPO_TIMELINE_H
