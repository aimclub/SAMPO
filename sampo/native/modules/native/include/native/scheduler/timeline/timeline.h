#ifndef SAMPO_TIMELINE_H
#define SAMPO_TIMELINE_H

#include "native/schemas/dtime.h"
#include "native/schemas/landscape.h"
#include "native/schemas/evaluator_types.h"
#include "native/schemas/workgraph.h"
#include "native/schemas/spec.h"
#include "native/schemas/scheduled_work.h"
#include "native/schemas/time_estimator.h"

#include "native/scheduler/timeline/timeline.h"

class Timeline {
public:
    Time find_min_start_time(GraphNode *node,
                             vector<Worker>& worker_team,
                             swork_dict_t &node2swork,
                             WorkSpec &spec,
                             Time parent_time,
                             const WorkTimeEstimator &work_estimator) {
        auto t = this->find_min_start_time_with_additional(node, worker_team, node2swork,
                                                           spec, Time::unassigned(),
                                                           parent_time, work_estimator);
        return get<0>(t);
    }

    Time schedule_with_inseparables(GraphNode *node,
                                    vector<Worker>& worker_team,
                                    swork_dict_t &node2swork,
                                    WorkSpec &spec,
                                    Contractor *contractor,
                                    Time start_time,
                                    vector<GraphNode*> &inseparable_chain,
                                    exec_times_t &exec_times,
                                    const WorkTimeEstimator &work_estimator) {
        Time c_ft = start_time;
        for (auto& dep_node : inseparable_chain) {
            Time max_parent_time = dep_node->min_start_time(node2swork);

            pair<Time, Time> node_lag_exec_time;
            auto it = exec_times.find(dep_node->id());
//            cout << "111" << endl;
            if (it == exec_times.end()) {
                node_lag_exec_time = { Time(0), work_estimator.estimateTime(*node->getWorkUnit(), worker_team) };
            } else {
                node_lag_exec_time = it->second;
            }
//            cout << "222" << endl;

            Time c_st = max(c_ft + node_lag_exec_time.first, max_parent_time);
            Time new_finish_time = c_st + node_lag_exec_time.second;

            vector<MaterialDelivery> deliveries;

            node2swork[dep_node->id()] = ScheduledWork(
                    dep_node->getWorkUnit(),
                    { c_st, new_finish_time },
                    worker_team, contractor, vector<Equipment>(), deliveries, ConstructionObject());
//            cout << "333" << endl;
            c_ft = new_finish_time;
        }

//        cout << "Works scheduled, update timeline start" << endl;

        this->update_timeline(node, worker_team, spec, c_ft, c_ft - start_time);
        return c_ft;
    }

    virtual tuple<Time, Time, exec_times_t> find_min_start_time_with_additional(GraphNode *node,
                                                                                vector<Worker>& worker_team,
                                                                                swork_dict_t &node2swork,
                                                                                WorkSpec &spec,
                                                                                Time assigned_start_time,
                                                                                Time assigned_parent_time,
                                                                                const WorkTimeEstimator &work_estimator) const = 0;

    virtual bool can_schedule_at_the_moment(GraphNode *node,
                                            vector<Worker>& worker_team,
                                            swork_dict_t &node2swork,
                                            WorkSpec &spec,
                                            Time start_time,
                                            Time exec_time) const = 0;

    virtual void update_timeline(GraphNode *node,
                                 vector<Worker>& worker_team,
                                 WorkSpec &spec,
                                 Time finish_time,
                                 Time exec_time) = 0;

    virtual Time schedule(GraphNode *node,
                          vector<Worker>& worker_team,
                          swork_dict_t &node2swork,
                          WorkSpec &spec,
                          Contractor *contractor,
                          Time assigned_start_time,
                          Time assigned_time,
                          Time assigned_parent_time,
                          const WorkTimeEstimator &work_estimator) = 0;
};

#endif //SAMPO_TIMELINE_H
