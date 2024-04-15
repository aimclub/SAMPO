#include <list>

#include "native/scheduler/sgs.h"
#include "native/scheduler/timeline/general_timeline.h"

inline vector<Worker> decode_worker_team(GraphNode* node,
                                         worker_pool_t &worker_pool,
                                         Chromosome* chromosome,
                                         int work_index,
                                         Contractor* contractor,
                                         const unordered_map<string, int> &worker_name2index) {
    vector<Worker> worker_team;
    for (auto& wreq : node->getWorkUnit()->worker_reqs) {
        auto &v = worker_pool[wreq.kind];
        int count = chromosome->getResources()[work_index][worker_name2index.at(wreq.kind)];
        worker_team.emplace_back(v[contractor->id].copy().with_count(count));
    }
    return worker_team;
}

swork_dict_t SGS::serial(Chromosome* chromosome,
                         worker_pool_t worker_pool,  // we need a full copy here, it is changing in runtime
                         const vector<vector<Worker*>> &worker_pool_indices,
                         const vector<GraphNode*> &index2node,
                         const vector<Contractor*> &index2contractor,
                         const vector<int> &index2zone,  // TODO
                         const unordered_map<string, int> &worker_name2index,
                         const unordered_map<string, int> &contractor2index,
                         const LandscapeConfiguration &landscape,
                         Time assigned_parent_time,
                         Timeline &timeline,
                         const WorkTimeEstimator &work_estimator) {
    swork_dict_t node2swork;

    for (auto& worker_state : worker_pool) {
        for (auto& contractor_state : worker_state.second) {
            contractor_state.second.with_count(
                    chromosome->getContractorBorder(contractor2index.at(contractor_state.first))
                    [worker_name2index.at(worker_state.first)]
            );
        }
    }

    for (int order_index = 0; order_index < chromosome->numWorks(); order_index++) {
        int work_index = *chromosome->getOrder()[order_index];
        GraphNode* node = index2node[work_index];
        auto& work_spec = chromosome->getSpec().work2spec[node->id()];

        int contractor_index = chromosome->getContractor(work_index);
        Contractor* contractor = index2contractor[contractor_index];

        // decompress worker team
        vector<Worker> worker_team = decode_worker_team(node, worker_pool, chromosome, work_index, contractor, worker_name2index);

//        auto[st, ft, exec_times] = timeline.find_min_start_time_with_additional(node, worker_team, node2swork,
//                                                                                work_spec, Time::unassigned(),
//                                                                                assigned_parent_time, work_estimator);

        // TODO Check if assigned_parent_time was not applied
        // TODO Check that find_min_start_time() is not calling if start time specified
        timeline.schedule(node, worker_team, node2swork, work_spec, contractor, Time::unassigned(),
                          work_spec.assigned_time, assigned_parent_time, work_estimator);

        // TODO Add other timelines
    }

    return node2swork;
}

swork_dict_t SGS::parallel(Chromosome* chromosome,
                           worker_pool_t worker_pool,  // we need a full copy here, it is changing in runtime
                           const vector<vector<Worker*>> &worker_pool_indices,
                           const vector<GraphNode*> &index2node,
                           const vector<Contractor*> &index2contractor,
                           const vector<int> &index2zone,  // TODO
                           const unordered_map<string, int> &worker_name2index,
                           const unordered_map<string, int> &contractor2index,
                           const LandscapeConfiguration &landscape,
                           Time assigned_parent_time,
                           Timeline &timeline,
                           const WorkTimeEstimator &work_estimator) {
    swork_dict_t node2swork;

    for (auto& worker_state : worker_pool) {
        for (auto& contractor_state : worker_state.second) {
            contractor_state.second.with_count(
                    chromosome->getContractorBorder(contractor2index.at(contractor_state.first))
                    [worker_name2index.at(worker_state.first)]
            );
        }
    }

    list<tuple<GraphNode*, Contractor*, vector<Worker>, WorkSpec &, Time>> enumerated_works_remaining;
    for (int i = 0; i < chromosome->numWorks(); i++) {
        int work_index = *chromosome->getOrder()[i];
        GraphNode* node = index2node[work_index];
        Contractor* contractor = index2contractor[chromosome->getContractor(work_index)];
        vector<Worker> worker_team = decode_worker_team(node, worker_pool, chromosome, work_index,
                                                        contractor, worker_name2index);
        Time exec_time = work_estimator.estimateTime(*node->getWorkUnit(), worker_team);
        enumerated_works_remaining.emplace_back(node, contractor, worker_team, exec_time);
    }

    GeneralTimeline<GraphNode*> work_timeline;

    Time start_time;
    Time pred_start_time;
    size_t cpkt_idx = 0;
    // while there are unprocessed checkpoints
    while (!enumerated_works_remaining.empty()) {
        if (cpkt_idx < work_timeline.size()) {
            start_time = get<0>(work_timeline[cpkt_idx]);
            if (pred_start_time == start_time) {
                cpkt_idx++;
                continue;
            }
            if (start_time.is_inf()) {
                // break because schedule already contains Time.inf(), that is incorrect schedule
                break;
            }
            pred_start_time = start_time;
        } else {
            start_time++;
        }

        enumerated_works_remaining.remove_if(
            [&assigned_parent_time, &start_time, &timeline, &node2swork, &work_estimator, &work_timeline]
                    (tuple<GraphNode *, Contractor *, vector<Worker>, WorkSpec &, Time> &decoded) {
                const auto &[node, contractor, worker_team, work_spec, exec_time] = decoded;

                if (timeline.can_schedule_at_the_moment(node, worker_team, node2swork,
                                                        work_spec, start_time, exec_time)) {
                    // TODO Apply spec
                    Time st = start_time;
                    if (node->parents().empty()) {
                        st = assigned_parent_time;
                    }

                    // TODO Add finalizing zones processing

                    timeline.schedule(node, worker_team, node2swork, work_spec, contractor,
                                      st, exec_time, assigned_parent_time, work_estimator);

                    work_timeline.update_timeline(st, exec_time, node);
                    return true;
                }

                return false;
        });
        cpkt_idx = min(cpkt_idx, work_timeline.size());
    }

    return node2swork;
}
