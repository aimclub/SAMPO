#include "native/scheduler/sgs.h"

swork_dict_t SGS::serial(Chromosome* chromosome,
                         worker_pool_t worker_pool,  // we need a full copy here, it is changing in runtime
                         vector<vector<Worker>> &worker_pool_indices,
                         vector<GraphNode*> &index2node,
                         vector<Contractor*> &index2contractor,
                         vector<int> &index2zone,  // TODO
                         unordered_map<string, int> &worker_name2index,
                         unordered_map<string, int> &contractor2index,
                         LandscapeConfiguration &landscape,
                         Time assigned_parent_time,
                         Timeline &timeline,
                         WorkTimeEstimator &work_estimator) {
    swork_dict_t node2swork;

    for (auto& worker_state : worker_pool) {
        for (auto& contractor_state : worker_state.second) {
            contractor_state.second.with_count(
                    chromosome->getContractorBorder(contractor2index[contractor_state.first])
                    [worker_name2index[worker_state.first]]
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
        vector<Worker> worker_team;
        for (auto& wreq : node->getWorkUnit()->worker_reqs) {
            auto &v = worker_pool[contractor->id];
            auto &ref = v[wreq.kind];
            worker_team.emplace_back();
        }

        auto[st, ft, exec_times] = timeline.find_min_start_time_with_additional(node, worker_team, node2swork,
                                                                                work_spec, Time::unassigned(),
                                                                                assigned_parent_time, work_estimator);

        // TODO Check if assigned_parent_time was not applied
        // TODO Check that find_min_start_time() is not calling if start time specified
        ft = timeline.schedule(node, worker_team, node2swork, work_spec, contractor, st,
                               work_spec.assigned_time, assigned_parent_time, work_estimator);

        // TODO Add other timelines
    }

    return node2swork;
}

swork_dict_t SGS::parallel(Chromosome* chromosome,
                           worker_pool_t worker_pool,  // we need a full copy here, it is changing in runtime
                           vector<vector<Worker>> &worker_pool_indices,
                           vector<GraphNode*> &index2node,
                           vector<Contractor*> &index2contractor,
                           vector<int> &index2zone,  // TODO
                           unordered_map<string, int> &worker_name2index,
                           unordered_map<string, int> &contractor2index,
                           LandscapeConfiguration &landscape,
                           Time assigned_parent_time,
                           Timeline &timeline,
                           WorkTimeEstimator &work_estimator) {
    // TODO
}
