#include <list>

#include "native/scheduler/sgs.h"

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
        vector<Worker> worker_team;
        for (auto& wreq : node->getWorkUnit()->worker_reqs) {
            auto &v = worker_pool[wreq.kind];
            int count = chromosome->getResources()[work_index][worker_name2index.at(wreq.kind)];
            worker_team.emplace_back(v[contractor->id].copy().with_count(count));
//            worker_team.emplace_back(v[contractor->id]);
        }

//        if (!worker_team.empty()) {
//            cout << "Node " << node->getWorkUnit()->name << endl;
//            cout << "Worker team mask: ";
//            for (int i = 0; i < chromosome->numResources(); i++) {
//                cout << chromosome->getResources()[work_index][i] << " ";
//            }
//            cout << endl;
//
//            cout << "Worker team: ";
//            for (auto& worker : worker_team) {
//                cout << worker.count << " ";
//            }
//            cout << endl;
//        }

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

    list<GraphNode*> enumerated_works_remaining;
    for (int i = 0; i < chromosome->numWorks(); i++) {
        int order_index = *chromosome->getOrder()[i];
        enumerated_works_remaining.push_back(index2node[order_index]);
    }



    return node2swork;
}
