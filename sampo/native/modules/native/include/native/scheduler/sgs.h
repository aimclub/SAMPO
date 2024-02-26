#ifndef SAMPO_SGS_H
#define SAMPO_SGS_H

#include <unordered_map>
#include <string>

#include "native/schemas/scheduled_work.h"
#include "native/schemas/workgraph.h"
#include "native/schemas/time_estimator.h"
#include "native/scheduler/timeline/timeline.h"
#include "native/schemas/evaluator_types.h"
#include "native/schemas/contractor.h"

namespace SGS {

    swork_dict_t serial_sgs(Chromosome* chromosome,
                            worker_pool_t worker_pool,  // we need a full copy here, it is changing in runtime
                            vector<vector<Worker>> &worker_pool_indices,
                            vector<GraphNode*> &index2node,
                            vector<Contractor*> &index2contractor,
                            vector<int> &index2zone,  // TODO
                            Array2D<int> &contractor_borders,
                            unordered_map<string, int> &worker_name2index,
                            unordered_map<string, int> &contractor2index,
                            LandscapeConfiguration &landscape,
                            Time &assigned_parent_time,
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
        }

        return node2swork;
    }
}

#endif //SAMPO_SGS_H
