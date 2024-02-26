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

    swork_dict_t serial(Chromosome* chromosome,
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
                        WorkTimeEstimator &work_estimator);

    swork_dict_t parallel(Chromosome* chromosome,
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
                          WorkTimeEstimator &work_estimator);
}

#endif //SAMPO_SGS_H
