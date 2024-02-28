#ifndef SAMPO_SGS_H
#define SAMPO_SGS_H

#include <unordered_map>
#include <string>

#include "native/schemas/scheduled_work.h"
#include "native/schemas/workgraph.h"
#include "native/schemas/chromosome.h"
#include "native/schemas/time_estimator.h"
#include "native/scheduler/timeline/timeline.h"
#include "native/schemas/evaluator_types.h"
#include "native/schemas/contractor.h"

namespace SGS {

    swork_dict_t serial(Chromosome* chromosome,
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
                        const WorkTimeEstimator &work_estimator);

    swork_dict_t parallel(Chromosome* chromosome,
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
                          const WorkTimeEstimator &work_estimator);
}

#endif //SAMPO_SGS_H
