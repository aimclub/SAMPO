#ifndef NATIVE_CHROMOSOME_EVALUATOR_H
#define NATIVE_CHROMOSOME_EVALUATOR_H

// #pragma GCC optimize("Ofast")
// #pragma GCC optimize("no-stack-protector")
// #pragma GCC optimize("unroll-loops")
// #pragma GCC target("sse,sse2,sse3,ssse3,popcnt,abm,mmx,tune=native")
// #pragma GCC optimize("fast-math")

// #pragma optimize( "O2", on )

#include <iostream>
#include <set>
#include <unordered_map>
#include <vector>

#include <omp.h>

#include "DLLoader.h"
#include "native/schemas/evaluator_types.h"
#include "native/schemas/external.h"
#include "native/schemas/time_estimator.h"
#include "native/schemas/chromosome.h"
#include "native/pycodec.h"

#include "sgs.h"
#include "native/scheduler/timeline/just_in_time.h"
#include "native/scheduler/fitness.h"

class ChromosomeEvaluator {
private:
    EvaluateInfo *info_ptr;

    WorkGraph *wg;

    vector<vector<int>> headParents;     // vertices' parents without inseparables
    vector<vector<int>> minReqs;         // work -> worker -> WorkUnit.min_req
    vector<vector<int>> maxReqs;         // work -> worker -> WorkUnit.max_req
    vector<string> id2work;
    vector<string> id2res;

    worker_pool_t worker_pool;
    const LandscapeConfiguration &landscape;
    const WorkTimeEstimator &work_estimator;

    vector<vector<Worker*>> worker_pool_indices;
    vector<GraphNode*> index2node;
    const vector<Contractor*> &index2contractor;
    vector<int> index2zone;  // TODO
    unordered_map<string, int> worker_name2index;
    unordered_map<string, int> contractor2index;

    unordered_map<string, unordered_map<string, int>> minReqNames;
    unordered_map<string, unordered_map<string, int>> maxReqNames;

    py::object pythonWrapper;
    bool usePythonWorkEstimator;

    // TODO (?) Make interop with Python work estimators like in old NativeWrapper was
//    WorkTimeEstimator *timeEstimator;
//    dlloader::DLLoader<ITimeEstimatorLibrary> loader {
//        External::timeEstimatorLibPath
//    };

public:
    int numThreads;

    explicit ChromosomeEvaluator(EvaluateInfo *info)
        : info_ptr(info),
          wg(info->wg),
          landscape(info->landscape),
          work_estimator(*info->work_estimator),
          index2contractor(info->contractors),
          pythonWrapper(info->pythonWrapper) {

        for (size_t i = 0; i < minReqs.size(); i++) {
            const auto &work = id2work[i];
            // TODO I think we can remove this
            minReqNames[work] = unordered_map<string, int>();
            maxReqNames[work] = unordered_map<string, int>();
            for (size_t j = 0; j < minReqs[i].size(); j++) {
                if (minReqs[i][j] != 0) {
                    const auto &res = id2res[j];
                    minReqNames[work][res] = minReqs[i][j];
                    maxReqNames[work][res] = maxReqs[i][j];
                }
            }
        }

        unordered_map<string, int> node_id2index;

        // construct the outer numeration
        // start with inseparable heads
        for (auto* node : wg->nodes) {
            if (!node->is_inseparable_son()) {
                node_id2index[node->id()] = (int) index2node.size();
                index2node.push_back(node);
            }
        }

        size_t heads_count = index2node.size();
        // continue with others
        // this should match the numeration on the Python-size backend,
        // otherwise everything will break
        for (auto* node : wg->nodes) {
            if (node->is_inseparable_son()) {
                node_id2index[node->id()] = (int) index2node.size();
                index2node.push_back(node);
            }
        }

        // prepare inseparable heads predecessor mapping
        vector<int> inseparable_heads;
        inseparable_heads.resize(wg->nodes.size());
        for (int i = 0; i < wg->nodes.size(); i++) {
            for (auto* inseparable_node : wg->nodes[i]->getInseparableChainWithSelf()) {
                inseparable_heads[i] = node_id2index[inseparable_node->id()];
            }
        }

        headParents.resize(heads_count);
        for (int i = 0; i < wg->nodes.size(); i++) {
            unordered_set<int> parent_inds;
            for (auto* parent : wg->nodes[inseparable_heads[i]]->parents()) {
                parent_inds.emplace(inseparable_heads[node_id2index[parent->id()]]);
            }
            vector<int> parents_vec;
            parents_vec.insert(parents_vec.begin(), parent_inds.begin(), parent_inds.end());
            headParents.emplace_back(parents_vec);
        }

        // TODO
//        worker_pool_indices.resize(index2contractor.size());
//        for (size_t contractor_ind = 0; contractor_ind < index2contractor.size(); contractor_ind++) {
//            for (size_t worker_ind = 0; worker_ind < index2contractor[contractor_ind]->workers.size(); worker_ind++) {
//                worker_pool_indices[worker_ind].resize(index2contractor.size());
//                worker_pool_indices[worker_ind][contractor_ind] = index2contractor[contractor_ind]->workers[worker_ind];
//            }
//        }

        set<string> worker_names;

        for (auto* contractor : index2contractor) {
            for (const auto& worker : contractor->workers) {
                worker_pool[worker.name].emplace(contractor->id, worker);
                worker_names.emplace(worker.name);
            }
        }

        int ind = 0;
        for (const auto& worker_name : worker_names) {
            worker_name2index[worker_name] = ind;
            ind++;
        }

        for (int index = 0; index < index2contractor.size(); index++) {
            contractor2index[index2contractor[index]->id] = index;
        }

        this->usePythonWorkEstimator = info->usePythonWorkEstimator;
        this->numThreads = this->usePythonWorkEstimator ? 1 : omp_get_num_procs();
//        printf("Genetic running threads: %i\n", this->numThreads);

//        if (info->useExternalWorkEstimator) {
//            loader.DLOpenLib();
//            auto library        = loader.DLGetInstance();
//            this->timeEstimator = library->create(info->timeEstimatorPath);
//        }
//        else if (!usePythonWorkEstimator) {
//            this->timeEstimator = new DefaultWorkTimeEstimator(minReqNames, maxReqNames);
//        }
    }

    // TODO Research why deleting timeEstimator causes Head Corruption crash
    //    ~ChromosomeEvaluator() {
    //        delete timeEstimator;
    //    }
    //    ~ChromosomeEvaluator() {
    //        loader.DLCloseLib();
    //    }
    ~ChromosomeEvaluator() {
        delete info_ptr;
    }

    bool isValid(Chromosome *chromosome) {
        bool visited[chromosome->numWorks()];

        // check edges
        for (int i = 0; i < chromosome->numWorks(); i++) {
            int node      = *chromosome->getOrder()[i];
            visited[node] = true;
            for (int parent : headParents[node]) {
                if (!visited[parent]) {
//                    cout << "Mismatch: " << node << " " << parent << endl;
//                    cout << "Invalid order: ";
//                    for (int k = 0; k < chromosome->numWorks(); k++) {
//                        cout << *chromosome->getOrder()[k] << " ";
//                    }
//                    cout << endl;
                    return false;
                }
            }
        }

        // check resources
        for (int node = 0; node < chromosome->numWorks(); node++) {
            int contractor = chromosome->getContractor(node);
            for (int res = 0; res < chromosome->numResources(); res++) {
                int count = chromosome->getResources()[node][res];
                if (count < minReqs[node][res] || count > chromosome->getContractors()[contractor][res]) {
//                    cout << "Invalid resources: " << minReqs[node][res] << " <= " << count << " <= " << chromosome->getContractors()[contractor][res] << endl;
                    return false;
                }
            }
        }

        return true;
    }

    void evaluate(vector<Chromosome *> &chromosomes) {
        auto fitness = TimeFitness();
 // #pragma omp parallel for shared(chromosomes, fitness) default(none) num_threads(this->numThreads)
        for (auto* chromosome : chromosomes) {
            if (isValid(chromosome)) {
                // TODO Add sgs_type parametrization
                JustInTimeTimeline timeline(worker_pool, landscape);
                Time assigned_parent_time;
                swork_dict_t schedule = SGS::serial(chromosome,
                                                    worker_pool,
                                                    worker_pool_indices,
                                                    index2node,
                                                    index2contractor,
                                                    index2zone,
                                                    worker_name2index,
                                                    contractor2index,
                                                    landscape,
                                                    assigned_parent_time,
                                                    timeline,
                                                    work_estimator);
                chromosome->fitness = fitness.evaluate(schedule);
//                cout << "Fitness written out: " << chromosome->fitness << endl;
            } else {
                chromosome->fitness = INT_MAX;
//                cout << "Chromosome invalid" << endl;
            }
        }
    }
};


#endif    // NATIVE_CHROMOSOME_EVALUATOR_H
