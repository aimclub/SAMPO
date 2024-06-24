#include <utility>

#include "native/scheduler/chromosome_evaluator.h"
#include "native/python_deserializer.h"

ChromosomeEvaluator::ChromosomeEvaluator(const WorkGraph *wg,
                                         vector<Contractor*> contractors,
                                         ScheduleSpec spec,
                                         const WorkTimeEstimator *work_estimator)
        : wg(wg),
          contractors(contractors),
          work_estimator(work_estimator),
          spec(std::move(spec)),
          sgs(ScheduleGenerationScheme::Parallel) {
    unordered_map<string, int> node_id2index;

    // construct the outer numeration
    // start with inseparable heads
    for (auto* node_1 : wg->nodes) {
        if (!node_1->is_inseparable_son()) {
            node_id2index[node_1->id()] = (int) this->index2node.size();
            this->index2node.push_back(node_1);
        }
    }

    size_t heads_count = index2node.size();
    // continue with others
    // this should match the numeration on the Python-size backend,
    // otherwise everything will break
    for (auto* node_2 : wg->nodes) {
        if (node_2->is_inseparable_son()) {
            node_id2index[node_2->id()] = (int) index2node.size();
            this->index2node.push_back(node_2);
        }
    }

    // prepare inseparable heads predecessor mapping
    vector<int> inseparable_heads;
    inseparable_heads.resize(wg->nodes.size());
    for (int i = 0; i < wg->nodes.size(); i++) {
        for (const auto* inseparable_node : wg->nodes[i]->getInseparableChainWithSelf()) {
            inseparable_heads[i] = node_id2index[inseparable_node->id()];
        }
    }

    this->headParents.resize(heads_count);
    for (int i = 0; i < wg->nodes.size(); i++) {
        unordered_set<int> parent_inds;
        for (auto* parent : wg->nodes[inseparable_heads[i]]->parents()) {
            parent_inds.emplace(inseparable_heads[node_id2index[parent->id()]]);
        }
        vector<int> parents_vec;
        parents_vec.insert(parents_vec.begin(), parent_inds.begin(), parent_inds.end());
        this->headParents.emplace_back(parents_vec);
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

    for (auto* contractor : contractors) {
        for (const auto& worker : contractor->workers) {
            this->worker_pool[worker.name].emplace(contractor->id, worker);
            worker_names.emplace(worker.name);
        }
    }

    int ind = 0;
    for (const auto& worker_name : worker_names) {
        this->worker_name2index[worker_name] = ind;
        ind++;
    }

    this->minReqs.resize(index2node.size());
    this->maxReqs.resize(index2node.size());
    for (int node = 0; node < index2node.size(); node++) {
        this->minReqs[node].resize(worker_names.size());
        this->maxReqs[node].resize(worker_names.size());
        auto* work_unit = this->index2node[node]->getWorkUnit();
        for (auto& worker_req : work_unit->worker_reqs) {
            int worker_ind = this->worker_name2index[worker_req.kind];
            this->minReqs[node][worker_ind] = worker_req.min_count;
            this->maxReqs[node][worker_ind] = worker_req.max_count;
        }
    }

    for (int index = 0; index < contractors.size(); index++) {
        contractor2index[contractors[index]->id] = index;
    }

    this->num_threads = omp_get_num_procs();
//        this->num_threads = this->usePythonWorkEstimator ? 1 : omp_get_num_procs();
//        printf("Genetic running threads: %i\n", this->numThreads);

//        if (info->useExternalWorkEstimator) {
//            loader.DLOpenLib();
//            auto library        = loader.DLGetInstance();
//            this->timeEstimator = library->create(info->timeEstimatorPath);
//        }
}

ChromosomeEvaluator::~ChromosomeEvaluator() {
    delete work_estimator;
    delete wg;
    for (auto* contractor : contractors) {
        delete contractor;
    }
}

void ChromosomeEvaluator::set_sgs(ScheduleGenerationScheme sgs) {
    this->sgs = sgs;
}

bool ChromosomeEvaluator::is_valid(Chromosome *chromosome) {
    bool visited[chromosome->numWorks()];

    // check edges
    for (int i = 0; i < chromosome->numWorks(); i++) {
        int node      = *chromosome->getOrder()[i];
        visited[node] = true;
        for (int parent : headParents[node]) {
            if (!visited[parent]) {
//                cout << "Mismatch: " << node << " " << parent << endl;
//                cout << "Invalid order: ";
//                for (int k = 0; k < chromosome->numWorks(); k++) {
//                    cout << *chromosome->getOrder()[k] << " ";
//                }
//                cout << endl;
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
//                cout << endl << "Invalid resources: " << minReqs[node][res] << " <= " << count
//                     << " <= " << chromosome->getContractors()[contractor][res] << endl;
//                for (const auto &entry : worker_name2index) {
//                    cout << entry.first << " " << entry.second << endl;
//                }
                return false;
            }
        }
    }

    return true;
}

void ChromosomeEvaluator::evaluate(vector<Chromosome *> &chromosomes) {
    auto fitness = TimeFitness();
    // #pragma omp parallel for shared(chromosomes, fitness) default(none) num_threads(this->numThreads)
    for (auto* chromosome : chromosomes) {
        if (is_valid(chromosome)) {
            // TODO Add sgs_type parametrization
            JustInTimeTimeline timeline(worker_pool, landscape);
            Time assigned_parent_time;
            swork_dict_t schedule = SGS::serial(chromosome,
                                                  worker_pool,
                                                  spec,
                                                  worker_pool_indices,
                                                  index2node,
                                                  contractors,
                                                  index2zone,
                                                  worker_name2index,
                                                  contractor2index,
                                                  landscape,
                                                  assigned_parent_time,
                                                  timeline,
                                                  *work_estimator);
//            vector<ScheduledWork*> sworks;
//            for (auto&[work_id, swork] : schedule) {
//                sworks.push_back(&swork);
//            }
//            std::sort(sworks.begin(), sworks.end(), [](ScheduledWork* swork1, ScheduledWork* swork2) {
//                return swork1->work_unit->name > swork2->work_unit->name;
//            });
//            for (const auto* node : wg->nodes) {
//                ScheduledWork& swork = schedule[node->id()];
//                cout << swork.work_unit->name << " : " << swork.start_time().val() << " " << swork.finish_time().val() << endl;
//            }
            chromosome->fitness = fitness.evaluate(schedule);
        } else {
            chromosome->fitness = INT_MAX;
//            cout << "Chromosome invalid" << endl;
        }
    }
}

vector<py::object> ChromosomeEvaluator::get_schedules(vector<Chromosome *> &chromosomes) {
    vector<py::object> schedules;
    auto fitness = TimeFitness();
    // #pragma omp parallel for shared(chromosomes, fitness) default(none) num_threads(this->numThreads)
    for (auto* chromosome : chromosomes) {
        if (is_valid(chromosome)) {
            // TODO Add sgs_type parametrization
            JustInTimeTimeline timeline(worker_pool, landscape);
            Time assigned_parent_time;
            swork_dict_t schedule = SGS::parallel(chromosome,
                                                  worker_pool,
                                                  spec,
                                                  worker_pool_indices,
                                                  index2node,
                                                  contractors,
                                                  index2zone,
                                                  worker_name2index,
                                                  contractor2index,
                                                  landscape,
                                                  assigned_parent_time,
                                                  timeline,
                                                  *work_estimator);

            schedules.push_back(PythonDeserializer::encodeSchedule(schedule));
        } else {
            chromosome->fitness = INT_MAX;
        }
    }

    return schedules;
}
