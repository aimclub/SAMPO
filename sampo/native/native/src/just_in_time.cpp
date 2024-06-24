#include <unordered_map>
#include <vector>
#include <string>

#include "native/scheduler/timeline/just_in_time.h"
#include "native/schemas/evaluator_types.h"
#include "native/schemas/workgraph.h"
#include "native/schemas/landscape.h"
#include "native/schemas/spec.h"
#include "native/schemas/time_estimator.h"

using namespace std;

JustInTimeTimeline::JustInTimeTimeline(const worker_pool_t &worker_pool, const LandscapeConfiguration &landscape) {
    for (const auto& it : worker_pool) {
        for (const auto& it_contractor : it.second) {
            vector<pair<Time, int>> resource_state;
            resource_state.emplace_back(0, it_contractor.second.count);
            this->timeline[it_contractor.first].insert(std::make_pair(it.first, resource_state));
        }
    }
//    for (const auto& it_contractor : timeline) {
//        cout << it_contractor.first << ":" << endl;
//        for (const auto &it : it_contractor.second) {
//            cout << it.first << " " << it.second[0].second << " ";
//        }
//        cout << endl;
//    }
    // TODO Add other timelines
}

JustInTimeTimeline::JustInTimeTimeline(const JustInTimeTimeline &other) : timeline(other.timeline) {}  // copy timeline

tuple<Time, Time, exec_times_t>
JustInTimeTimeline::find_min_start_time_with_additional(const GraphNode *node,
                                                        const vector<Worker>& worker_team,
                                                        const swork_dict_t &node2swork,
                                                        const WorkSpec *spec,
                                                        Time assigned_start_time,
                                                        Time assigned_parent_time,
                                                        const WorkTimeEstimator &work_estimator) const {
    if (node2swork.empty()) {
        // first work
        return { assigned_parent_time, assigned_parent_time, exec_times_t() };
    }

    Time max_parent_time = max(node->min_start_time(node2swork), assigned_parent_time);
    if (worker_team.empty()) {
        return { max_parent_time, max_parent_time, exec_times_t() };
    }

    Time max_agent_time(0);

    const auto& contractor_state = this->timeline.at(worker_team[0].contractor_id);

    if (spec != nullptr && spec->is_independent) {
        // grab from the end
        for (const auto& worker : worker_team) {
            const vector<pair<Time, int>>& offer_stack = contractor_state.at(worker.name);
            max_agent_time = max(max_agent_time, offer_stack[0].first);
        }
    } else {
        // grab from whole sequence
        // for each resource type
        for (const auto& worker : worker_team) {
            int needed_count = worker.count;

            const vector<pair<Time, int>> &offer_stack = contractor_state.at(worker.name);

            size_t ind = offer_stack.size() - 1;
            while (needed_count > 0) {
                auto[offer_time, offer_count] = offer_stack[ind];
                max_agent_time = max(max_agent_time, offer_time);

                if (needed_count < offer_count) {
                    offer_count = needed_count;
                }
                needed_count -= offer_count;
                ind--;
            }
        }
    }

    // TODO Remove multiple search time instances, replace with one
    Time c_st = max(max_agent_time, max_parent_time);
    Time new_finish_time = c_st;
    for (const auto* dep_node : node->getInseparableChainWithSelf()) {
        Time dep_parent_time = dep_node->min_start_time(node2swork);

        Time dep_st = max(new_finish_time, dep_parent_time);
        Time working_time = work_estimator.estimateTime(*dep_node->getWorkUnit(), worker_team);
        new_finish_time = dep_st + working_time;
    }

    return { c_st, new_finish_time, exec_times_t() };
}

bool JustInTimeTimeline::can_schedule_at_the_moment(const GraphNode *node,
                                                    const vector<Worker>& worker_team,
                                                    const swork_dict_t &node2swork,
                                                    const WorkSpec *spec,
                                                    Time start_time,
                                                    Time exec_time) const {
    if (spec != nullptr && spec->is_independent) {
        if (!worker_team.empty()) {
            const unordered_map<string, vector<pair<Time, int>>> &contractor_timeline
                = this->timeline.at(worker_team[0].contractor_id);
            for (const auto &worker: worker_team) {
                const auto &worker_timeline = contractor_timeline.at(worker.name);
                Time last_cpkt_time = worker_timeline[0].first;
                if (last_cpkt_time > start_time) {
                    return false;
                }
            }
        }
        return true;
    }
    // checking edges
    for (const auto* dep_node : node->getInseparableChainWithSelf()) {
//        for (auto* p : dep_node->parents()) {
//            if (p != dep_node->inseparable_parent()) {
//                auto swork_it = node2swork.find(p->id());
//                if (swork_it == node2swork.end() || swork_it->second.finish_time() > start_time) {
//                    cout << node->getWorkUnit()->name << " not passing on time "
//                         << start_time.val() << " because of edges" << endl;
//                    return false;
//                }
//            }
//        }
        Time dep_node_time = dep_node->min_start_time(node2swork);
        if (dep_node_time > start_time) {
//            cout << dep_node->getWorkUnit()->name << " not passing on time "
//                 << start_time.val() << " because of edges" << endl;
            return false;
        }
    }

    if (worker_team.empty()) {
        // empty worker team, passing
        return true;
    }

    const unordered_map<string, vector<pair<Time, int>>>& contractor_timeline = this->timeline.at(worker_team[0].contractor_id);

    // checking workers
    Time max_agent_time(0);

    for (const auto& worker : worker_team) {
        int needed_count = worker.count;
        const auto &offer_stack = contractor_timeline.at(worker.name);

        size_t ind = offer_stack.size() - 1;
        while (needed_count > 0) {
            auto[offer_time, offer_count] = offer_stack[ind];
            max_agent_time = max(max_agent_time, offer_time);

            if (needed_count < offer_count) {
                offer_count = needed_count;
            }
            needed_count -= offer_count;
            ind--;
        }
    }

//    if (max_agent_time > start_time) {
//        cout << node->getWorkUnit()->name << " with agent time " << max_agent_time.val()
//             << " not passing on time " << start_time.val() << " because of workers" << endl;
//    } else {
//        cout << node->getWorkUnit()->name << " passed" << endl;
//    }

    return max_agent_time <= start_time;
}

void JustInTimeTimeline::update_timeline(const GraphNode *node,
                                         const vector<Worker>& worker_team,
                                         const WorkSpec *spec,
                                         Time finish_time,
                                         Time exec_time) {
    if (worker_team.empty()) {
        // empty worker team, passing
        return;
    }

    auto& contractor_timeline = this->timeline.at(worker_team[0].contractor_id);

    if (spec != nullptr && spec->is_independent) {
        // squash all the timeline to the last point
        for (const auto& worker : worker_team) {
            auto &worker_timeline = contractor_timeline[worker.name];
            int count_workers = 0;
            for (auto& entry : worker_timeline) {
                count_workers += entry.second;
            }
            worker_timeline.clear();
            worker_timeline.emplace_back(finish_time, count_workers);
        }
    } else {
        // For each worker type consume the nearest available needed worker amount
        // and re-add it to the time when current work should be finished.
        // Addition performed as step in bubble-sort algorithm.
        for (const auto& worker : worker_team) {
            int needed_count = worker.count;
            auto &worker_timeline = contractor_timeline.at(worker.name);
//            cout << worker.name << " size: " << worker_timeline.size() << endl;

            // consume needed workers
            while (needed_count > 0) {
                if (worker_timeline.empty()) {
                    cout << "ERROR!!! JustInTimeTimeline#update_timeline" << endl;
                }
                pair<Time, int> next = worker_timeline.back();
                worker_timeline.pop_back();
                if (next.second > needed_count || worker_timeline.empty()) {
                    worker_timeline.emplace_back(next.first, next.second - needed_count);
                    break;
                }
                needed_count -= next.second;
            }

            // add to the right place
            worker_timeline.emplace_back(finish_time, worker.count);
            // FIXME long long here because using size_t causes unsigned overflow
            long long ind = worker_timeline.size() - 1;
            while (ind > 0 && worker_timeline[ind].first > worker_timeline[ind - 1].first) {
                std::swap(worker_timeline[ind], worker_timeline[ind - 1]);
                ind--;
            }
        }
    }
}

Time JustInTimeTimeline::schedule(const GraphNode *node,
                                  const vector<Worker>& worker_team,
                                  swork_dict_t &node2swork,
                                  const WorkSpec *spec,
                                  const Contractor *contractor,
                                  Time assigned_start_time,
                                  Time assigned_time,
                                  Time assigned_parent_time,
                                  const WorkTimeEstimator &work_estimator) {
    auto inseparable_chain = node->getInseparableChainWithSelf();
    Time start_time = assigned_start_time;
    if (start_time.is_unassigned()) {
//        cout << "Start time unassigned, searching" << endl;
        start_time = this->find_min_start_time(node, worker_team, node2swork, spec, assigned_parent_time, work_estimator);
    }

    exec_times_t exec_times;
    if (!assigned_time.is_unassigned()) {
        for (const auto* n : inseparable_chain) {
            exec_times[n->id()] = { Time(0), assigned_time / int(inseparable_chain.size()) };
        }
    }

    return this->schedule_with_inseparables(node, worker_team, node2swork, spec, contractor, start_time,
                                            inseparable_chain, exec_times, work_estimator);
}
