#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>

#include "native/scheduler/timeline/just_in_time.h"
#include "native/schemas/evaluator_types.h"
#include "native/schemas/workgraph.h"
#include "native/schemas/landscape.h"
#include "native/schemas/spec.h"

using namespace std;

class JustInTimeTimeline : Timeline {
private:
    unordered_map<string, unordered_map<string, vector<pair<Time, int>>>> *timeline;

    Time schedule_with_inseparables(GraphNode *node,
                                    vector<Worker>& worker_team,
                                    swork_dict_t &node2swork,
                                    ScheduleSpec &spec,
                                    Contractor *contractor,
                                    Time &start_time,
                                    vector<GraphNode*> &inseparable_chain,
                                    exec_times_t &exec_times,
                                    WorkTimeEstimator &work_estimator) {
        Time c_ft = start_time;
        for (auto& dep_node : inseparable_chain) {
            Time max_parent_time = dep_node->min_start_time(node2swork);

            pair<Time, Time>& node_lag_exec_time;
            auto it = exec_times.find(dep_node->id());
            if (it == exec_times.end()) {
                node_lag_exec_time = { Time(0), work_estimator.estimateTime(node->getWorkUnit(), worker_team) };
            } else {
                node_lag_exec_time = it->second;
            }

            Time c_st = max(c_ft + node_lag_exec_time.first, max_parent_time);
            Time new_finish_time = c_st + node_lag_exec_time.second;

            node2swork.emplace(dep_node->id(), dep_node->getWorkUnit(),
                               { c_st, new_finish_time },
                               worker_team, contractor, deliveries);
            c_ft = new_finish_time;
        }

        this->update_timeline(node, worker_team, spec, c_ft, c_ft - start_time);
        return c_ft;
     }
public:
    explicit JustInTimeTimeline(worker_pool_t &worker_pool, LandscapeConfiguration &landscape) {
        this->timeline = new unordered_map<string, unordered_map<string, vector<pair<Time, int>>>>();

        for (auto& it : worker_pool) {
            timeline[it.first] = unordered_map<string, vector<pair<Time, int>>>();
            for (auto& it_contractor : it.second) {
                vector<pair<Time, int>> resource_state;
                resource_state.emplace_back(0, workers[contractor][worker]);
                timeline[it.first][worker] = resource_state;
            }
        }
        // TODO Add other timelines
    }

    ~JustInTimeTimeline() {
        delete this->timeline;
    }

    tuple<Time, Time, exec_times_t> find_min_start_time_with_additional(GraphNode *node,
                                                                        vector<Worker>& worker_team,
                                                                        swork_dict_t &node2swork,
                                                                        WorkSpec &spec,
                                                                        Time &assigned_start_time,
                                                                        Time &assigned_parent_time,
                                                                        WorkTimeEstimator &work_estimator) override {
        if (node2swork.size() == 0) {
            // first work
            return { assigned_parent_time, assigned_parent_time, exec_times_t() };
        }

        Time max_parent_time = max(node->min_start_time(node2swork), assigned_parent_time);
        if (worker_team.size() == 0) {
            return { max_parent_time, max_parent_time, exec_times_t() };
        }

        Time max_agent_time(0);

        auto& contractor_state = this->timeline[worker_team[0].get_contractor_id()];

        if (spec.is_independent) {
            // grab from the end
            for (auto& worker : worker_team) {
                vector<pair<Time, int>>& offer_stack = contractor_state[worker];
                max_agent_time = max(max_agent_time, offer_stack[0].first);
            }
        } else {
            // grab from whole sequence
            // for each resource type
            for (auto& worker : worker_team) {
                int needed_count = worker.get_count();
                vector<pair<Time, int>> &offer_stack = contractor_state[worker];

                size_t ind = offer_stack.size() - 1;
                while (needed_count > 0) {
                    auto&[offer_time, offer_count] = offer_stack[ind];
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
        for (auto& dep_node : node->getInseparableChainWithSelf()) {
            Time dep_parent_time = dep_node->min_start_time(node2swork);

            Time dep_st = max(new_finish_time, dep_parent_time);
            Time working_time = work_estimator.estimateTime(dep_node->getWorkUnit(), worker_team);
            new_finish_time = dep_st + working_time;
        }

        return { c_st, new_finish_time, exec_times_t() };
    }

    bool can_schedule_at_the_moment(GraphNode *node,
                                    vector<Worker>& worker_team,
                                    swork_dict_t &node2swork,
                                    WorkSpec &spec,
                                    Time &start_time,
                                    Time &exec_time) override {
        if (worker_team.size() == 0) {
            // empty worker team, passing
            return true;
        }

        unordered_map<string, vector<pair<Time, int>>>& contractor_timeline = this->timeline[worker_team[0].get_contractor_id()];

        if (spec.is_independent) {
            for (auto& worker : worker_team) {
                auto& worker_timeline = contractor_timeline[worker.get_name()];
                Time last_cpkt_time = worker_timeline[0].first;
                if (last_cpkt_time > start_time) {
                    return false;
                }
            }
        } else {
            // checking edges
            for (auto& dep_node : node->getInseparableChainWithSelf()) {
                for (auto& p : dep_node->parents()) {
                    if (p->id() != dep_node->id()) {
                        auto swork_it = node2swork.find(p->id());
                        if (swork_it == node2swork.end() || swork_it->second.finish_time() > start_time) {
                            return false;
                        }
                    }
                }
            }

            // checking workers
            Time max_agent_time(0);

            for (auto& worker : worker_team) {
                int needed_count = worker.get_count();
                auto &offer_stack = contractor_timeline[worker.get_name()];

                size_t ind = offer_stack.size() - 1;
                while (needed_count > 0) {
                    auto&[offer_time, offer_count] = offer_stack[ind];
                    max_agent_time = max(max_agent_time, offer_time);

                    if (needed_count < offer_count) {
                        offer_count = needed_count;
                    }
                    needed_count -= offer_count;
                    ind--;
                }
            }

            if (max_agent_time > start_time) {
                return false;
            }

            return true;
        }
    }

    void update_timeline(GraphNode *node,
                         vector<Worker>& worker_team,
                         WorkSpec &spec,
                         Time &finish_time,
                         Time &exec_time) override {
        if (worker_team.size() == 0) {
            // empty worker team, passing
            return true;
        }

        unordered_map<string, vector<pair<Time, int>>>& contractor_timeline = this->timeline[worker_team[0].get_contractor_id()];

        if (spec.is_independent) {
            // squash all the timeline to the last point
            for (auto& worker : worker_team) {
                auto &worker_timeline = contractor_timeline[worker.get_name()];
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
            for (auto& worker : worker_team) {
                int needed_count = worker.get_count();
                auto &worker_timeline = contractor_timeline[worker.get_name()];

                // consume needed workers
                while (needed_count > 0) {
                    pair<Time, int>* next = worker_timeline.end();
                    worker_timeline.pop_back();
                    if (next->second > needed_count && worker_timeline.size() == 0) {
                        worker_timeline.emplace_back(next->first, next->second - needed_count);
                        break;
                    }
                    needed_count -= next->second;
                }

                // add to the right place
                worker_timeline.emplace_back({ finish_time, worker.get_count() });
                size_t ind = len(worker_timeline) - 1;
                while (ind > 0 && worker_timeline[ind].first > worker_timeline[ind - 1].first) {
                    std::swap(worker_timeline[ind], worker_timeline[ind - 1]);
                    ind--;
                }
            }
        }
    }

    Time schedule(GraphNode *node,
                  vector<Worker>& worker_team,
                  swork_dict_t &node2swork,
                  ScheduleSpec &spec,
                  Contractor *contractor,
                  Time &assigned_start_time,
                  Time &assigned_time,
                  Time &assigned_parent_time,
                  WorkTimeEstimator &work_estimator) override {
        auto& inseparable_chain = node->getInseparableChainWithSelf();
        Time start_time = assigned_start_time;
        if (start_time.unassigned()) {
            start_time = this->find_min_start_time(node, worker_team, node2swork, spec, assigned_parent_time, work_estimator);
        }

        exec_times_t exec_times;
        if (!assigned_time.unassigned()) {
            for (auto& n : inseparable_chain) {
                exec_times[n.id()] = { Time(0), assigned_time / inseparable_chain.size() };
            }
        }

        return this->schedule_with_inseparables(node, worker_team, node2swork, spec, contractor, start,
                                                inseparable_chain, exec_times, work_estimator);
    }
};
