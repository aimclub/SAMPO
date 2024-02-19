#include "native/scheduler/timeline/just_in_time.h"

class JustInTimeTimeline : Timeline {
private:
    unordered_map<string, unordered_map<string, vector<pair<Time, int>>*>> *timeline;
public:
    explicit JustInTimeTimeline(worker_pool_t &worker_pool, LandscapeConfiguration &landscape) {
        for (int contractor = 0; contractor < workers.size(); contractor++) {
            timeline[contractor] = unordered_map;
            for (int worker = 0; worker < workers[0].size(); worker++) {
                vector<pair<Time, int>> resource_state;
                auto it = this->timeline.find({ contractor, worker });
                timeline[contractor].resize(workers[0].size());
                timeline[contractor][worker].emplace_back(0, workers[contractor][worker]);
            }
        }
    }

    tuple<Time, Time, exec_times_t> find_min_start_time_with_additional(GraphNode *node,
                                                                        vector<Worker>& worker_team,
                                                                        swork_dict_t &node2swork,
                                                                        ScheduleSpec &spec,
                                                                        Time &assigned_start_time,
                                                                        Time &assigned_parent_time,
                                                                        WorkTimeEstimator &work_estimator) {

    }

    Time schedule(GraphNode *node,
                  vector<Worker>& worker_team,
                  swork_dict_t &node2swork,
                  ScheduleSpec &spec,
                  Contractor *contractor,
                  Time &assigned_start_time,
                  Time &assigned_time,
                  Time &assigned_parent_time,
                  WorkTimeEstimator &work_estimator) {

    }

    bool can_schedule_at_the_moment(GraphNode *node,
                                    vector<Worker>& worker_team,
                                    swork_dict_t &node2swork,
                                    ScheduleSpec &spec,
                                    Time &start_time,
                                    Time &exec_time) {

    }

    void update_timeline(GraphNode *node,
                         vector<Worker>& worker_team,
                         ScheduleSpec &spec,
                         Time &finish_time,
                         Time &exec_time) {

    }
};
