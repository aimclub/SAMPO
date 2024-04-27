#include "native/schemas/time_estimator.h"

Time DefaultWorkTimeEstimator::estimateTime(const WorkUnit &work, const vector<Worker> &workers) const {
    // the _abstract_estimate from WorkUnit
    int time = 0;

    // TODO Rework with OOP

    // build worker_req index
    auto worker_reqs = build_index<WorkerReq, string>(work.worker_reqs,
                                                      [](const WorkerReq &req) { return req.kind; });

    for (const auto& worker : workers) {
        const auto& worker_req = worker_reqs[worker.name];
        if (worker_req.min_count == 0)
            continue;
        int actual_count = worker.count;
        if (actual_count < worker_req.min_count) {
            cout << "Not conforms to min_req: " << worker.name << " count: "
                 << actual_count << ", required: " << worker_req.min_count << endl;
            return TIME_INF;
        }
        int max_req = worker_req.max_count;

        float productivity = get_productivity(actual_count);
        productivity *= communication_coefficient(actual_count, max_req);

        //        if (productivity < 0.000001) {
        //            return TIME_INF;
        //        }
        //        productivity = 0.1;
        int new_time = ceil((float) worker_req.volume.val() / productivity);
        if (new_time > time) {
            time = new_time;
        }
    }

    return Time(time);
}

