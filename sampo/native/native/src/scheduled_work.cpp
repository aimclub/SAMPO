#include "native/schemas/scheduled_work.h"

ScheduledWork::ScheduledWork(const WorkUnit* work_unit,
                             pair<Time, Time> start_end_time,
                             vector<Worker> workers,
                             const Contractor *contractor,
                             vector<Equipment> equipments,
                             vector<MaterialDelivery> materials,
                             ConstructionObject object)
    : work_unit(work_unit), start_end_time(std::move(start_end_time)),
      workers(std::move(workers)), contractor(contractor), equipments(std::move(equipments)),
      materials(std::move(materials)), object(std::move(object)) {

    //if contractor is not None:
    //	if isinstance(contractor, str) :
    //		self.contractor = contractor
    //	else :
    //		self.contractor = contractor.name if contractor.name else contractor.id
    //else:
    //	self.contractor = ""

    //self.cost = 0
    //for worker in self.workers :
    //	self.cost += worker.get_cost() * self.duration.value

    //if (!(contractor == NULL)) {
    /*if (!(contractor == NULL)) {
        contractor->get_id
    }*/

    cost = 0;

    // TODO
//		for (auto& worker : workers) {
//			_cost += worker.get_cost() * this->duration().val();
//		}
}

ScheduledWork::ScheduledWork()
    : ScheduledWork(nullptr, { Time(0), Time(0) }, vector<Worker>(), nullptr,
      vector<Equipment>(),vector<MaterialDelivery>(), ConstructionObject()) {}

const Time& ScheduledWork::start_time() const {
    return start_end_time.first;
}

void ScheduledWork::start_time(const Time& val) {
    start_end_time = make_pair(val, start_end_time.second);
}

const Time& ScheduledWork::finish_time() const {
    return start_end_time.second;
}

Time ScheduledWork::duration() const {
    return finish_time() - start_time();
}


