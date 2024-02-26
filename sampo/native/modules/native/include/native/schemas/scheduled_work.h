#pragma once
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include "workgraph.h"
#include "resources.h"
#include "contractor.h"
#include "landscape.h"
//#include "seria"

using namespace std;

class ScheduledWork {
public:
	vector<string> ignored_fields = { "equipments", "materials", "object" };
	WorkUnit work_unit;
	pair<Time, Time> start_end_time;
	vector<Worker> workers;
	Contractor *contractor;
	//string contractor;
	vector<Equipment> equipments;
	vector<MaterialDelivery> materials;
	ConstructionObject object;
	int _cost;

    ScheduledWork() : ScheduledWork(WorkUnit(), { Time(0), Time(0) }, vector<Worker>(), nullptr,
            vector<Equipment>(),vector<MaterialDelivery>(), ConstructionObject()) {}

	ScheduledWork(WorkUnit work_unit, pair<Time, Time> start_end_time, const vector<Worker>& workers, Contractor *contractor,
		vector<Equipment> equipments, vector<MaterialDelivery> materials, ConstructionObject object)
        : work_unit(std::move(work_unit)), start_end_time(start_end_time),
		  workers(workers), contractor(contractor), equipments(std::move(equipments)),
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

		_cost = 0;

		for (auto& worker : workers) {
			_cost += worker.get_cost() * this->duration().get_val();
		}
	}

	Time duration() {
		Time start = get<0>(start_end_time), end = get<1>(start_end_time);
		return end - start;
	}
//	void __str__ (void){
//		cout << "ScheduledWork[work_unit = " << this->work_unit << "start_end_time = " << this->start_end_time <<
//			"workers = " << this->workers << "contractor = " << this->contractor << "]" << endl;
//	}
//	void __repr__(void) {
//		this->__str__();
//	}
	Time start_time() {
		return start_end_time.first;
	}
	void start_time(Time val) {
		start_end_time = make_pair(val, start_end_time.second);
	}
	Time finish_time(void) {
		return start_end_time.second;
	}
	Time min_child_start_time() {
        return work_unit.isServiceUnit ? finish_time() : finish_time() + 1;
    }
};