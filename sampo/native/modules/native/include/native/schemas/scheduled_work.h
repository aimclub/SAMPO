#pragma once
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include "workgraph.h"
#include "resources.h"
#include "contractor.h"
#include "landscape.h"

using namespace std;

class ScheduledWork {
public:
	WorkUnit* work_unit;
	pair<Time, Time> start_end_time;
	vector<Worker> workers;
	Contractor *contractor;
	//string contractor;
	vector<Equipment> equipments;
	vector<MaterialDelivery> materials;
	ConstructionObject object;
	int _cost;

    ScheduledWork() : ScheduledWork(nullptr, { Time(0), Time(0) }, vector<Worker>(), nullptr,
            vector<Equipment>(),vector<MaterialDelivery>(), ConstructionObject()) {}

	ScheduledWork(WorkUnit* work_unit, pair<Time, Time> start_end_time, vector<Worker> workers, Contractor *contractor,
		vector<Equipment> equipments, vector<MaterialDelivery> materials, ConstructionObject object)
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

		_cost = 0;

        // TODO
//		for (auto& worker : workers) {
//			_cost += worker.get_cost() * this->duration().val();
//		}
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
	Time start_time() const {
		return start_end_time.first;
	}
	void start_time(Time val) {
		start_end_time = make_pair(std::move(val), start_end_time.second);
	}
	Time finish_time() const {
		return start_end_time.second;
	}
};