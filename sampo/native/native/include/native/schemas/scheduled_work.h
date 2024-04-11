#pragma once

#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include "native/schemas/works.h"
#include "native/schemas/resources.h"
#include "native/schemas/contractor.h"
#include "native/schemas/landscape.h"

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

    ScheduledWork();

	ScheduledWork(WorkUnit* work_unit, pair<Time, Time> start_end_time, vector<Worker> workers, Contractor *contractor,
		          vector<Equipment> equipments, vector<MaterialDelivery> materials, ConstructionObject object);

	Time duration() const;

	const Time& start_time() const;

	void start_time(Time val);

	const Time& finish_time() const;
};