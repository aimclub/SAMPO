//sheduler/shemas/resources -
#ifndef RESOURCES_H
#define RESOURCES_H

#include <utility>
#include <vector>
#include <random>
#include "interval.h"
#include "identifiable.h"

using namespace std;

class WorkerProductivityMode {
public:
	std::string Static = "static";
	std::string Stohastic = "stohastic";
};

typedef struct {
	std::string name;
	std::string contractor_id;
} agentId;

class Resource {				//íàäî ñäåëàòü êîíñòðóêòîð
//private:
public:
	std::string id;
	agentId AgentId;
	int _count;
	

	/*typedef struct {
		std::string name;
		std::string contractor_id;
	} AgentId;*/
public:
	Resource(string id, string name, int count, string contractor_id = "") :
					id(id), /*name(name),*/ _count(count) {
		AgentId.name = name;
		AgentId.contractor_id = contractor_id;
	}
	/*void set_id(std::string value) {
		id = value;
	}
	void set_name(std::string value) {
		name = value;
	}
	void set_count(int value) {
		count = value;
	}
	void set_contractor_id(std::string value) {
		contractor_id = value;
	}
	std::string get_id() {
		return id;
	}
	std::string get_name() {
		return name;
	}
	int get_count() {
		return count;
	}
	std::string get_contractor_id() {
		return contractor_id;
	}*/
	void count(int value) {
		_count = value;
	}
	int count() {
		return _count;
	}
};

class Worker : public Identifiable {
    string id;
    string name;
    int count;
	int cost;
    string contractor_id;
    IntervalGaussian productivity;

public:
    Worker(string id, string name, int count, int cost, string contractorId, const IntervalGaussian& productivity)
        : id(std::move(id)), name(std::move(name)), count(count), cost(cost),
        contractor_id(std::move(contractorId)), productivity(productivity) {}

	std::string get_id() {
		return id;
	}
	std::string get_name() {
		return name;
	}
	int get_count() {
		return count;
	}
	int get_cost() {
		return cost;
	}
	std::string get_contractor_id() {
		return contractor_id;
	}
	IntervalGaussian get_productivity() {
		return productivity;
	}
};

class ConstructionObject : public Resource {

};

class Equipment : public Resource {

};

class Material : public Resource {
//private:
public:
	std::string id;
	std::string name;
	int count;
	float cost_one_unit;
public:
	Material(string id, string name, int count, float cost_one_unit = 1) : Resource (id, name, count) {
	//Material () : Resource(string id, string name, int count)
	}
	std::string get_id() {
		return id;
	}
	std::string get_name() {
		return name;
	}
	int get_count() {
		return count;
	}
	float get_cost_one_unit() {
		return cost_one_unit;
	}
	void set_id(std::string id) {
		this->id = id;
	}
	void set_name(std::string name) {
		this->name = name;
	}
	void set_count(int count) {
		this->count = count;
	}
	void set_cost_one_unit(float cost_one_unit) {
		this->cost_one_unit = cost_one_unit;
	}
	/*Material* copy_M(Material* prev)
	{
		Material* newW;

		newW->set_id(prev->get_id());
		newW->set_name(prev->get_name());
		newW->set_count(prev->get_count());

		return newW;
	}*/
	void copy() {
		this->id = get_id();
		this->name = get_name();
		this->count = get_count();
		//this->productivity = get_productivity();
	}
	int with_count(int value) {
		set_count(value);
		return get_count();
	}
};

#endif