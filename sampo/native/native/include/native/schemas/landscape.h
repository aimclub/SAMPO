#pragma once
//SAMPO-main/sampo/schemas/landscape - 

#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "resources.h"
#include "interval.h"

using namespace std;

class ResourceSupply : public Resource {
public:
	//string id;
	//string name;
	//int count;

	ResourceSupply(string id, string name, int count) : Resource(id, name, count){};
};
class Road : public ResourceSupply {
public:
	//string  id;
	//string name;
	//IntervalGaussian throughput;

	Road(string id, string name, IntervalGaussian throughput) : ResourceSupply(id, name, int(throughput.mean())) {};
};
class ResourceHolder : public ResourceSupply {
public:
	IntervalGaussian _productivity;
	vector<Material> _materials;

	ResourceHolder(string id, string name, IntervalGaussian productivity, vector<Material> materials)
		: ResourceSupply(std::move(id),
                         std::move(name),
                         int(productivity.mean())),
                         _productivity(std::move(productivity)),
                         _materials(std::move(materials)) {};

	ResourceHolder copy() {
		return ResourceHolder(id, name(), _productivity, _materials);
	}
	vector<pair<int, string>> get_available_resources() {
		vector<pair<int, string>> available_resources;

		for (const Material &materials : _materials) {
			available_resources.emplace_back(materials.count, materials.name);
		}

		return available_resources;
	}
};
class LandscapeConfiguration {
private:
	vector<Road> roads;
	vector<ResourceHolder> holders;

public:
	LandscapeConfiguration(vector<Road> roads = {}, vector<ResourceHolder> holders = {})
		: roads(std::move(roads)), holders(std::move(holders)) {};

	vector<ResourceSupply> get_all_resources() {
		vector<ResourceSupply> all_resourses;

		//for (auto _road : roads) {
		for (int i = 0; i < roads.size(); i++) {
			//all_resourses.push_back(&contractorid);
			//Road roads;

			ResourceSupply res_supl(roads[i].id, roads[i].AgentId.name, roads[i]._count);

			all_resourses.push_back(res_supl);
			//Road road;
			
			//all_resourses.push_back(&contractorid);
		}
		for (int i = 0; i < holders.size(); i++) {
			ResourceSupply res_supl(holders[i].id, holders[i].AgentId.name, holders[i]._count);
			all_resourses.push_back(res_supl);
		}

		return all_resourses;
	}
};

class MaterialDelivery {
private:
	//string work_id;
	string id;
	//vector<auto> delivery;
	unordered_map<string, vector<pair<Time, int>>> delivery;
public:
	explicit MaterialDelivery(std::string work_id) : id(std::move(work_id)) {
		//vector<map<string, string>> delivery;
	}

	void add_delivery(const string &name, Time time, int count) {
		auto material_delivery = delivery.find(name);

		if (material_delivery == delivery.end()) {
			vector<pair<Time, int>> new_material_delivery;
			delivery.insert({ name, new_material_delivery });
			material_delivery = delivery.find(name);
		}
		material_delivery->second.push_back(std::make_pair(time, count));
	}

	void add_deliveries(const string &name, vector<pair<Time, int>> deliveries) {
		auto material_delivery = delivery.find(name);

		if (material_delivery == delivery.end()) {
			delivery.insert({ name, deliveries });
		} else {
            material_delivery->second.insert(material_delivery->second.end(), deliveries.begin(), deliveries.end());
        }
	}
};
