#pragma once
//SAMPO-main/sampo/schemas/landscape - 

#include <string>
#include <vector>
#include "interval.h"
#include "resources.h"

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

	Road(string id, string name, IntervalGaussian throughput) : ResourceSupply(id, name, int(throughput.mean)) {};
};
class ResourceHolder : public ResourceSupply {
public:
	string _id;
	string _name;
	IntervalGaussian _productivity;
	vector<Material> _materials;

	ResourceHolder(string id, string name, IntervalGaussian productivity, vector<Material> materials)
		: ResourceSupply(id, name, int(productivity.mean)), _id(id), _name(name), _productivity(productivity), _materials(materials) {};

	ResourceHolder copy() {
		return ResourceHolder(_id, _name, _productivity, _materials);
	}
	vector<pair<int, string>> get_available_resources() {
		vector<pair<int, string>> AvailableResources;

		for (Material materials : _materials) {
			AvailableResources.push_back(make_pair(materials.count, materials.name));
		}

		return AvailableResources;
	}
};
class LandscapeConfiguration {
public:
	vector<Road> roads;
	vector<ResourceHolder> holders;

public:
	LandscapeConfiguration(vector<Road> roads = {}, vector<ResourceHolder> holders = {})
		: roads(roads), holders(holders) {};

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
public:
	//string work_id;
	string _id;
	//vector<auto> delivery;
	map<string, vector<pair<Time, int>>> delivery;
public:
	MaterialDelivery(const std::string& work_id) : _id(work_id) {
		//vector<map<string, string>> delivery;
	}

	void add_delivery(string name, Time time, int count) {
		auto material_delivery = delivery.find(name);

		if (material_delivery == delivery.end()) {
			vector<pair<Time, int>> new_material_delivery;
			delivery.insert({ name, new_material_delivery });
			material_delivery = delivery.find(name);
		}
		material_delivery->second.push_back(std::make_pair(time, count));
	}

	void add_deliveries(string name, vector<pair<Time, int>> deliveries) {
		auto material_delivery = delivery.find(name);

		if (material_delivery == delivery.end()) {
			delivery.insert({ name, deliveries });
			material_delivery = delivery.find(name);
		}
		else
			material_delivery->second.insert(material_delivery->second.end(), deliveries.begin(), deliveries.end());
	}
};

//class LandscapeConfiguration {
//public:
//	LandscapeConfiguration(vector<Road> roads = {}, std::vector<ResourceHolder> holders = {})
//		: _roads(roads), _holders(holders) {}
//	vector<ResourceSupply> get_all_resources() {
//		vector<ResourceSupply> all_resources;
//		all_resources.insert(all_resources.end(), _roads.begin(), _roads.end());
//		all_resources.insert(all_resources.end(), _holders.begin(), _holders.end());
//		return all_resources;
//	}
//private:
//	std::vector<Road> _roads;
//	std::vector<ResourceHolder> _holders;
//};

//class LandscapeConfiguration {
//private:
//	std::vector<Road> roads;
//	std::vector<ResourceHolder> holders;
//
//public:
//	LandscapeConfiguration(std::vector<Road> roads = {}, std::vector<ResourceHolder> holders = {})
//		: roads(roads), holders(holders) {}
//
//	std::vector<ResourceSupply*> get_all_resources() {
//		std::vector<ResourceSupply*> all_resources;
//		for (const auto& road : roads) {
//			all_resources.push_back(&road);
//		}
//		for (const auto& holder : holders) {
//			all_resources.push_back(&holder);
//		}
//		return all_resources;
//	}
//};