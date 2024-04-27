#pragma once

#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "native/schemas/resources.h"
#include "native/schemas/interval.h"
#include "native/schemas/dtime.h"

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

	ResourceHolder(string id, string name, IntervalGaussian productivity, vector<Material> materials);

	ResourceHolder copy();

	vector<pair<int, string>> get_available_resources();
};
class LandscapeConfiguration {
private:
	vector<Road> roads;
	vector<ResourceHolder> holders;

public:
	LandscapeConfiguration(vector<Road> roads = {}, vector<ResourceHolder> holders = {});

	vector<ResourceSupply> get_all_resources();
};

class MaterialDelivery {
private:
	string id;
	unordered_map<string, vector<pair<Time, int>>> delivery;
public:
	explicit MaterialDelivery(std::string work_id);

	void add_delivery(const string &name, Time time, int count);

	void add_deliveries(const string &name, vector<pair<Time, int>> deliveries);
};
