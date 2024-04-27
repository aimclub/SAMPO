#include "native/schemas/landscape.h"

LandscapeConfiguration::LandscapeConfiguration(vector<Road> roads, vector<ResourceHolder> holders)
    : roads(std::move(roads)), holders(std::move(holders)) {}

vector<ResourceSupply> LandscapeConfiguration::get_all_resources() {
    vector<ResourceSupply> all_resources;

    for (size_t i = 0; i < roads.size(); i++) {
        all_resources.emplace_back(roads[i].id, roads[i].AgentId.name, roads[i]._count);
    }
    for (size_t i = 0; i < holders.size(); i++) {
        all_resources.emplace_back(holders[i].id, holders[i].AgentId.name, holders[i]._count);
    }

    return all_resources;
}

MaterialDelivery::MaterialDelivery(std::string work_id) : id(std::move(work_id)) {}

void MaterialDelivery::add_delivery(const string &name, Time time, int count) {
    auto material_delivery = delivery.find(name);

    if (material_delivery == delivery.end()) {
        vector<pair<Time, int>> new_material_delivery;
        delivery.insert({ name, new_material_delivery });
        material_delivery = delivery.find(name);
    }
    material_delivery->second.push_back(std::make_pair(time, count));
}

void MaterialDelivery::add_deliveries(const string &name, vector<pair<Time, int>> deliveries) {
    auto material_delivery = delivery.find(name);

    if (material_delivery == delivery.end()) {
        delivery.insert({ name, deliveries });
    } else {
        material_delivery->second.insert(material_delivery->second.end(), deliveries.begin(), deliveries.end());
    }
}

ResourceHolder::ResourceHolder(string id, string name, IntervalGaussian productivity, vector<Material> materials)
    : ResourceSupply(std::move(id),
                     std::move(name),
                     int(productivity.mean())),
                     _productivity(std::move(productivity)),
                     _materials(std::move(materials)) {}

ResourceHolder ResourceHolder::copy() {
    return ResourceHolder(id, name(), _productivity, _materials);
}
vector<pair<int, string>> ResourceHolder::get_available_resources() {
    vector<pair<int, string>> available_resources;

    for (const Material &materials : _materials) {
        available_resources.emplace_back(materials.count, materials.name);
    }

    return available_resources;
}
