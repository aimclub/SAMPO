#include "native/schemas/chromosome.h"

Chromosome::Chromosome(int worksCount, int resourcesCount, int contractorsCount, ScheduleSpec spec)
        : worksCount(worksCount),
          resourcesCount(resourcesCount),
          contractorsCount(contractorsCount),
          spec(std::move(spec)) {

    size_t ORDER_SHIFT     = 0;
    size_t RESOURCES_SHIFT = ORDER_SHIFT + worksCount;
    size_t CONTRACTORS_SHIFT =
            RESOURCES_SHIFT + worksCount * (resourcesCount + 1);
    this->DATA_SIZE =
            (CONTRACTORS_SHIFT + contractorsCount * resourcesCount)
            * sizeof(int);
    //        cout << ORDER_SHIFT << " " << RESOURCES_SHIFT << " " <<
    //        CONTRACTORS_SHIFT << endl; cout << DATA_SIZE << endl;
    this->data = (int *) malloc(DATA_SIZE);
    if (data == nullptr) {
        cout << "Not enough memory" << endl;
        return;
    }
    this->order     = Array2D<int>(worksCount, 1, this->data);
    this->resources = Array2D<int>(
            worksCount * (resourcesCount + 1),
            resourcesCount + 1,
            this->data + RESOURCES_SHIFT
    );
    this->contractors = Array2D<int>(
            contractorsCount * resourcesCount,
            resourcesCount,
            this->data + CONTRACTORS_SHIFT
    );
}

Chromosome::Chromosome(Chromosome *other) : Chromosome(other->worksCount, other->resourcesCount, other->contractorsCount) {
    // copy all packed data
    memcpy(this->data, other->data, DATA_SIZE);
    this->fitness = other->fitness;
}

Chromosome* Chromosome::from_schedule(unordered_map<string, int> &work_id2index,
                                      unordered_map<string, int> &worker_name2index,
                                      unordered_map<string, int> &contractor2index,
                                      Array2D<int> &contractor_borders,
                                      unordered_map<string, ScheduledWork> &schedule,
                                      ScheduleSpec &spec,
                                      LandscapeConfiguration &landscape,
                                      vector<string> order) {
    auto* chromosome = new Chromosome(work_id2index.size(),
                                      worker_name2index.size(),
                                      contractor2index.size());

    if (order.size() == 0) {
        // if order not specified, create
        for (auto& entry : work_id2index) {
            order.emplace_back(entry.first);
        }
    }

    for (size_t i = 0; i < order.size(); i++) {
        auto& node = order[i];
        int index = work_id2index[node];
        *chromosome->getOrder()[i] = index;
        for (auto& resource : schedule[node].workers) {
            int res_index = worker_name2index[resource.name];
            chromosome->getResources()[index][res_index] = resource.count;
            chromosome->getContractor(index) = contractor2index[resource.contractor_id];
        }
    }

    // TODO Implement this using memcpy
    auto& contractor_chromosome = chromosome->getContractors();
    for (int contractor = 0; contractor < contractor_borders.height(); contractor++) {
        for (int resource = 0; resource < contractor_borders.width(); resource++) {
            contractor_chromosome[contractor][resource] = contractor_borders[contractor][resource];
        }
    }

    return chromosome;
}
