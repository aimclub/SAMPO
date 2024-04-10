#include "native/scheduler/fitness.h"


float TimeFitness::evaluate(const swork_dict_t &schedule) const {
    float result = 0;

    for (const auto& swork : schedule) {
        result = std::max(result, float(swork.second.finish_time().val()));
    }

    return result;
}
