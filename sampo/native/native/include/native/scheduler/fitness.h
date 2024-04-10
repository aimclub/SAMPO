#ifndef SAMPO_FITNESS_H
#define SAMPO_FITNESS_H

#include "native/schemas/evaluator_types.h"

class FitnessFunction {
public:
    virtual ~FitnessFunction() = default;

    virtual float evaluate(const swork_dict_t &schedule) const = 0;
};

class TimeFitness : public FitnessFunction {
public:
    float evaluate(const swork_dict_t &schedule) const override;
};

// TODO Make more fitness functions

#endif //SAMPO_FITNESS_H
