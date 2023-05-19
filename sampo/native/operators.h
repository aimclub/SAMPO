#ifndef NATIVE_OPERATORS_H
#define NATIVE_OPERATORS_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"
#include "pycodec.h"
#include "evaluator_types.h"

#include <vector>

using namespace std;

namespace Operators {

    void applyOrder(vector<Chromosome*>& chromosomes, vector<Chromosome*>& target, float mutate, float cross) {
        for (int i = 0; i < chromosomes.size(); i++) {
            // without i = j & j = i duplications
            for (int j = i; j < chromosomes.size(); j++) {
                
            }
        }
    }

    void applyResources(vector<Chromosome*>& chromosomes, vector<Chromosome*>& target, float mutate, float cross) {

    }

    void applyContractors(vector<Chromosome*>& chromosomes, vector<Chromosome*>& target, float mutate, float cross) {

    }

    vector<Chromosome*> applyAll(vector<Chromosome*>& chromosomes,
                                 float mutateOrder, float mutateResources, float mutateContractors,
                                 float crossOrder, float crossResources, float crossContractors) {
        vector<Chromosome*> offspring;
        applyOrder(chromosomes, offspring, mutateOrder, crossOrder);
        applyResources(chromosomes, offspring, mutateResources, crossResources);
        applyContractors(chromosomes, offspring, mutateContractors, crossContractors);
        return offspring;
    }
}

#endif //NATIVE_OPERATORS_H
