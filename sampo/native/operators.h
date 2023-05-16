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

    vector<Chromosome*> applyAll(vector<Chromosome>& chromosomes,
                                 float mutateOrder, float mutateResources, float mutateContractors,
                                 float crossOrder, float crossResources, float crossContractors) {

    }

    vector<Chromosome*> applyOrder(vector<Chromosome*>& chromosomes, float mutate, float cross) {

    }

    vector<Chromosome*> applyResources(vector<Chromosome*>& chromosomes, float mutate, float cross) {

    }

    vector<Chromosome*> applyContractors(vector<Chromosome*>& chromosomes, float mutate, float cross) {

    }
}

#endif //NATIVE_OPERATORS_H
