#ifndef NATIVE_GENETIC_H
#define NATIVE_GENETIC_H

#define PY_SSIZE_T_CLEAN

#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "native/scheduler/chromosome_evaluator.h"
#include "native/schemas/evaluator_types.h"
#include "native/scheduler/timeline/timeline.h"
#include "native/scheduler/timeline/just_in_time.h"
#include "native/scheduler/sgs.h"
#include "native/pycodec.h"

using namespace std;

class Genetic {
private:
    vector<vector<int>> resourceMinBorder;
    unsigned seed;

    float mutateOrderProb, mutateResourcesProb, mutateContractorsProb;
    float crossOrderProb, crossResourcesProb, crossContractorsProb;

    int sizeSelection;
    int numThreads;

    ChromosomeEvaluator evaluator;

    static void deleteChromosomes(vector<Chromosome *> &chromosomes) {
        for (auto *chromosome : chromosomes) {
            delete chromosome;
        }
    }

public:
    void mutateOrder(Chromosome *chromosome);

    void crossOrder(Chromosome *a, Chromosome *b);

    void mutateResources(Chromosome *chromosome);

    void crossResources(Chromosome *a, Chromosome *b);

    void mutateContractors(Chromosome *chromosome);

    void crossContractors(Chromosome *a, Chromosome *b);

    void applyOperators(vector<Chromosome *> &chromosomes,
                        vector<Chromosome *> &target,
                        float mutate,
                        float cross,
                        void (Genetic::*mutator)(Chromosome *),
                        void (Genetic::*crossover)(Chromosome *, Chromosome *));

    void applyOrder(vector<Chromosome *> &chromosomes, vector<Chromosome *> &target);

    void applyResources(vector<Chromosome *> &chromosomes, vector<Chromosome *> &target);

    void applyContractors(vector<Chromosome *> &chromosomes, vector<Chromosome *> &target);

    typedef void (Genetic::* p_op_apply)(vector<Chromosome *> &, vector<Chromosome *> &);
    const static int APPLY_FUNCTIONS_COUNT = 3;
    p_op_apply applyFunctions[APPLY_FUNCTIONS_COUNT];

    vector<Chromosome *> applyAll(vector<Chromosome *> &chromosomes);

    vector<Chromosome *> selection(vector<Chromosome *> &population);

    static Chromosome *updateHOF(vector<Chromosome *> &population, Chromosome *oldLeader);

    explicit Genetic(
        vector<vector<int>> &resourcesMinBorder,
        float mutateOrder,
        float mutateResources,
        float mutateContractors,
        float crossOrder,
        float crossResources,
        float crossContractors,
        int sizeSelection,
        ChromosomeEvaluator &evaluator,
        unsigned seed = 0
    )
        : resourceMinBorder(resourcesMinBorder),
          seed(seed),
          evaluator(evaluator),
          sizeSelection(sizeSelection),
          mutateOrderProb(mutateOrder),
          mutateResourcesProb(mutateResources),
          mutateContractorsProb(mutateContractors),
          crossOrderProb(crossOrder),
          crossResourcesProb(crossResources),
          crossContractorsProb(crossContractors),
          numThreads(evaluator.num_threads) {
        // TODO
        applyFunctions[0] = &Genetic::applyOrder;
        applyFunctions[1] = &Genetic::applyResources;
        applyFunctions[2] = &Genetic::applyContractors;
    }

    Chromosome* run(vector<Chromosome *> &initialPopulation);
};

#endif    // NATIVE_GENETIC_H
