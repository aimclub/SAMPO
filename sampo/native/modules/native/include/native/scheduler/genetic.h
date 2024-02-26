#ifndef NATIVE_GENETIC_H
#define NATIVE_GENETIC_H

#define PY_SSIZE_T_CLEAN

#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "Python.h"
#include "native/scheduler/chromosome_evaluator.h"
#include "native/schemas/evaluator_types.h"
#include "native/scheduler/timeline/timeline.h"
#include "native/scheduler/timeline/just_in_time.h"
#include "native/scheduler/sgs.h"
#include "native/pycodec.h"
#include "native/utils.h"

#include "numpy/arrayobject.h"

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

    //    template<typename T>
    //    void swap(T* o1, T* o2) {
    //        T tmp = *o1;
    //        *o1 = *o2;
    //        *o2 = tmp;
    //    }
    //
    //    template<typename T>
    //    void shuffle(T* data, int first, int last, unsigned seed) {
    //        auto rnd = default_random_engine(seed);
    //        for (int i = last - 1; i > first; i--) {
    //            std::uniform_int_distribution<int> d(first, i);
    //            swap(data + i, data + d(rnd));
    //        }
    //    }

    void mutateOrder(Chromosome *chromosome) {
        random_device rd;
        int *order = chromosome->getOrder()[0];
        std::shuffle(order, order + chromosome->getOrder().size(), rd);
    }

    void crossOrder(Chromosome *a, Chromosome *b) {
        random_device rd;
        uniform_int_distribution<int> rnd(0, a->getOrder().size());

        if (a->numWorks() != b->numWorks()) {
            cout << "Wrong chromosomes!" << a->numWorks() << " "
                 << b->numWorks() << endl;
            return;
        }

        int crossPoint = rnd(rd);

        set<int> headA;
        set<int> headB;
        vector<int> toA;
        vector<int> toB;
        // store heads
        for (int i = 0; i < crossPoint; i++) {
            headA.insert(*a->getOrder()[i]);
            headB.insert(*b->getOrder()[i]);
        }
        int indA = crossPoint;
        int indB = crossPoint;
        for (int i = 0; i < a->getOrder().size(); i++) {
            int curA = *a->getOrder()[i];
            int curB = *b->getOrder()[i];
            // if cur work not in head, insert work
            if (headA.find(curB) == headA.end()) {
                if (indA >= a->numWorks()) {
                    cout << "Overflow A" << endl;
                }
                else {
                    *a->getOrder()[indA++] = curB;
                }
            }
            if (headB.find(curA) == headB.end()) {
                if (indB >= a->numWorks()) {
                    cout << "Overflow B" << endl;
                }
                else {
                    *b->getOrder()[indB++] = curA;
                }
            }
        }
    }

    void mutateResources(Chromosome *chromosome) {
        random_device rd;
        auto resourcesToChange =
            sample_ind(chromosome->getResources().width() - 1, 1, rd);
        auto worksToChange =
            sample_ind(chromosome->getOrder().width(), mutateResourcesProb, rd);

        // mutate resources
        for (int work : worksToChange) {
            auto resMin = this->resourceMinBorder[work];
            int *resMax = chromosome->getWorkResourceBorder(work);
            for (int res : resourcesToChange) {
                chromosome->getResources()[work][res] =
                    randInt(resMin[res], resMax[res]);
            }
        }
        // TODO mutate contractor
    }

    void crossResources(Chromosome *a, Chromosome *b) {
        random_device rd;
        auto resourcesToMate = sample_ind(a->getResources().width(), 1, rd);

        for (int work = 0; work < a->getOrder().size(); work++) {
            for (int res : resourcesToMate) {
                int tmp                      = a->getResources()[work][res];
                a->getResources()[work][res] = b->getResources()[work][res];
                b->getResources()[work][res] = tmp;
            }
        }
    }

    void mutateContractors(Chromosome *chromosome) {
        random_device rd;
        auto resourcesToChange =
            sample_ind(chromosome->numResources(), mutateResourcesProb, rd);
        auto contractorsToChange =
            sample_ind(chromosome->numContractors(), mutateContractorsProb, rd);

        // mutate resources
        for (int work : contractorsToChange) {
            auto resMin = this->resourceMinBorder[work];
            for (int res : resourcesToChange) {
                chromosome->getContractors()[work][res] -= randInt(
                    resMin[res] + 1,
                    max(resMin[res] + 1,
                        (int)(chromosome->getContractors()[work][res] / 1.2))
                );
            }
        }
    }

    void crossContractors(Chromosome *a, Chromosome *b) {
        random_device rd;
        auto resourcesToMate = sample_ind(a->numResources(), 1, rd);
        auto contractorsToMate =
            sample_ind(a->numContractors(), mutateContractorsProb, rd);

        // TODO Decide should we mate whole contractor borders
        for (int contractor = 0; contractor < a->numContractors();
             contractor++) {
            for (int res : resourcesToMate) {
                int tmp = a->getResources()[contractor][res];
                a->getContractors()[contractor][res] =
                    b->getResources()[contractor][res];
                b->getContractors()[contractor][res] = tmp;
            }
        }
    }

    void applyOperators(
        vector<Chromosome *> &chromosomes,
        vector<Chromosome *> &target,
        float mutate,
        float cross,
        void (Genetic::*mutator)(Chromosome *),
        void (Genetic::*crossover)(Chromosome *, Chromosome *)
    ) {
        random_device rd;
        auto toMutate = sample(chromosomes, mutate, rd);
#pragma omp parallel for firstprivate(mutator) shared(toMutate) default(none)  \
    num_threads(this->numThreads)
        for (int i = 0; i < toMutate.size(); i++) {
            (this->*mutator)(toMutate[i]);
        }

        // multiply with 2 because of 2 chromosomes needed for crossover
        // operation
        auto toCross = sample(chromosomes, 2 * cross, rd);
#pragma omp parallel for firstprivate(crossover) shared(toCross) default(none) \
    num_threads(this->numThreads)
        for (int i = 0; i < toCross.size() - 1; i += 2) {
            (this->*crossover)(toCross[i], toCross[i + 1]);
        }
        // add to global list
        target.insert(target.end(), toMutate.begin(), toMutate.end());
        target.insert(target.end(), toCross.begin(), toCross.end());
    }

    void applyOrder(
        vector<Chromosome *> &chromosomes, vector<Chromosome *> &target
    ) {
        applyOperators(
            chromosomes,
            target,
            mutateOrderProb,
            crossOrderProb,
            &Genetic::mutateOrder,
            &Genetic::crossOrder
        );
    }

    void applyResources(
        vector<Chromosome *> &chromosomes, vector<Chromosome *> &target
    ) {
        applyOperators(
            chromosomes,
            target,
            mutateResourcesProb,
            crossResourcesProb,
            &Genetic::mutateResources,
            &Genetic::crossResources
        );
    }

    void applyContractors(
        vector<Chromosome *> &chromosomes, vector<Chromosome *> &target
    ) {
        applyOperators(
            chromosomes,
            target,
            mutateContractorsProb,
            crossContractorsProb,
            &Genetic::mutateContractors,
            &Genetic::crossContractors
        );
    }

    typedef void (Genetic::*
                      p_op_apply)(vector<Chromosome *> &, vector<Chromosome *> &);
    const static int APPLY_FUNCTIONS_COUNT = 3;
    p_op_apply applyFunctions[APPLY_FUNCTIONS_COUNT];

    inline vector<Chromosome *> applyAll(vector<Chromosome *> &chromosomes) {
        vector<Chromosome *> nextGen[APPLY_FUNCTIONS_COUNT];

        // TODO Research about HOW adding this directive slows down the runtime
        //        #pragma omp parallel for shared(chromosomes, nextGen) default
        //        (none) num_threads(this->numThreads)
        for (int i = 0; i < APPLY_FUNCTIONS_COUNT; i++) {
            (this->*applyFunctions[i])(chromosomes, nextGen[i]);
        }

        // aggregate results
        vector<Chromosome *> results;
        for (auto &functionResult : nextGen) {
            results.insert(
                results.end(), functionResult.begin(), functionResult.end()
            );
        }

        return results;
    }

    inline vector<Chromosome *> selection(vector<Chromosome *> &population) {
        auto top = argsort(population);
        vector<Chromosome *> result;
        result.resize(min(population.size(), (size_t)sizeSelection));
        for (int i = 0; i < result.size(); i++) {
            result[i] = new Chromosome(population[top[i]]);
        }
        return result;
    }

    static Chromosome *updateHOF(vector<Chromosome *> &population, Chromosome *oldLeader) {
        Chromosome *leader = oldLeader;
        for (auto &chromosome : population) {
            if (chromosome->fitness < leader->fitness) {
                leader = chromosome;
            }
        }
        return leader;
    }

    static void deleteChromosomes(vector<Chromosome *> &chromosomes) {
        for (auto *chromosome : chromosomes) {
            delete chromosome;
        }
    }

public:
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
          numThreads(evaluator.numThreads) {
        // TODO
        applyFunctions[0] = &Genetic::applyOrder;
        applyFunctions[1] = &Genetic::applyResources;
        applyFunctions[2] = &Genetic::applyContractors;
    }

    // TODO Add multi-criteria optimization (object hierarchy of fitness
    // functions)
    Chromosome *run(vector<Chromosome *> &initialPopulation) {
        // TODO Ensure this is copy
        auto population = initialPopulation;

        int maxPlateau = 100;
        int curPlateau = 0;

        evaluator.evaluate(population);

        int prevFitness            = INT_MAX;
        Chromosome *bestChromosome = nullptr;

        for (auto i : population) {
            if (bestChromosome == nullptr
                || i->fitness < bestChromosome->fitness) {
                bestChromosome = i;
            }
        }

        int g = 0;
        // TODO Propagate from Python
        const int MAX_GENERATIONS = 50;

        while (g < MAX_GENERATIONS && curPlateau < maxPlateau) {
            //        printf("--- Generation %i | fitness = %i\n", g,
            //        bestChromosome->fitness);
            // update plateau info
            if (bestChromosome->fitness == prevFitness) {
                curPlateau++;
            }
            else {
                curPlateau = 0;
            }

            auto offspring = selection(population);

            //        cout << "Selected " << offspring.size() << " individuals"
            //        << endl;

            auto nextGeneration = applyAll(offspring);

            evaluator.evaluate(nextGeneration);

            auto newBest   = updateHOF(nextGeneration, bestChromosome);
            bestChromosome = new Chromosome(newBest);

            // renew population
            deleteChromosomes(population);
            population.clear();
            population.insert(
                population.end(), offspring.begin(), offspring.end()
            );
            population.insert(
                population.end(), nextGeneration.begin(), nextGeneration.end()
            );

            g++;
        }
        deleteChromosomes(population);

        //        cout << "Result validation: " <<
        //        evaluator.isValid(bestChromosome) << endl;

        return bestChromosome;
    }
};

#endif    // NATIVE_GENETIC_H
