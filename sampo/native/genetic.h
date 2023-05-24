#ifndef NATIVE_GENETIC_H
#define NATIVE_GENETIC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"
#include "pycodec.h"
#include "evaluator_types.h"
#include "chromosome_evaluator.h"

#include <bits/stdc++.h>
#include <vector>

using namespace std;

class Genetic {
private:
    vector<vector<int>> resourceMinBorder;
    unsigned seed;

    float mutateOrderProb, mutateResourcesProb, mutateContractorsProb;
    float crossOrderProb, crossResourcesProb, crossContractorsProb;

    int sizeSelection;

    ChromosomeEvaluator evaluator;

    inline void mutateOrder(Chromosome* chromosome) {
        int* order = chromosome->getOrder()[0];
        shuffle(order, order + chromosome->getOrder().size(),
                default_random_engine());
    }

    inline void crossOrder(Chromosome* a, Chromosome* b) {
        random_device rd;
        uniform_int_distribution<int> rnd(0, a->getOrder().size());

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
                *a->getOrder()[indA] = curB;
            }
            if (headB.find(curA) == headB.end()) {
                *b->getOrder()[indB] = curA;
            }
        }
    }

    inline vector<int> sample_ind(int size, float prob, random_device& rd) {
        uniform_int_distribution<int> rnd(0, size);

        vector<int> result;
        result.resize((int)(prob * (float) rnd(rd)));

        for (int & i : result) {
            i = rnd(rd);
        }
        return result;
    }

    template<typename T>
    inline vector<T*> sample(vector<T*>& src, float prob, random_device& rd) {
        uniform_int_distribution<int> rnd(0, src.size());

        vector<T*> result;
        result.resize((int)(prob * (float) rnd(rd)));

        for (int i = 0; i < result.size(); i++) {
            result[i] = new T(src[rnd(rd)]);
        }
        return result;
    }

    static inline int randInt(int from, int to) {
        // TODO implement using C++ 11 and without re-creating distribution objects each call
        return (int)(rand() * (to - from));
    }

    inline void mutateResources(Chromosome* chromosome) {
        random_device rd;
        auto resourcesToChange = sample_ind(chromosome->getResources().width() - 1, 1, rd);
        auto worksToChange = sample_ind(chromosome->getOrder().width(), mutateResourcesProb, rd);

        // mutate resources
        for (int work : worksToChange) {
            auto resMin = this->resourceMinBorder[work];
            int* resMax = chromosome->getWorkResourceBorder(work);
            for (int res : resourcesToChange) {
                chromosome->getResources()[work][res] = randInt(resMin[res], resMax[res]);
            }
        }
        // TODO mutate contractor
    }

    inline void crossResources(Chromosome* a, Chromosome* b) {
        random_device rd;
        auto resourcesToMate = sample_ind(a->getResources().width(), 1, rd);

        for (int work = 0; work < a->getOrder().size(); work++) {
            for (int res : resourcesToMate) {
                int tmp = a->getResources()[work][res];
                a->getResources()[work][res] = b->getResources()[work][res];
                b->getResources()[work][res] = tmp;
            }
        }
    }

    inline void mutateContractors(Chromosome* chromosome) {
        random_device rd;
        auto resourcesToChange = sample_ind(chromosome->numResources(), mutateResourcesProb, rd);
        auto contractorsToChange = sample_ind(chromosome->numContractors(), mutateContractorsProb, rd);

        // mutate resources
        for (int work : contractorsToChange) {
            auto resMin = this->resourceMinBorder[work];
            for (int res : resourcesToChange) {
                chromosome->getContractors()[work][res] -= randInt(resMin[res] + 1, max(resMin[res] + 1,
                                                                   (int)(chromosome->getContractors()[work][res] / 1.2)));
            }
        }
    }

    inline void crossContractors(Chromosome* a, Chromosome* b) {
        random_device rd;
        auto resourcesToMate = sample_ind(a->numResources(), 1, rd);
        auto contractorsToMate = sample_ind(a->numContractors(), mutateContractorsProb, rd);

        // TODO Decide should we mate whole contractor borders
        for (int contractor = 0; contractor < a->getOrder().size(); contractor++) {
            for (int res : resourcesToMate) {
                int tmp = a->getResources()[contractor][res];
                a->getContractors()[contractor][res] = b->getResources()[contractor][res];
                b->getContractors()[contractor][res] = tmp;
            }
        }
    }

    void applyOperators(vector<Chromosome*>& chromosomes, vector<Chromosome*>& target, float mutate, float cross,
                        void (Genetic::*mutator)(Chromosome*), void (Genetic::*crossover)(Chromosome*, Chromosome*)) {
        random_device rd;
        auto toMutate = sample(chromosomes, mutate, rd);
        // TODO pragma omp parallel
        for (int i = 0; i < toMutate.size(); i++) {
            (this->*mutator)(toMutate[i]);
        }

        // multiply with 2 because of 2 chromosomes needed for crossover operation
        auto toCross = sample(chromosomes, 2 * cross, rd);
        // TODO pragma omp parallel
        for (int i = 0; i < toCross.size(); i += 2) {
            (this->*crossover)(toCross[i], toCross[i + 1]);
        }
        // add to global list
        target.insert(target.end(), toMutate.begin(), toMutate.end());
        target.insert(target.end(), toCross.begin(), toCross.end());
    }

    inline void applyOrder(vector<Chromosome*>& chromosomes, vector<Chromosome*>& target) {
        applyOperators(chromosomes, target, mutateOrderProb, crossOrderProb, &Genetic::mutateOrder, &Genetic::crossOrder);
    }

    inline void applyResources(vector<Chromosome*>& chromosomes, vector<Chromosome*>& target) {
        applyOperators(chromosomes, target, mutateResourcesProb, crossResourcesProb, &Genetic::mutateResources, &Genetic::crossResources);
    }

    inline void applyContractors(vector<Chromosome*>& chromosomes, vector<Chromosome*>& target) {
        applyOperators(chromosomes, target, mutateContractorsProb, crossContractorsProb, &Genetic::mutateContractors, &Genetic::crossContractors);
    }

    vector<Chromosome*> applyAll(vector<Chromosome*>& chromosomes) {
        vector<Chromosome*> offspring;
        applyOrder(chromosomes, offspring);
        applyResources(chromosomes, offspring);
        applyContractors(chromosomes, offspring);
        return offspring;
    }

    /**
     * Argsort(currently support ascending sort)
     * @tparam T array element type
     * @param array input array
     * @return indices w.r.t sorted array
     */
    template<typename T>
    std::vector<size_t> argsort(const std::vector<T> &array) {
        std::vector<size_t> indices(array.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&array](int left, int right) -> bool {
                      // sort indices according to corresponding array element
                      return array[left] < array[right];
                  });

        return indices;
    }

    inline vector<Chromosome*> selection(vector<Chromosome*>& population, vector<int>& fitness) {
        auto top = argsort(fitness);
        vector<Chromosome*> result;
        result.resize(sizeSelection);
        for (int i = 0; i < sizeSelection; i++) {
            result[i] = population[top[i]];
        }
        return result;
    }

    // return the position of new leader, and -1 if the old one should be used
    inline static int updateHOF(vector<Chromosome*>& population, vector<int>& fitness, int oldFit) {
        int leaderPos = 0;
        int leaderFit = INT_MAX;
        for (int i = 0; i < population.size(); i++) {
            if (fitness[i] < leaderFit) {
                leaderPos = i;
                leaderFit = fitness[i];
            }
        }
        return oldFit <= leaderFit ? -1 : leaderPos;
    }

public:
    explicit Genetic(vector<vector<int>>& resourcesMinBorder,
                     float mutateOrder, float mutateResources, float mutateContractors,
                     float crossOrder, float crossResources, float crossContractors,
                     int sizeSelection,
                     ChromosomeEvaluator& evaluator, unsigned seed = 0)
        : resourceMinBorder(resourcesMinBorder), seed(seed), evaluator(evaluator), sizeSelection(sizeSelection),
          mutateOrderProb(mutateOrder), mutateResourcesProb(mutateResources), mutateContractorsProb(mutateContractors),
          crossOrderProb(crossOrder),   crossResourcesProb(crossResources),   crossContractorsProb(crossContractors) {
        // TODO
    }

    // TODO Add multi-criteria optimization (object hierarchy of fitness functions)
    Chromosome* run(vector<Chromosome*>& initialPopulation) {
        // TODO Ensure this is copy
        auto population = initialPopulation;

        int maxPlateau = 10;
        int curPlateau = 0;

        int prevFitness = INT_MAX;
        int bestFitness = prevFitness;
        Chromosome* bestChromosome = nullptr;

        auto fitness = evaluator.evaluate(initialPopulation);
        int g = 0;

        while (curPlateau < maxPlateau) {
            printf("--- Generation %i | fitness = %i\n", g, bestFitness);
            // update plateau info
            if (bestFitness == prevFitness) {
                curPlateau++;
            } else {
                curPlateau = 0;
            }

            auto offspring = selection(population, fitness);
            auto nextGeneration = applyAll(offspring);
            fitness = evaluator.evaluate(nextGeneration);

            int leaderPos = updateHOF(nextGeneration, fitness, bestFitness);
            if (leaderPos != -1) {
                bestChromosome = nextGeneration[leaderPos];
                bestFitness = fitness[leaderPos];
            }

            // renew population
            population.clear();
            population.insert(population.end(), offspring.begin(), offspring.end());
            population.insert(population.end(), nextGeneration.begin(), nextGeneration.end());

            g++;
        }

        return bestChromosome;
    }
};

#endif //NATIVE_GENETIC_H
