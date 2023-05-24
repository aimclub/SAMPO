#ifndef NATIVE_OPERATORS_H
#define NATIVE_OPERATORS_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"
#include "pycodec.h"
#include "evaluator_types.h"

#include <bits/stdc++.h>
#include <vector>

using namespace std;

namespace Operators {

    inline void mutateOrder(Chromosome* chromosome, unsigned seed) {
        int* order = chromosome->getOrder()[0];
        shuffle(order, order + chromosome->getOrder().size(),
                default_random_engine(seed));
    }

    inline void crossOrder(Chromosome* a, Chromosome* b, unsigned seed) {
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

    inline vector<int> sample_ind(int size, float prob, unsigned seed) {
        random_device rd;
        uniform_int_distribution<int> rnd(0, size);

        vector<int> result;
        result.resize((int)(prob * (float) rnd(rd)));

        for (int & i : result) {
            i = rnd(rd);
        }
        return result;
    }

    template<typename T>
    inline vector<T*> sample(vector<T*>& src, float prob, unsigned seed) {
        random_device rd;
        uniform_int_distribution<int> rnd(0, src.size());

        vector<T*> result;
        result.resize((int)(prob * (float) rnd(rd)));

        for (int i = 0; i < result.size(); i++) {
            result[i] = new T(src[rnd(rd)]);
        }
        return result;
    }

    void applyOrder(vector<Chromosome*>& chromosomes, vector<Chromosome*>& target,
                    float mutate, float cross, unsigned seed) {
        auto toMutate = sample(chromosomes, mutate, seed);
        // TODO pragma omp parallel
        for (int i = 0; i < toMutate.size(); i++) {
            mutateOrder(toMutate[i], seed);
        }

        // multiply with 2 because of 2 chromosomes needed for crossover operation
        auto toCross = sample(chromosomes, 2 * cross, seed);
        // TODO pragma omp parallel
        for (int i = 0; i < toCross.size(); i += 2) {
            crossOrder(toCross[i], toCross[i + 1], seed);
        }
        // add to global list
        for (auto i : toMutate) {
            target.push_back(i);
        }
        for (auto i : toCross) {
            target.push_back(i);
        }
    }

    inline void mutateResources(Chromosome* chromosome, Array2D<int>& resMin, float prob, unsigned seed) {
        auto resourcesToChange = sample_ind(chromosome->getResources().width(), 1, seed);
        auto worksToChange = sample_ind(chromosome->getOrder().width(), prob, seed);

        for (int work : worksToChange) {
            int* resMax = chromosome->getWorkResourceBorder(work);
            for (int res : resourcesToChange) {
                // TODO
            }
        }
    }

    inline void crossResources(Chromosome* a, Chromosome* b, unsigned seed) {

    }

    void applyResources(vector<Chromosome*>& chromosomes, vector<Chromosome*>& target,
                        float mutate, float cross, unsigned seed) {
        auto toMutate = sample(chromosomes, mutate, seed);
        // TODO pragma omp parallel
        for (int i = 0; i < toMutate.size(); i++) {
            mutateResources(toMutate[i], seed);
        }

        for (int i = 0; i < chromosomes.size(); i++) {
            Chromosome* a = chromosomes[i];
            // without i = j & j = i duplications
            for (int j = i; j < chromosomes.size(); j++) {
                Chromosome* b = chromosomes[j];
                mutateResources();
            }
        }
    }

    void applyContractors(vector<Chromosome*>& chromosomes, vector<Chromosome*>& target, float mutate, float cross) {

    }

    vector<Chromosome*> applyAll(vector<Chromosome*>& chromosomes,
                                 float mutateOrder, float mutateResources, float mutateContractors,
                                 float crossOrder, float crossResources, float crossContractors, unsigned seed) {
        vector<Chromosome*> offspring;
        applyOrder(chromosomes, offspring, mutateOrder, crossOrder, seed);
        applyResources(chromosomes, offspring, mutateResources, crossResources);
        applyContractors(chromosomes, offspring, mutateContractors, crossContractors);
        return offspring;
    }
}

#endif //NATIVE_OPERATORS_H
