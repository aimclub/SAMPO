#include "native/scheduler/genetic.h"
#include "native/utils.h"

void Genetic::mutateOrder(Chromosome *chromosome) {
    random_device rd;
    int *order = chromosome->getOrder()[0];
    std::shuffle(order, order + chromosome->getOrder().size(), rd);
}

void Genetic::crossOrder(Chromosome *a, Chromosome *b) {
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

void Genetic::mutateResources(Chromosome *chromosome) {
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

void Genetic::crossResources(Chromosome *a, Chromosome *b) {
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

void Genetic::mutateContractors(Chromosome *chromosome) {
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

void Genetic::crossContractors(Chromosome *a, Chromosome *b) {
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

void Genetic::applyOperators(
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

void Genetic::applyOrder(
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

void Genetic::applyResources(
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

void Genetic::applyContractors(
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

vector<Chromosome *> Genetic::applyAll(vector<Chromosome *> &chromosomes) {
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

vector<Chromosome *> Genetic::selection(vector<Chromosome *> &population) {
    auto top = argsort(population);
    vector<Chromosome *> result;
    result.resize(min(population.size(), (size_t)sizeSelection));
    for (int i = 0; i < result.size(); i++) {
        result[i] = new Chromosome(population[top[i]]);
    }
    return result;
}

Chromosome* Genetic::updateHOF(vector<Chromosome *> &population, Chromosome *oldLeader) {
    Chromosome *leader = oldLeader;
    for (auto &chromosome : population) {
        if (chromosome->fitness < leader->fitness) {
            leader = chromosome;
        }
    }
    return leader;
}
