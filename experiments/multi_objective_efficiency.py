import random
from uuid import uuid4
from functools import partial

import numpy as np

import multiprocess as mp

from deap import base, tools

from sampo.schemas import LandscapeConfiguration, Worker, Contractor, MaterialReq, WorkGraph, EdgeType, Time
from sampo.structurator import graph_restructuring
from sampo.utilities.sampler import Sampler

from sampo.scheduler.genetic import (GeneticScheduler, TimeAndResourcesFitness, SumOfResourcesPeaksFitness,
                                     DeadlineResourcesFitness, TimeFitness, ScheduleGenerationScheme)
from sampo.scheduler.genetic.operators import FitnessFunction, register_individual_constructor
from sampo.utilities.resource_usage import resources_peaks_sum

NUM_GEN = 100*2
POP_SIZE = 50*2
ORDER_PB = 0.05
RES_PB = 0.05
N_CPU = 5
VERBOSE = True

random.seed(2024)


class WeightedTimeWithResourcesFitness(FitnessFunction):
    """
    Fitness function that relies on finish time and the set of resources.
    """

    def __init__(self, evaluator, min_time=0, max_time=1, min_res=0, max_res=1, alpha=0.5):
        super().__init__(evaluator)
        min_time = min_time if max_time != min_time else 0
        max_time = max_time if max_time != min_time else 1
        min_res = min_res if max_res != min_res else 0
        max_res = max_res if max_res != min_res else 1
        self.min_time = min_time
        self.time_range = max_time - min_time
        self.min_res = min_res
        self.res_range = max_res - min_res
        self.alpha = alpha

    @staticmethod
    def prepare(min_time, max_time, min_res, max_res, alpha):
        return partial(WeightedTimeWithResourcesFitness, min_time=min_time, max_time=max_time, min_res=min_res,
                       max_res=max_res, alpha=alpha)

    def evaluate(self, chromosomes):
        evaluated = self._evaluator(chromosomes)
        return [(self.alpha * (schedule.execution_time.value - self.min_time) / self.time_range +
                 (1 - self.alpha) * (resources_peaks_sum(schedule) - self.min_res) / self.res_range,)
                for schedule in evaluated]


def setup_landscape():
    return LandscapeConfiguration()


def setup_wg():
    sr = Sampler(1e-1)
    l1n1 = sr.graph_node('l1n1', [], group='0', work_id='000001')
    l1n2 = sr.graph_node('l1n2', [], group='0', work_id='000002')

    l2n1 = sr.graph_node('l2n1', [(l1n1, 0, EdgeType.FinishStart)], group='1', work_id='000011')
    l2n2 = sr.graph_node('l2n2', [(l1n1, 0, EdgeType.FinishStart),
                                  (l1n2, 0, EdgeType.FinishStart)], group='1', work_id='000012')
    l2n3 = sr.graph_node('l2n3', [(l1n2, 1, EdgeType.LagFinishStart)], group='1', work_id='000013')

    l3n1 = sr.graph_node('l3n1', [(l2n1, 0, EdgeType.FinishStart),
                                  (l2n2, 0, EdgeType.FinishStart)], group='2', work_id='000021')
    l3n2 = sr.graph_node('l3n2', [(l2n2, 0, EdgeType.FinishStart)], group='2', work_id='000022')
    l3n3 = sr.graph_node('l3n3', [(l2n3, 1, EdgeType.LagFinishStart),
                                  (l2n2, 0, EdgeType.FinishStart)], group='2', work_id='000023')

    wg = WorkGraph.from_nodes([l1n1, l1n2, l2n1, l2n2, l2n3, l3n1, l3n2, l3n3])

    wg = graph_restructuring(wg)

    return wg


def setup_contractors(wg, num_contractors=1):
    resource_req_mean = {}
    resource_req_min = {}
    resource_req_max = {}
    for node in wg.nodes:
        for req in node.work_unit.worker_reqs:
            resource_req_min[req.kind] = max(resource_req_min.get(req.kind, 0), req.min_count)
            resource_req_max[req.kind] = max(resource_req_max.get(req.kind, 0), req.max_count)
            resource_req_mean[req.kind] = max(resource_req_mean.get(req.kind, 0), (req.min_count + req.max_count) // 2)

    contractors = []
    for i in range(num_contractors - 1):
        contractor_id = str(uuid4())
        workers = {}
        for req in resource_req_mean.keys():
            pb = random.random()
            if pb < 0.33:
                resource_req = resource_req_min
            elif pb < 0.66:
                resource_req = resource_req_mean
            else:
                resource_req = resource_req_max
            workers[req] = Worker(str(uuid4()), req, resource_req[req], contractor_id=contractor_id)
        contractors.append(Contractor(id=contractor_id,
                                      name='OOO Berezka',
                                      workers=workers,
                                      equipments={}))

    return contractors


def get_single_objective_solution(wg, contractors, landscape, fitness, optimize_resources=False):
    schedules = []
    for sgs_type in [ScheduleGenerationScheme.Parallel, ScheduleGenerationScheme.Serial]:
        genetic = GeneticScheduler(number_of_generation=NUM_GEN,
                                   mutate_order=ORDER_PB,
                                   mutate_resources=RES_PB,
                                   size_of_population=POP_SIZE,
                                   fitness_constructor=fitness,
                                   sgs_type=sgs_type,
                                   optimize_resources=optimize_resources,
                                   rand=random,
                                   verbose=VERBOSE)

        schedule = genetic.schedule(wg, contractors, landscape=landscape)
        schedules.append(schedule)
    return schedules


def get_weighted_sum_solutions(wg, contractors, landscape, min_time, max_time, min_res, max_res):
    def weighted_sum_solution(sgs_type, alpha):
        genetic = GeneticScheduler(number_of_generation=NUM_GEN,
                                   mutate_order=ORDER_PB,
                                   mutate_resources=RES_PB,
                                   size_of_population=POP_SIZE,
                                   fitness_constructor=WeightedTimeWithResourcesFitness.prepare(min_time, max_time,
                                                                                                min_res, max_res,
                                                                                                alpha),
                                   sgs_type=sgs_type,
                                   optimize_resources=True,
                                   rand=random,
                                   verbose=VERBOSE)

        schedule = genetic.schedule(wg, contractors, landscape=landscape)
        return schedule

    schedules = []
    configs = [(sgs_type, alpha / 10) for alpha in range(1, 10)
               for sgs_type in [ScheduleGenerationScheme.Parallel, ScheduleGenerationScheme.Serial]]
    with mp.Pool(N_CPU) as pool:
        schedules.extend(pool.starmap(weighted_sum_solution, configs))
    return schedules


def get_lexicographic_and_combined_solutions(wg, contractors, landscape, fitness_combined, fitness_res, start_deadline,
                                             end_deadline):
    def lexicographic_schedule(sgs_type, deadline):
        scheduler_lexicographic = GeneticScheduler(number_of_generation=NUM_GEN,
                                                   mutate_order=ORDER_PB,
                                                   mutate_resources=RES_PB,
                                                   size_of_population=POP_SIZE,
                                                   fitness_constructor=fitness_res,
                                                   sgs_type=sgs_type,
                                                   rand=random,
                                                   verbose=VERBOSE)

        scheduler_lexicographic.set_deadline(Time(deadline))
        schedule = scheduler_lexicographic.schedule(wg, contractors, landscape=landscape)
        return schedule

    def combined_schedule(sgs_type, deadline):
        scheduler_combined = GeneticScheduler(number_of_generation=NUM_GEN,
                                              mutate_order=ORDER_PB,
                                              mutate_resources=RES_PB,
                                              size_of_population=POP_SIZE,
                                              fitness_constructor=fitness_combined.prepare(Time(deadline)),
                                              sgs_type=sgs_type,
                                              optimize_resources=True,
                                              rand=random,
                                              verbose=VERBOSE)

        scheduler_combined.set_deadline(Time(deadline))
        schedule = scheduler_combined.schedule(wg, contractors, landscape=landscape)
        return schedule

    schedules = []
    configs = [(sgs_type, deadline) for deadline in range(start_deadline + 1, end_deadline + 1)
               for sgs_type in [ScheduleGenerationScheme.Parallel, ScheduleGenerationScheme.Serial]]
    with mp.Pool(N_CPU) as pool:
        schedules.extend(pool.starmap(lexicographic_schedule, configs))
        schedules.extend(pool.starmap(combined_schedule, configs))
    return schedules


def get_pareto_front_by_multiobjective(wg, contractors, landscape, fitness_multiobjective):
    schedules = []
    for sgs_type in [ScheduleGenerationScheme.Parallel, ScheduleGenerationScheme.Serial]:
        genetic = GeneticScheduler(number_of_generation=NUM_GEN,
                                   mutate_order=ORDER_PB,
                                   mutate_resources=RES_PB,
                                   size_of_population=POP_SIZE,
                                   fitness_constructor=fitness_multiobjective,
                                   fitness_weights=(-1, -1),
                                   optimize_resources=False,
                                   sgs_type=sgs_type,
                                   rand=random,
                                   verbose=VERBOSE)

        schedules.extend(genetic.schedule_multiobjective(wg, contractors, landscape=landscape))
    return schedules


def convert_schedules_multiobjective(schedules, fitness_res_f):
    fitness_time_f = lambda schedule: schedule.execution_time.value
    return [(fitness_time_f(schedule), fitness_res_f(schedule)) for schedule in schedules]


def get_pareto_front(wg, contractors, landscape, fitness_multiobjective, fitness_combined, fitness_res, fitness_res_f):
    pareto_front = []
    fitness_time = TimeFitness

    time_schedules = get_single_objective_solution(wg, contractors, landscape, fitness_time)
    time_fitnesses = convert_schedules_multiobjective(time_schedules, fitness_res_f)
    pareto_front.append(time_fitnesses)
    best_time_i = np.argmin([time for time, res in time_fitnesses])
    best_time, worst_res = time_fitnesses[best_time_i]
    time_fit = time_fitnesses[best_time_i]

    res_schedules = get_single_objective_solution(wg, contractors, landscape, fitness_res, optimize_resources=True)
    res_fitnesses = convert_schedules_multiobjective(res_schedules, fitness_res_f)
    pareto_front.append(res_fitnesses)
    best_res_i = np.argmin([res for time, res in res_fitnesses])
    worst_time, best_res = res_fitnesses[best_res_i]
    res_fit = res_fitnesses[best_res_i]

    schedules = get_weighted_sum_solutions(wg, contractors, landscape, best_time, worst_time, best_res, worst_res)
    fitnesses = convert_schedules_multiobjective(schedules, fitness_res_f)
    pareto_front.extend(fitnesses)
    sum_fit = fitnesses

    schedules = get_lexicographic_and_combined_solutions(wg, contractors, landscape, fitness_combined,
                                                         fitness_res, best_time, worst_time)
    fitnesses = convert_schedules_multiobjective(schedules, fitness_res_f)
    pareto_front.extend(fitnesses)
    lex_com_fit = fitnesses

    schedules = get_pareto_front_by_multiobjective(wg, contractors, landscape, fitness_multiobjective)
    fitnesses = convert_schedules_multiobjective(schedules, fitness_res_f)
    pareto_front.extend(fitnesses)
    mo_fit = fitnesses

    # pareto_front = list(set(pareto_front))

    tb = base.Toolbox()
    register_individual_constructor((-1, -1), tb)

    new_pareto = []
    ind = tb.Individual([])
    ind.fitness.values = time_fit
    ind.type = 'time'
    new_pareto.append(ind)
    ind = tb.Individual([])
    ind.fitness.values = res_fit
    ind.type = 'res'
    new_pareto.append(ind)
    for fit in sum_fit:
        ind = tb.Individual([])
        ind.fitness.values = fit
        ind.type = 'sum'
        new_pareto.append(ind)
    for fit in lex_com_fit:
        ind = tb.Individual([])
        ind.fitness.values = fit
        ind.type = 'lex_com'
        new_pareto.append(ind)
    for fit in mo_fit:
        ind = tb.Individual([])
        ind.fitness.values = fit
        ind.type = 'mo'
        new_pareto.append(ind)
    new_pareto = tools.sortNondominated(new_pareto, k=len(new_pareto), first_front_only=True)[0]
    algs_pareto = [ind.type for ind in new_pareto]
    print(algs_pareto)
    new_pareto = list(set([ind.fitness.values for ind in new_pareto]))
    return new_pareto

    # individuals_pareto_front = [tb.Individual([]) for _ in pareto_front]
    # for ind, values in zip(individuals_pareto_front, pareto_front):
    #     ind.fitness.values = values
    # individuals_pareto_front = tools.sortNondominated(individuals_pareto_front, k=len(pareto_front),
    #                                                   first_front_only=True)[0]
    # pareto_front = [ind.fitness.values for ind in individuals_pareto_front]
    #
    # return pareto_front


def calculate_distance(ref_set, mo_schedules, fitness_res_f):
    mo_fitnesses = np.array(convert_schedules_multiobjective(mo_schedules, fitness_res_f))
    ref_fitnesses = np.array(ref_set)
    weights = 1 / (ref_fitnesses.max() - ref_fitnesses.min())
    dist = 0
    for fitness in mo_fitnesses:
        c = np.max(weights * (fitness - ref_fitnesses), axis=1, initial=0)
        dist += min(c)
    dist /= len(ref_fitnesses)
    return dist


if __name__ == '__main__':
    wg = setup_wg()
    contractors = setup_contractors(wg, 3)
    landscape = setup_landscape()
    # pareto_front = get_pareto_front(wg, contractors, landscape, TimeAndResourcesFitness, DeadlineResourcesFitness,
    #                                 SumOfResourcesPeaksFitness, resources_peaks_sum)
    # print(pareto_front)
    pareto_front = я
    schedules = get_pareto_front_by_multiobjective(wg, contractors, landscape, TimeAndResourcesFitness)
    print('\nMeasures:')
    print(len(schedules))
    print(calculate_distance(pareto_front, schedules, resources_peaks_sum))

