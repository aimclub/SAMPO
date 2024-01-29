import random
from uuid import uuid4

import numpy as np

import multiprocess as mp

from deap import base, tools

from sampo.schemas import LandscapeConfiguration, Worker, Contractor, MaterialReq, WorkGraph, EdgeType
from sampo.structurator import graph_restructuring
from sampo.utilities.sampler import Sampler

from sampo.scheduler.genetic import (GeneticScheduler, TimeAndResourcesFitness, SumOfResourcesPeaksFitness,
                                     DeadlineResourcesFitness, TimeFitness, ScheduleGenerationScheme)
from sampo.scheduler.genetic.operators import FitnessFunction
from sampo.utilities.resource_usage import resources_peaks_sum

random.seed(2024)


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
    l3n1.work_unit.material_reqs = [MaterialReq('mat1', 50)]
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
    # def get_schedule(sgs_type):
    #     genetic = GeneticScheduler(number_of_generation=100,
    #                                mutate_order=0.05,
    #                                mutate_resources=0.05,
    #                                size_of_population=50,
    #                                fitness_constructor=fitness,
    #                                sgs_type=sgs_type,
    #                                optimize_resources=optimize_resources,
    #                                rand=random,
    #                                verbose=False)
    #
    #     schedule = genetic.schedule(wg, contractors, landscape=landscape)
    #     return schedule

    schedules = []
    for sgs_type in [ScheduleGenerationScheme.Parallel, ScheduleGenerationScheme.Serial]:
        genetic = GeneticScheduler(number_of_generation=100,
                                   mutate_order=0.05,
                                   mutate_resources=0.05,
                                   size_of_population=50,
                                   fitness_constructor=fitness,
                                   sgs_type=sgs_type,
                                   optimize_resources=optimize_resources,
                                   rand=random)

        schedule = genetic.schedule(wg, contractors, landscape=landscape)
        schedules.append(schedule)
    return schedules


def get_pareto_front_by_lexicographic_and_combined(wg, contractors, landscape, fitness_res, start_deadline, end_deadline):
    # def lexicographic_schedules(sgs_type, deadline):
    #     scheduler_lexicographic = GeneticScheduler(number_of_generation=100,
    #                                                mutate_order=0.05,
    #                                                mutate_resources=0.05,
    #                                                size_of_population=50,
    #                                                fitness_constructor=fitness_res,
    #                                                sgs_type=sgs_type,
    #                                                rand=random,
    #                                                verbose=False)
    #
    #     scheduler_lexicographic.set_deadline(deadline)
    #     schedule = scheduler_lexicographic.schedule(wg, contractors, landscape=landscape)
    #     return schedule
    #
    # def combined_schedules(sgs_type, deadline):
    #     scheduler_combined = GeneticScheduler(number_of_generation=100,
    #                                           mutate_order=0.05,
    #                                           mutate_resources=0.05,
    #                                           size_of_population=50,
    #                                           fitness_constructor=DeadlineResourcesFitness.prepare(deadline),
    #                                           sgs_type=sgs_type,
    #                                           optimize_resources=True,
    #                                           rand=random,
    #                                           verbose=False)
    #
    #     scheduler_combined.set_deadline(deadline)
    #
    #     schedule = scheduler_combined.schedule(wg, contractors, landscape=landscape)
    #     return schedule

    schedules = []
    for sgs_type in [ScheduleGenerationScheme.Parallel, ScheduleGenerationScheme.Serial]:
        for deadline in range(start_deadline + 1, end_deadline + 1):
            scheduler_lexicographic = GeneticScheduler(number_of_generation=100,
                                                       mutate_order=0.05,
                                                       mutate_resources=0.05,
                                                       size_of_population=50,
                                                       fitness_constructor=fitness_res,
                                                       sgs_type=sgs_type,
                                                       rand=random)

            scheduler_lexicographic.set_deadline(deadline)
            schedule = scheduler_lexicographic.schedule(wg, contractors, landscape=landscape)

            schedules.append(schedule)

            scheduler_combined = GeneticScheduler(number_of_generation=100,
                                                  mutate_order=0.05,
                                                  mutate_resources=0.05,
                                                  size_of_population=50,
                                                  fitness_constructor=DeadlineResourcesFitness.prepare(deadline),
                                                  sgs_type=sgs_type,
                                                  optimize_resources=True,
                                                  rand=random)

            scheduler_combined.set_deadline(deadline)

            schedule = scheduler_combined.schedule(wg, contractors, landscape=landscape)
            schedules.append(schedule)
    return schedules


def get_pareto_front_by_multiobjective(wg, contractors, landscape, fitness_multiobjective):
    pass


def convert_schedules_multiobjective(schedules, fitness_time_f, fitness_res_f):
    return [(fitness_time_f(schedule), fitness_res_f(schedule)) for schedule in schedules]


def get_pareto_front(wg, contractors, landscape, fitness_multiobjective, fitness_res, fitness_res_f):
    pareto_front = []
    fitness_time = TimeFitness
    fitness_time_f = lambda schedule: schedule.execution_time.value

    time_schedules = get_single_objective_solution(wg, contractors, landscape, fitness_time)
    time_fitnesses = convert_schedules_multiobjective(time_schedules, fitness_time_f, fitness_res_f)
    best_time_i = np.argmin([time for time, res in time_fitnesses])
    best_time, worst_res = time_fitnesses[best_time_i]
    pareto_front.append(time_fitnesses[best_time_i])

    res_schedules = get_single_objective_solution(wg, contractors, landscape, fitness_res, optimize_resources=True)
    res_fitnesses = convert_schedules_multiobjective(res_schedules, fitness_time_f, fitness_res_f)
    best_res_i = np.argmin([res for time, res in res_fitnesses])
    worst_time, best_res = res_fitnesses[best_res_i]
    pareto_front.append(res_fitnesses[best_res_i])

    get_pareto_front_by_lexicographic_and_combined(wg, contractors, landscape, fitness_multiobjective)


if __name__ == '__main__':
    wg = setup_wg()
    contractors = setup_contractors(wg, 3)
    landscape = setup_landscape()

