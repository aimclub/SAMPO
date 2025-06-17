from copy import copy
from random import Random
from typing import Any

import numpy as np
import pandas as pd

from sampo.generator import SimpleSynthetic, SyntheticGraphType
from sampo.generator.environment import get_contractor_by_wg, ContractorGenerationMethod
from sampo.pipeline import SchedulingPipeline
from sampo.scheduler import HEFTScheduler, HEFTBetweenScheduler
from sampo.scheduler.heft.prioritization import prioritization, prioritization_nodes, ford_bellman
from sampo.scheduler.resource import AverageReqResourceOptimizer
from sampo.scheduler.utils.time_computaion import calculate_working_time_cascade, work_priority
from sampo.schemas import WorkGraph, WorkTimeEstimator, GraphNode, WorkerProductivityMode, IntervalGaussian, WorkUnit, \
    uuid_str, WorkerReq, Time, ZoneConfiguration, DefaultZoneStatuses, ZoneReq, LandscapeConfiguration, Worker
from sampo.schemas.stochastic_graph import StochasticGraph, ProbabilisticFollowingStochasticGraphScheme, \
    StochasticGraphScheme
from sampo.schemas.time_estimator import DefaultWorkEstimator
from sampo.utilities.visualization import schedule_gant_chart_fig, VisualizationMode


def compare_results(static_wg: WorkGraph,
                    stochastic_graph: StochasticGraph,
                    work_estimator: WorkTimeEstimator,
                    work_estimator_without_zones: WorkTimeEstimator):
    static_prioritization = prioritization(static_wg, work_estimator)
    stochastic_wg = stochastic_graph.to_work_graph()
    contractors = [get_contractor_by_wg(stochastic_wg,
                                        method=ContractorGenerationMethod.AVG,
                                        contractor_name=f'Contractor_{i}') for i in range(10)]
    # contractors = [get_contractor_by_wg(stochastic_wg)]

    def stochastic_prioritization_function(wg: WorkGraph, work_estimator: WorkTimeEstimator) -> list[GraphNode]:
        # wg is stochastic-generated graph
        # here we can use only information from static graph
        # let's use static prioritization

        order = []
        for original_static_node in static_prioritization:
            static_node = stochastic_wg[original_static_node.id]
            order.append(static_node)
            # haha, get all defects...
            children = static_node.get_following_nodes()
            children_prioritization = prioritization_nodes(children, work_estimator)
            order.extend(children_prioritization)
        return order

    def stochastic_prioritization_function_enhanced(wg: WorkGraph, work_estimator: WorkTimeEstimator) -> list[GraphNode]:
        # wg is stochastic-generated graph
        # here we can use information from static graph's followers statistics
        # let's use static prioritization

        weights = {node: -(work_priority(node, calculate_working_time_cascade, work_estimator) + stochastic_graph.average_labor_cost(node))  # stochastic_wg[node.id].get_following_nodes())
                   for node in static_prioritization}

        path_weights = ford_bellman(static_prioritization, weights)

        ordered_nodes = [i[0] for i in sorted(path_weights.items(), key=lambda x: (x[1], x[0].id))
                         if not i[0].is_inseparable_son()]
        final_order = []
        for original_static_node in ordered_nodes:
            static_node = stochastic_wg[original_static_node.id]
            final_order.append(static_node)
            children_prioritization = prioritization_nodes(static_node.get_following_nodes(), work_estimator)
            final_order.extend(children_prioritization)
        return final_order

    zone_config = prepare_zone_config(stochastic_wg.nodes)
    landscape_config = LandscapeConfiguration(zone_config=zone_config)

    # GO!
    scheduler1 = HEFTBetweenScheduler(prioritization_f=stochastic_prioritization_function,
                                      resource_optimizer=AverageReqResourceOptimizer(k=100),
                                      work_estimator=work_estimator_without_zones)

    project1 = SchedulingPipeline.create() \
        .wg(stochastic_wg) \
        .contractors(contractors) \
        .work_estimator(work_estimator_without_zones) \
        .landscape(landscape_config) \
        .schedule(scheduler1) \
        .finish()[0]

    scheduler2 = HEFTBetweenScheduler(prioritization_f=stochastic_prioritization_function_enhanced,
                                      resource_optimizer=AverageReqResourceOptimizer(k=100),
                                      work_estimator=work_estimator)

    project2 = SchedulingPipeline.create() \
        .wg(stochastic_wg) \
        .contractors(contractors) \
        .work_estimator(work_estimator) \
        .landscape(landscape_config) \
        .schedule(scheduler2) \
        .finish()[0]

    # merged_schedule1 = project1.schedule.merged_stages_datetime_df('2022-01-01')
    #
    # # here we build an interactive HTML form with all info about scheduled works
    # schedule_gant_chart_fig(schedule_dataframe=merged_schedule1,
    #                         visualization=VisualizationMode.ShowFig,
    #                         remove_service_tasks=False)
    #
    # merged_schedule2 = project2.schedule.merged_stages_datetime_df('2022-01-01')
    #
    # # here we build an interactive HTML form with all info about scheduled works
    # schedule_gant_chart_fig(schedule_dataframe=merged_schedule2,
    #                         visualization=VisualizationMode.ShowFig,
    #                         remove_service_tasks=False)

    return project1.schedule.execution_time, project2.schedule.execution_time


def get_checks_wg(nodes_count: int, rand: Random) -> WorkGraph:
    nodes = [GraphNode(work_unit=WorkUnit(id=uuid_str(rand),
                                          name=f'Check #{i}',
                                          worker_reqs=[WorkerReq(kind='engineer', volume=Time(0), min_count=1, max_count=1)]),
                       parent_works=[]) for i in range(nodes_count)]
    construct_zones_for(nodes, rand, max_zone_count=10)
    return WorkGraph.from_nodes(nodes)


def construct_zones_for(nodes: list[GraphNode], rand: Random, max_zone_count: int = 10):
    for node in nodes:
        for i in range(rand.randint(0, max_zone_count + 1)):
            node.work_unit.zone_reqs.append(ZoneReq(kind=f'Zone #{i}', required_status=rand.randint(1, 2 + 1)))


def prepare_zone_config(graph: list[GraphNode]):
    def parse_requirements(reqs: list[list[ZoneReq]], parent: set[str] = None) -> set[str]:
        names = set()
        if parent is not None:
            names = copy(parent)
        for req_dict in reqs:
            for req in req_dict:
                names.add(req.kind)
        return names

    zone_names = parse_requirements([node.work_unit.zone_reqs for node in graph])
    zones_count = len(zone_names)

    return ZoneConfiguration(
        start_statuses={zone: 1 for zone in zone_names},
        time_costs=np.array([[1 for _ in range(zones_count)] for _ in range(zones_count)]),
        statuses=DefaultZoneStatuses()
    )


def construct_graph_scheme(graph_size: int, rand: Random, ss: SimpleSynthetic, defects_count: int = 5) -> tuple[WorkGraph, StochasticGraphScheme]:
    fixed_wg = get_checks_wg(graph_size, rand)

    # construct the stochastic graph scheme
    graph_scheme = ProbabilisticFollowingStochasticGraphScheme(rand=rand, wg=fixed_wg)
    defect_graphs = []
    for i, node in enumerate(fixed_wg.nodes):
        for _ in range(defects_count):
            defect = ss.graph_nodes(top_border=20)
            defect_graphs.append(defect)

            for defect_node in defect:
                defect_node.work_unit.zone_reqs = copy(node.work_unit.zone_reqs)
            # construct_zones_for(defect, rand, max_zone_count=20)

            defect_prob = rand.random()
            graph_scheme.add_part(node=node.id, nodes=defect, prob=defect_prob)

    return fixed_wg, graph_scheme


class ZoneReqToInitialValueWorkEstimator(DefaultWorkEstimator):
    def estimate_time(self, work_unit: WorkUnit, worker_list: list[Worker]) -> Time:
        return DefaultWorkEstimator.estimate_time(self, work_unit, worker_list) + 1  # zone statuses closing


if __name__ == '__main__':
    def construct_work_estimators(i: int, productivity: WorkerProductivityMode) \
            -> tuple[WorkTimeEstimator, WorkTimeEstimator]:
        work_estimator = DefaultWorkEstimator()
        work_estimator.set_productivity_mode(productivity)
        for worker in ['driver', 'fitter', 'manager', 'handyman', 'electrician', 'engineer']:
            work_estimator.set_worker_productivity(IntervalGaussian(0.2 * i + 0.2, 1, 0, 2), worker)

        zone_req_to_initial_value = ZoneReqToInitialValueWorkEstimator()
        zone_req_to_initial_value.set_productivity_mode(productivity)
        for worker in ['driver', 'fitter', 'manager', 'handyman', 'electrician', 'engineer']:
            zone_req_to_initial_value.set_worker_productivity(IntervalGaussian(0.2 * i + 0.2, 1, 0, 2), worker)

        return work_estimator, zone_req_to_initial_value

    work_estimator, zone_req_to_initial_value = construct_work_estimators(5, WorkerProductivityMode.Static)

    rand = Random(136)
    ss = SimpleSynthetic(rand)


    graph_sizes = range(100, 500 + 1, 100)
    attempts = 10
    # graph_sizes = [100]
    # attempts = 20

    results = []
    winrates = []

    for graph_size in graph_sizes:
        fixed_wg, graph_scheme = construct_graph_scheme(graph_size, rand, ss, defects_count=1)
        prepared_graph = graph_scheme.prepare_graph()

        for attempt in range(attempts):
            print(f'Running experiment on graph size {fixed_wg.vertex_count}, attempt {attempt}')
            baseline_makespan, solution_makespan = compare_results(fixed_wg, prepared_graph, work_estimator, zone_req_to_initial_value)
            print(f'Results | baseline = {baseline_makespan} | solution = {solution_makespan}')

            results.append([fixed_wg.vertex_count, attempt, baseline_makespan.value, solution_makespan.value])

            if solution_makespan < baseline_makespan:
                winrates.append((baseline_makespan - solution_makespan) / baseline_makespan)

    wins = len(winrates)
    total_attempts = len(graph_sizes) * attempts

    print(f'Wins: {wins}/{total_attempts}, average winrate = {round(sum(winrates) / wins * 100, 1)}%')

    df = pd.DataFrame.from_records(results, columns=['vertex_count', 'attempt', 'baseline_makespan', 'solution_makespan'])
    df.to_csv('ggg_results_1.csv')
