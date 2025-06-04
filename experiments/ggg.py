from random import Random

import pandas as pd

from sampo.generator import SimpleSynthetic, SyntheticGraphType
from sampo.generator.environment import get_contractor_by_wg
from sampo.pipeline import SchedulingPipeline
from sampo.scheduler import HEFTScheduler
from sampo.scheduler.heft.prioritization import prioritization, prioritization_nodes, ford_bellman
from sampo.scheduler.utils.time_computaion import calculate_working_time_cascade, work_priority
from sampo.schemas import WorkGraph, WorkTimeEstimator, GraphNode, WorkerProductivityMode, IntervalGaussian, WorkUnit, \
    uuid_str, WorkerReq, Time
from sampo.schemas.stochastic_graph import StochasticGraph, ProbabilisticFollowingStochasticGraphScheme, \
    StochasticGraphScheme
from sampo.schemas.time_estimator import DefaultWorkEstimator


def compare_results(static_wg: WorkGraph, stochastic_graph: StochasticGraph, work_estimator: WorkTimeEstimator):
    static_prioritization = prioritization(static_wg, work_estimator)
    stochastic_wg = stochastic_graph.to_work_graph()
    contractors = [get_contractor_by_wg(stochastic_wg)]

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

    # GO!
    scheduler1 = HEFTScheduler(prioritization_f=stochastic_prioritization_function,
                               work_estimator=work_estimator)

    project1 = SchedulingPipeline.create() \
        .wg(stochastic_wg) \
        .contractors(contractors) \
        .work_estimator(work_estimator) \
        .schedule(scheduler1) \
        .finish()[0]

    scheduler2 = HEFTScheduler(prioritization_f=stochastic_prioritization_function_enhanced,
                               work_estimator=work_estimator)

    project2 = SchedulingPipeline.create() \
        .wg(stochastic_wg) \
        .contractors(contractors) \
        .work_estimator(work_estimator) \
        .schedule(scheduler2) \
        .finish()[0]

    return project1.schedule.execution_time, project2.schedule.execution_time


def get_checks_wg(nodes_count: int) -> WorkGraph:
    nodes = [GraphNode(work_unit=WorkUnit(id=uuid_str(rand),
                                          name=f'Check #{i}',
                                          worker_reqs=[WorkerReq(kind='engineer', volume=Time(0), min_count=1, max_count=3)]),
                       parent_works=[]) for i in range(nodes_count)]
    return WorkGraph.from_nodes(nodes)


def construct_graph_scheme(graph_size: int, rand: Random, ss: SimpleSynthetic) -> tuple[WorkGraph, StochasticGraphScheme]:
    fixed_wg = get_checks_wg(graph_size)

    # construct the stochastic graph scheme
    graph_scheme = ProbabilisticFollowingStochasticGraphScheme(rand=rand, wg=fixed_wg)
    defect_graphs = []
    for i, node in enumerate(fixed_wg.nodes):
        defect = ss.graph_nodes(top_border=20)
        defect_graphs.append(defect)

        defect_prob = rand.random() * 0.5
        graph_scheme.add_part(node=node.id, nodes=defect, prob=defect_prob)

    return fixed_wg, graph_scheme


if __name__ == '__main__':
    def construct_work_estimator(i: int, productivity: WorkerProductivityMode) -> WorkTimeEstimator:
        work_estimator = DefaultWorkEstimator()
        work_estimator.set_productivity_mode(productivity)
        for worker in ['driver', 'fitter', 'manager', 'handyman', 'electrician', 'engineer']:
            work_estimator.set_worker_productivity(IntervalGaussian(0.2 * i + 0.2, 1, 0, 2), worker)
        return work_estimator

    work_estimator = construct_work_estimator(5, WorkerProductivityMode.Static)

    rand = Random()
    ss = SimpleSynthetic(rand)

    # graph_sizes = range(100, 500 + 1, 100)
    # attempts = 20
    graph_sizes = [100]
    attempts = 20

    results = []

    for graph_size in graph_sizes:
        fixed_wg, graph_scheme = construct_graph_scheme(graph_size, rand, ss)
        prepared_graph = graph_scheme.prepare_graph()

        for attempt in range(attempts):
            print(f'Running experiment on graph size {fixed_wg.vertex_count}, attempt {attempt}')
            baseline_makespan, solution_makespan = compare_results(fixed_wg, prepared_graph, work_estimator)
            print(f'Results | baseline = {baseline_makespan} | solution = {solution_makespan}')

            results.append([fixed_wg.vertex_count, attempt, baseline_makespan.value, solution_makespan.value])

    df = pd.DataFrame.from_records(results, columns=['vertex_count', 'attempt', 'baseline_makespan', 'solution_makespan'])
    df.to_csv('ggg_results.csv')
