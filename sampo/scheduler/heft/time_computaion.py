from typing import Callable
from uuid import uuid4

from sampo.schemas.graph import GraphNode
from sampo.schemas.resources import Worker
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator
from sampo.schemas.works import WorkUnit


def calculate_working_time_cascade(node: GraphNode, appointed_workers: list[Worker],
                                   work_estimator: WorkTimeEstimator) -> Time:
    """
    Calculate the working time of the appointed workers at a current job for prioritization.
    O(1) - at worst case |inseparable_edges|

    :param appointed_worker:
    :param work_estimator:
    :param node: the target node
    :return: working time
    """
    if node.is_inseparable_son():
        # time of employment of resources and task execution is calculated only for the first job
        # in the chain of connected inextricably
        return Time(0)

    # calculation of the time for all work_units inextricably linked to the given
    common_time = Time(0)
    for dep_node in node.get_inseparable_chain_with_self():
        common_time += work_estimator.estimate_time(dep_node.work_unit, appointed_workers)
    return common_time


def calculate_working_time(work_unit: WorkUnit, appointed_worker: list[Worker],
                           work_estimator: WorkTimeEstimator = DefaultWorkEstimator) -> Time:
    """
    Calculate the working time of the appointed workers at a current job for final schedule

    :return: working time
    """
    return work_estimator.estimate_time(work_unit, appointed_worker)  # working time


PRIORITY_DELTA = 1


def work_priority(node: GraphNode,
                  comp_cost: Callable[[GraphNode, list[Worker], WorkTimeEstimator], Time],
                  work_estimator: WorkTimeEstimator) -> float:
    """
    Calculate the average time to complete the work when assigning the minimum and maximum number of employees
    for the correct calculations of rank in prioritization
    O(sum_of_max_counts_of_workers) of current work

    :param node: the target node
    :param work_estimator:
    :param comp_cost: function for calculating working time (calculate_working_time)
    :return: average working time
    """

    work_unit = node.work_unit

    passed_workers_min = [Worker(str(uuid4()), req.kind, req.min_count)
                          for req in work_unit.worker_reqs]

    passed_workers_max = [Worker(str(uuid4()), req.kind, req.max_count)
                          for req in work_unit.worker_reqs]

    res = (comp_cost(node, passed_workers_min, work_estimator) +
           comp_cost(node, passed_workers_max, work_estimator)) / 2

    return res + PRIORITY_DELTA
