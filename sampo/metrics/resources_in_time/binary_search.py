from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Union

import numpy as np

from sampo.metrics.resources_in_time.base import ResourceOptimizer, is_resources_good, init_borders, prepare_answer
from sampo.scheduler.base import Scheduler
from sampo.schemas.contractor import Contractor, WorkerContractorPool, get_worker_contractor_pool
from sampo.schemas.graph import WorkGraph
from sampo.schemas.time import Time


class BinarySearchOptimizationType(Enum):
    Fast = auto()
    ItemByItem = auto()
    ItemByItemFastInit = auto()


@dataclass(frozen=True)
class BinarySearchOptimizer(ResourceOptimizer):
    scheduler: Scheduler
    method: BinarySearchOptimizationType = BinarySearchOptimizationType.ItemByItemFastInit
    worker_factor: int | None = 1000
    max_workers: int | None = 1000

    def optimize(self,
                 wg: WorkGraph,
                 deadline: Time,
                 agents: WorkerContractorPool | None = None,
                 dry_resources: bool | None = True) -> Union[tuple[Contractor, Time], tuple[None, None]]:
        if self.method is BinarySearchOptimizationType.Fast:
            optimize_func = union_search
        elif self.method is BinarySearchOptimizationType.ItemByItem:
            optimize_func = item_by_item_search
        elif self.method is BinarySearchOptimizationType.ItemByItemFastInit:
            optimize_func = item_by_item_fast_init_search
        else:
            raise Exception(f'Unknown Binary Search type: {self.method}')
        return optimize_func(wg, self.scheduler, deadline, self.worker_factor, self.max_workers,
                             agents, dry_resources=dry_resources)


def union_search(wg: WorkGraph,
                 scheduler: Scheduler,
                 deadline: Time,
                 worker_factor: int | None = 1000,
                 max_workers: int | None = 1000,
                 right_agents: WorkerContractorPool | None = None,
                 dry_resources: bool | None = True) -> (Contractor, Time) or (None, None):
    """
    Get optimal amount of resources with binary search.

    :param wg:
    :param scheduler:
    :param deadline:
    :param worker_factor:
    :param max_workers:
    :param right_agents:
    :param dry_resources:
    :return:
    """
    left_counts, right_counts, agent_names = \
        init_borders(wg, scheduler, deadline, worker_factor, max_workers, right_agents)
    if right_counts is None:
        return prepare_answer(left_counts, agent_names, wg, scheduler, False)
    if left_counts is None:
        return prepare_answer(right_counts, agent_names, wg, scheduler, False)

    while True:
        mid_counts: np.array = ((left_counts + right_counts) / 2).astype(int)
        if (mid_counts == right_counts).all() or (mid_counts == left_counts).all():
            break
        if is_resources_good(wg, mid_counts, agent_names, scheduler, deadline):
            right_counts = mid_counts
        else:
            left_counts = mid_counts

    return prepare_answer(right_counts, agent_names, wg, scheduler, dry_resources)


def item_by_item_fast_init_search(wg: WorkGraph,
                                  scheduler: Scheduler,
                                  deadline: Time,
                                  worker_factor: Optional[int] = 1000,
                                  max_workers: Optional[int] = 1000,
                                  right_agents: Optional[WorkerContractorPool] = None,
                                  dry_resources: Optional[bool] = False) -> (Contractor, Time) or (None, None):
    """
    Search optimal amount of each type of resources with binary search

    :param wg:
    :param scheduler:
    :param deadline:
    :param worker_factor:
    :param max_workers:
    :param right_agents:
    :param dry_resources:
    :return:
    """
    contractor, max_time = union_search(wg, scheduler, deadline,
                                        worker_factor, max_workers, right_agents,
                                        dry_resources=False)

    if contractor is None:
        return None, None
    agents = get_worker_contractor_pool([contractor])
    contractor, max_time = item_by_item_search(wg, scheduler, deadline, worker_factor, max_workers, agents,
                                               dry_resources)
    return contractor, max_time


def item_by_item_search(wg: WorkGraph, scheduler: Scheduler, deadline: Time,
                        worker_factor: Optional[int] = 1000,
                        max_workers: Optional[int] = 1000,
                        right_agents: Optional[WorkerContractorPool] = None,
                        dry_resources: Optional[bool] = True) -> (Contractor, Time) or (None, None):
    """
    Helper function for item_by_item_fast_init_search()

    :param wg:
    :param scheduler:
    :param deadline:
    :param worker_factor:
    :param max_workers:
    :param right_agents:
    :param dry_resources:
    :return:
    """
    left_counts, right_counts, agent_names = \
        init_borders(wg, scheduler, deadline, worker_factor, max_workers, right_agents)
    if right_counts is None:
        return prepare_answer(left_counts, agent_names, wg, scheduler, False)
    if left_counts is None:
        return prepare_answer(right_counts, agent_names, wg, scheduler, True)

    # loop to run through all types of resources
    for index in range(len(left_counts)):
        while True:
            mid_counts = right_counts.copy()
            mid_counts[index] = (right_counts[index] + left_counts[index]) / 2
            if mid_counts[index] == right_counts[index] or mid_counts[index] == left_counts[index]:
                break
            if is_resources_good(wg, mid_counts, agent_names, scheduler, deadline):
                right_counts[index] = mid_counts[index]
            else:
                left_counts[index] = mid_counts[index]

    return prepare_answer(right_counts, agent_names, wg, scheduler, dry_resources)
