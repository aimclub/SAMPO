from abc import ABC, abstractmethod
from operator import attrgetter
from typing import Iterable

from sampo.scheduler.timeline.base import Timeline
from sampo.schemas.contractor import WorkerContractorPool, Contractor
from sampo.schemas.graph import GraphNode
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.requirements import WorkerReq
from sampo.schemas.resources import Worker
from sampo.schemas.schedule import ScheduledWork
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.utilities.collections_util import build_index

PRIORITY_SHUFFLE_RADIUS = 0.5


class OrderLocalOptimizer(ABC):

    @abstractmethod
    def optimize(self, node_order: list[GraphNode], area: range) -> list[GraphNode]:
        ...


class ScheduleLocalOptimizer(ABC):
    """
    Base class for building local optimization methods applicable to the schedule.
    """

    def __init__(self, timeline_type: type[Timeline]):
        """
        :param timeline_type: timeline used for schedule recalculation
        """
        self._timeline_type = timeline_type

    @abstractmethod
    def optimize(self, scheduled_works: dict[GraphNode, ScheduledWork], node_order: list[GraphNode],
                 contractors: list[Contractor], landscape_config: LandscapeConfiguration,
                 spec: ScheduleSpec, worker_pool: WorkerContractorPool,
                 work_estimator: WorkTimeEstimator, assigned_parent_time: Time,
                 area: range) -> dict[GraphNode, ScheduledWork]:
        """
        Optimizes works `scheduled_works`, referenced by `node_order` and `area` parameters.

        Result writes to `scheduled_works` in-place.
        :param node_order:
        :param contractors:
        :param landscape_config:
        :param spec:
        :param worker_pool:
        :param work_estimator:
        :param assigned_parent_time:
        :param scheduled_works:
        :param area:
        """
        ...


def get_swap_candidates(node: GraphNode,
                        node_index: int,
                        candidates: Iterable[GraphNode],
                        node2ind: dict[GraphNode, int],
                        processed: set[GraphNode]) -> list[GraphNode]:
    """
    Abstract function to find nodes that can be swapped
    with given node without breaking topological order

    :param node: target node
    :param node_index: index of target node in global sequence
    :param candidates: list of candidates to swapping
    :param node2ind: a dict from node to its index
    :param processed: a set of nodes that should not be swapped yet
    """
    cur_children: set[GraphNode] = node.children_set

    def is_candidate_accepted(candidate: GraphNode) -> bool:
        if candidate in cur_children or candidate in processed:
            return False
        candidate_ind = node2ind[candidate]
        for child in cur_children:
            if node2ind.get(child, 0) >= candidate_ind:  # we have a child between us and candidate
                return False
        candidate_parents = candidate.parents
        for parent in candidate_parents:
            if node2ind.get(parent, 0) <= node_index:  # candidate has a parent between us and candidate
                return False
        return True

    return [candidate for candidate in candidates if is_candidate_accepted(candidate)]


class SwapOrderLocalOptimizer(OrderLocalOptimizer):
    """
    This performs just small shuffle that not breaks topological order.
    """

    def optimize(self, node_order: list[GraphNode], area: range) -> list[GraphNode]:
        """
        Change order of nodes (not breaks topological order) by swapping several nodes.
        """
        if node_order is None:
            return

        start_index = area.start
        end_index = area.stop

        # TODO Examine what is better: perform shuffling in nearly placed sub-seq or in whole sequence
        # node2cost = {node: work_priority(node, calculate_working_time_cascade, work_estimator) for node in sub_seq}

        # preprocessing
        node2ind: dict[GraphNode, int] = {node: start_index + ind for ind, node in
                                          enumerate(node_order[start_index:end_index])}

        # temporary for usability measurement
        swapped = 0

        processed: set[GraphNode] = set()
        for i in reversed(area):
            node = node_order[i]
            if node in processed:
                continue
            # cur_cost = node2cost[node]
            chain_candidates: list[GraphNode] = node_order[start_index:i]

            accepted_candidates = get_swap_candidates(node, i, chain_candidates, node2ind, processed)

            if accepted_candidates:
                chain_candidate = accepted_candidates[0]
                swap_idx = node2ind[chain_candidate]
                node_order[i], node_order[swap_idx] = node_order[swap_idx], node_order[i]
                # print(f'Swapped {i} and {swap_idx}')
                processed.add(chain_candidate)
                node2ind[chain_candidate] = i
                node2ind[node] = swap_idx
                swapped += 1

            processed.add(node)
        print(f'Swapped {swapped} times!')

        return node_order


class ParallelizeScheduleLocalOptimizer(ScheduleLocalOptimizer):
    """
    This method finds near placed works and turns it to run in parallel.
    It will take effect only if it's launched after scheduling.
    """

    def __init__(self, timeline_type: type[Timeline]):
        super().__init__(timeline_type)

    def recalc_schedule(self,
                        node_order: Iterable[GraphNode],
                        contractors: list[Contractor],
                        landscape_config: LandscapeConfiguration,
                        spec: ScheduleSpec,
                        node2swork: dict[GraphNode, ScheduledWork],
                        worker_pool: WorkerContractorPool,
                        assigned_parent_time: Time,
                        work_estimator: WorkTimeEstimator) -> dict[GraphNode, ScheduledWork]:
        """
        Recalculates duration and start-finish times in the whole given `seq`.
        This will be useful to call after `parallelize_local_sequence` method
        or other methods that can change the appointed set of workers.

        :param node_order: scheduled works to process
        :param contractors:
        :param landscape_config:
        :param spec:
        :param node2swork:
        :param worker_pool:
        :param assigned_parent_time:
        :param work_estimator: an optional WorkTimeEstimator object to estimate time of work
        """

        timeline = self._timeline_type(worker_pool, landscape_config)
        node2swork_new: dict[GraphNode, ScheduledWork] = {}

        id2contractor = build_index(contractors, attrgetter('name'))

        for index, node in enumerate(node_order):
            node_schedule = node2swork[node]
            work_spec = spec.get_work_spec(node.id)
            # st = timeline.find_min_start_time(node, node_schedule.workers, node2swork_new)
            # ft = st + node_schedule.get_actual_duration(work_estimator)
            timeline.schedule(node, node2swork_new, node_schedule.workers,
                              id2contractor[node_schedule.contractor], work_spec, None, work_spec.assigned_time,
                              assigned_parent_time, work_estimator)
            # node_schedule.start_end_time = (st, ft)
            node2swork_new[node] = node_schedule

        return node2swork_new

    def optimize(self, scheduled_works: dict[GraphNode, ScheduledWork], node_order: list[GraphNode],
                 contractors: list[Contractor], landscape_config: LandscapeConfiguration,
                 spec: ScheduleSpec, worker_pool: WorkerContractorPool,
                 work_estimator: WorkTimeEstimator, assigned_parent_time: Time,
                 area: range) -> dict[GraphNode, ScheduledWork]:
        """
        Finds near placed works and turns it to run in parallel.

        :param scheduled_works:
        :param node_order:
        :param contractors:
        :param landscape_config:
        :param spec:
        :param worker_pool:
        :param work_estimator:
        :param assigned_parent_time:
        :param area:
        :return:
        """
        start_index = area.start
        end_index = area.stop
        
        # preprocessing
        node2ind: dict[GraphNode, int] = {node: start_index + ind for ind, node in
                                          enumerate(node_order[start_index:end_index])}

        processed: set[GraphNode] = set()

        for i in reversed(area):
            node = node_order[i]
            if node in processed:
                continue
            chain_candidates: list[GraphNode] = node_order[start_index:i]
            accepted_candidates = get_swap_candidates(node, i, chain_candidates, node2ind, processed)

            my_schedule: ScheduledWork = scheduled_works[node]
            my_workers: dict[str, Worker] = build_index(my_schedule.workers, attrgetter('name'))
            my_schedule_reqs: dict[str, WorkerReq] = build_index(node.work_unit.worker_reqs, attrgetter('kind'))

            new_my_workers = {}

            # now accepted_candidates is a list of nodes that can(according to dependencies) run in parallel
            for candidate in accepted_candidates:
                candidate_schedule = scheduled_works[candidate]

                candidate_schedule_reqs: dict[str, WorkerReq] = build_index(candidate.work_unit.worker_reqs,
                                                                            attrgetter('kind'))

                new_candidate_workers: dict[str, int] = {}

                satisfy = True

                for candidate_worker in candidate_schedule.workers:
                    my_worker = my_workers.get(candidate_worker.name, None)
                    if my_worker is None:  # these two works are not compete for this worker
                        continue

                    need_me = my_workers[candidate_worker.name].count
                    need_candidate = candidate_worker.count

                    total = need_me + need_candidate
                    my_req = my_schedule_reqs[candidate_worker.name]
                    candidate_req = candidate_schedule_reqs[candidate_worker.name]
                    needed_min = my_req.min_count + candidate_req.min_count

                    if needed_min > total:  # these two works can't run in parallel
                        satisfy = False
                        break

                    candidate_worker_count = candidate_req.min_count
                    my_worker_count = my_req.min_count
                    total -= needed_min

                    add_me = min(my_req.max_count, total // 2)
                    add_candidate = min(candidate_req.max_count, total - add_me)

                    my_worker_count += add_me
                    candidate_worker_count += add_candidate

                    new_my_workers[candidate_worker.name] = my_worker_count
                    new_candidate_workers[candidate_worker.name] = candidate_worker_count

                if satisfy:  # replacement found, apply changes and leave candidates bruteforce
                    print(f'Found! {candidate.work_unit.name} {node.work_unit.name}')
                    for worker in my_schedule.workers:
                        worker_count = new_my_workers.get(worker.name, None)
                        if worker_count is not None:
                            worker.count = worker_count
                    for worker in candidate_schedule.workers:
                        worker_count = new_candidate_workers.get(worker.name, None)
                        if worker_count is not None:
                            worker.count = worker_count
                    # candidate_schedule.start_time = my_schedule.start_time
                    break

        return self.recalc_schedule(reversed(node_order), contractors, landscape_config, spec, scheduled_works,
                                    worker_pool, assigned_parent_time, work_estimator)


def optimize_local_sequence(seq: list[GraphNode],
                            start_ind: int,
                            end_ind: int,
                            work_estimator: WorkTimeEstimator):
    # TODO Try to find sets of works with nearly same resources and turn in to run in parallel or vice-versa
    pass
