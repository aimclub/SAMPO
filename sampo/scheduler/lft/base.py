import random
import numpy as np
from functools import partial
from typing import Type, Callable

from sampo.scheduler.base import SchedulerType
from sampo.scheduler.generic import GenericScheduler, get_finish_time_default
from sampo.scheduler.timeline import Timeline, MomentumTimeline
from sampo.scheduler.utils import WorkerContractorPool, get_worker_contractor_pool
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator
from sampo.scheduler.lft.prioritization import lft_prioritization, lft_randomized_prioritization
from sampo.scheduler.lft.time_computaion import work_chain_durations

from sampo.schemas import (Contractor, WorkGraph, GraphNode, LandscapeConfiguration, Worker, Schedule, ScheduledWork,
                           Time, WorkUnit)
from sampo.schemas.schedule_spec import ScheduleSpec, WorkSpec
from sampo.utilities.validation import validate_schedule

from sampo.schemas.exceptions import IncorrectAmountOfWorker, NoSufficientContractorError


def get_contractors_and_workers_amounts_for_work(work_unit: WorkUnit, contractors: list[Contractor],
                                                 spec: ScheduleSpec, worker_pool: WorkerContractorPool,
                                                 get_workers_amounts: Callable[[np.ndarray, np.ndarray], np.ndarray]) \
        -> tuple[list[Contractor], np.ndarray]:
    """
    This function selects contractors that can perform the work.
    For each selected contractor, the maximum possible amount of workers is assigned,
    if they are not specified in the ScheduleSpec, otherwise the amount from the ScheduleSpec is used.
    """
    work_reqs = work_unit.worker_reqs
    work_spec = spec.get_work_spec(work_unit.id)
    # get assigned amounts of workers in schedule spec
    work_spec_amounts = np.array([work_spec.assigned_workers.get(req.kind, -1) for req in work_reqs])
    # make bool mask of unassigned amounts of workers
    in_spec_mask = work_spec_amounts != -1

    # get min amounts of workers
    min_req_amounts = np.array([req.min_count for req in work_reqs])
    # check validity of assigned in schedule spec amounts of workers
    if (work_spec_amounts[in_spec_mask] < min_req_amounts[in_spec_mask]).any():
        raise IncorrectAmountOfWorker(f"ScheduleSpec assigns not enough workers for work {work_unit.id}")

    # get max amounts of workers
    max_req_amounts = np.array([req.max_count for req in work_reqs])
    # check validity of assigned in schedule spec amounts of workers
    if (work_spec_amounts[in_spec_mask] > max_req_amounts[in_spec_mask]).any():
        raise IncorrectAmountOfWorker(f"ScheduleSpec assigns too many workers for work {work_unit.id}")

    # get contractors borders
    contractors_amounts = np.array([[worker_pool[req.kind][contractor.id].count
                                     if contractor.id in worker_pool[req.kind] else -1
                                     for req in work_reqs]
                                    for contractor in contractors])

    # make bool mask of contractors that satisfy min amounts of workers
    contractors_mask = (contractors_amounts >= min_req_amounts).all(axis=1)
    # update bool mask of contractors to satisfy amounts of workers assigned in schedule spec
    contractors_mask &= (contractors_amounts[:, in_spec_mask] >= work_spec_amounts[in_spec_mask]).all(axis=1)
    # check that there is at least one contractor that satisfies all the constraints
    if not contractors_mask.any():
        raise NoSufficientContractorError(f'There is no contractor that can satisfy given search; contractors: '
                                          f'{contractors}')

    # get contractors that satisfy all the constraints
    accepted_contractors = [contractor for contractor, is_satisfying in zip(contractors, contractors_mask)
                            if is_satisfying]
    if in_spec_mask.all():
        # if all workers are assigned in schedule spec
        # broadcast these amounts on all accepted contractors
        workers_amounts = np.broadcast_to(work_spec_amounts,
                                          (len(accepted_contractors), len(work_spec_amounts)))
    else:
        # if some workers are not assigned in schedule spec
        # then we should assign maximum to them for each contractor
        max_amounts = contractors_amounts[contractors_mask]  # get max amounts of accepted contractors
        # bring max amounts of accepted contractors and max amounts of workers to the same size
        # and take the minimum of them to satisfy all constraints
        max_amounts = np.stack(np.broadcast_arrays(max_amounts, max_req_amounts), axis=0).min(axis=0)
        workers_amounts = get_workers_amounts(max_amounts, np.broadcast_to(min_req_amounts, max_amounts.shape))
        # assign to all accepted contractors assigned in schedule spec amounts of workers
        workers_amounts[:, in_spec_mask] = work_spec_amounts[in_spec_mask]

    return accepted_contractors, workers_amounts


class LFTScheduler(GenericScheduler):
    """
    Scheduler, which assigns contractors evenly, allocates maximum resources
    and schedules works in MIN-LFT priority rule order
    """

    def __init__(self,
                 scheduler_type: SchedulerType = SchedulerType.LFT,
                 timeline_type: Type = MomentumTimeline,
                 work_estimator: WorkTimeEstimator = DefaultWorkEstimator()):
        super().__init__(scheduler_type, None, timeline_type, None, self.get_default_res_opt_function(),
                         work_estimator)
        self.prioritization = lft_prioritization

    def get_default_res_opt_function(self, get_finish_time=get_finish_time_default) \
            -> Callable[[GraphNode, list[Contractor], WorkSpec, WorkerContractorPool,
                         dict[GraphNode, ScheduledWork], Time, Timeline, WorkTimeEstimator],
            tuple[Time, Time, Contractor, list[Worker]]]:
        def optimize_resources_def(node: GraphNode, contractors: list[Contractor], spec: WorkSpec,
                                   worker_pool: WorkerContractorPool, node2swork: dict[GraphNode, ScheduledWork],
                                   assigned_parent_time: Time, timeline: Timeline, work_estimator: WorkTimeEstimator) \
                -> tuple[Time, Time, Contractor, list[Worker]]:
            # get assigned contractor and workers
            contractor, workers = self._node_id2workers[node.id]
            # find start time
            start_time, finish_time, _ = timeline.find_min_start_time_with_additional(node, workers, node2swork,
                                                                                      spec, None, assigned_parent_time,
                                                                                      work_estimator)
            return start_time, finish_time, contractor, workers

        return optimize_resources_def

    def schedule_with_cache(self,
                            wg: WorkGraph,
                            contractors: list[Contractor],
                            spec: ScheduleSpec = ScheduleSpec(),
                            validate: bool = False,
                            assigned_parent_time: Time = Time(0),
                            timeline: Timeline | None = None,
                            landscape: LandscapeConfiguration() = LandscapeConfiguration()) -> list[
        tuple[Schedule, Time, Timeline, list[GraphNode]]]:
        # get contractors borders
        worker_pool = get_worker_contractor_pool(contractors)

        # first of all assign workers and contractors to nodes
        # and estimate nodes' durations
        node_id2duration = self._contractor_workers_assignment(wg, contractors, worker_pool, spec)

        # order nodes based on estimated nodes' durations
        ordered_nodes = self.prioritization(wg, node_id2duration)

        if not isinstance(timeline, self._timeline_type):
            timeline = self._timeline_type(worker_pool, landscape)

        # make schedule based on assigned workers, contractors and order
        schedule, schedule_start_time, timeline = self.build_scheduler(ordered_nodes, contractors, landscape, spec,
                                                                       self.work_estimator, assigned_parent_time,
                                                                       timeline)
        del self._node_id2workers
        schedule = Schedule.from_scheduled_works(
            schedule,
            wg
        )

        if validate:
            validate_schedule(schedule, wg, contractors)

        return [(schedule, schedule_start_time, timeline, ordered_nodes)]

    def _contractor_workers_assignment(self, wg: WorkGraph, contractors: list[Contractor],
                                       worker_pool: WorkerContractorPool, spec: ScheduleSpec = ScheduleSpec()
                                       ) -> dict[str, int]:
        # get only heads of chains from work graph nodes
        nodes = [node for node in wg.nodes if not node.is_inseparable_son()]
        # counter for contractors assignments to the works
        contractors_assignments_count = np.ones_like(contractors)
        # mapper of nodes and assigned workers
        self._node_id2workers = {}
        # mapper of nodes and estimated duration
        node_id2duration = {}
        for node in nodes:
            work_unit = node.work_unit
            # get contractors that can perform this work and workers amounts for them
            accepted_contractors, workers_amounts = get_contractors_and_workers_amounts_for_work(work_unit,
                                                                                                 contractors,
                                                                                                 spec,
                                                                                                 worker_pool,
                                                                                                 self._get_workers_amounts)

            # estimate chain durations for each accepted contractor
            durations_for_chain = [work_chain_durations(node, amounts, self.work_estimator)
                                   for amounts in workers_amounts]
            # get the sum of the estimated durations for each contractor
            durations = np.array([sum(chain_durations) for chain_durations in durations_for_chain])

            # assign a score for each contractor equal to the sum of the ratios of
            # the duration of this work for this contractor to all durations
            # and the number of assignments of this contractor to the total amount of contractors assignments
            scores = durations / durations.sum() + contractors_assignments_count / contractors_assignments_count.sum()
            # since the maximum possible score value is 2 subtract the resulting scores from 2,
            # so that the higher the score, the more suitable the contractor is for the assignment
            scores = 2 - scores

            # assign contractor based on received scores by implemented strategy
            contractor_index = self._get_contractor_index(scores)
            assigned_contractor = accepted_contractors[contractor_index]

            # get workers amounts of the assigned contractor
            assigned_amount = workers_amounts[contractor_index]

            # increase the counter for the assigned contractor
            contractors_assignments_count[contractor_index] += 1

            # get workers of the assigned contractor and assign them to the node in mapper
            workers = [worker_pool[req.kind][assigned_contractor.id].copy().with_count(amount)
                       for req, amount in zip(work_unit.worker_reqs, assigned_amount)]
            self._node_id2workers[node.id] = (assigned_contractor, workers)

            # assign the received durations to each node in the chain
            for duration, dep_node in zip(durations_for_chain[contractor_index],
                                          node.get_inseparable_chain_with_self()):
                node_id2duration[dep_node.id] = duration

        return node_id2duration

    def _get_contractor_index(self, scores: np.ndarray) -> int:
        return np.argmax(scores)

    def _get_workers_amounts(self, max_amounts: np.ndarray, min_amounts: np.ndarray) -> np.ndarray:
        return max_amounts


class RandomizedLFTScheduler(LFTScheduler):
    """
    Scheduler, which assigns contractors evenly with stochasticity, allocates maximum resources
    and schedules works in order sampled by MIN-LFT and MIN-LST priority rules
    """

    def __init__(self,
                 scheduler_type: SchedulerType = SchedulerType.LFT,
                 timeline_type: Type = MomentumTimeline,
                 work_estimator: WorkTimeEstimator = DefaultWorkEstimator(),
                 rand: random.Random = random.Random()):
        super().__init__(scheduler_type, timeline_type, work_estimator)
        self._random = rand
        self.prioritization = partial(lft_randomized_prioritization, rand=self._random)

    def _get_contractor_index(self, scores: np.ndarray) -> int:
        return self._random.choices(np.arange(len(scores)), weights=scores)[0] if scores.size > 1 else 0

    def _get_workers_amounts(self, max_amounts: np.ndarray, min_amounts: np.ndarray) -> np.ndarray:
        amounts = np.array([[self._random.randint(min_, max_) for min_, max_ in zip(contractor_min, contractor_max)]
                            for contractor_min, contractor_max in zip(min_amounts, max_amounts)])
        mask = amounts.sum(axis=1) == 0
        if mask.any():
            indexes_to_max = [self._random.choice(np.arange(len(contractor_max))[contractor_max > 0])
                              for contractor_max in max_amounts[mask]]
            amounts[mask, indexes_to_max] = max_amounts[mask, indexes_to_max]
        return amounts
