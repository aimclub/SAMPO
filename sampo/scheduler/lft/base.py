import random
import numpy as np
from functools import partial
from typing import Type, Iterable

from sampo.scheduler.base import Scheduler, SchedulerType
from sampo.scheduler.timeline import Timeline, MomentumTimeline
from sampo.scheduler.utils import (WorkerContractorPool, get_worker_contractor_pool,
                                   get_head_nodes_with_connections_mappings)
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator
from sampo.scheduler.lft.prioritization import lft_prioritization, lft_randomized_prioritization_core, \
    lft_prioritization_core
from sampo.scheduler.lft.time_computaion import get_chain_duration

from sampo.schemas import (Contractor, WorkGraph, GraphNode, LandscapeConfiguration, Schedule, ScheduledWork,
                           Time, WorkUnit)
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.utilities.validation import validate_schedule

from sampo.schemas.exceptions import IncorrectAmountOfWorker, NoSufficientContractorError


def get_contractors_and_workers_amounts_for_work(work_unit: WorkUnit, contractors: list[Contractor],
                                                 spec: ScheduleSpec, worker_pool: WorkerContractorPool) \
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
        workers_amounts = max_amounts
        # assign to all accepted contractors assigned in schedule spec amounts of workers
        workers_amounts[:, in_spec_mask] = work_spec_amounts[in_spec_mask]

    return accepted_contractors, workers_amounts


class LFTScheduler(Scheduler):
    """
    Scheduler, which assigns contractors evenly, allocates maximum resources
    and schedules works in MIN-LFT priority rule order
    """

    def __init__(self,
                 scheduler_type: SchedulerType = SchedulerType.LFT,
                 timeline_type: Type = MomentumTimeline,
                 work_estimator: WorkTimeEstimator = DefaultWorkEstimator()):
        super().__init__(scheduler_type, None, work_estimator)
        self._timeline_type = timeline_type
        self._prioritization = partial(lft_prioritization, core_f=lft_prioritization_core)

    def schedule_with_cache(self,
                            wg: WorkGraph,
                            contractors: list[Contractor],
                            spec: ScheduleSpec = ScheduleSpec(),
                            validate: bool = False,
                            assigned_parent_time: Time = Time(0),
                            timeline: Timeline | None = None,
                            landscape: LandscapeConfiguration() = LandscapeConfiguration()
                            ) -> list[tuple[Schedule, Time, Timeline, list[GraphNode]]]:
        # get contractors borders
        worker_pool = get_worker_contractor_pool(contractors)

        # get head nodes with mappings
        head_nodes, node_id2parent_ids, node_id2child_ids = get_head_nodes_with_connections_mappings(wg)

        # first of all assign workers and contractors to head nodes
        # and estimate head nodes' durations
        node_id2duration = self._contractor_workers_assignment(head_nodes, contractors, worker_pool, spec)

        # order head nodes based on estimated durations
        ordered_nodes = self._prioritization(head_nodes, node_id2parent_ids, node_id2child_ids, node_id2duration)

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
            validate_schedule(schedule, wg, contractors, spec)

        return [(schedule, schedule_start_time, timeline, ordered_nodes)]

    def build_scheduler(self,
                        ordered_nodes: list[GraphNode],
                        contractors: list[Contractor],
                        landscape: LandscapeConfiguration = LandscapeConfiguration(),
                        spec: ScheduleSpec = ScheduleSpec(),
                        work_estimator: WorkTimeEstimator = DefaultWorkEstimator(),
                        assigned_parent_time: Time = Time(0),
                        timeline: Timeline | None = None) \
            -> tuple[Iterable[ScheduledWork], Time, Timeline]:
        worker_pool = get_worker_contractor_pool(contractors)
        # dict for writing parameters of completed_jobs
        node2swork: dict[GraphNode, ScheduledWork] = {}
        # list for support the queue of workers
        if not isinstance(timeline, self._timeline_type):
            timeline = self._timeline_type(worker_pool, landscape)

        for index, node in enumerate(ordered_nodes):
            work_unit = node.work_unit
            work_spec = spec.get_work_spec(work_unit.id)

            # get assigned contractor and workers
            contractor, best_worker_team = self._node_id2workers[node.id]
            # find start time
            start_time, finish_time, _ = timeline.find_min_start_time_with_additional(node, best_worker_team,
                                                                                      node2swork, work_spec, None,
                                                                                      assigned_parent_time,
                                                                                      work_estimator)
            # we are scheduling the work `start of the project`
            if index == 0:
                # this work should always have start_time = 0, so we just re-assign it
                start_time = assigned_parent_time
                finish_time += start_time

            if index == len(ordered_nodes) - 1:  # we are scheduling the work `end of the project`
                finish_time, finalizing_zones = timeline.zone_timeline.finish_statuses()
                start_time = max(start_time, finish_time)

            # apply work to scheduling
            timeline.schedule(node, node2swork, best_worker_team, contractor, work_spec,
                              start_time, work_spec.assigned_time, assigned_parent_time, work_estimator)

            if index == len(ordered_nodes) - 1:  # we are scheduling the work `end of the project`
                node2swork[node].zones_pre = finalizing_zones

        return node2swork.values(), assigned_parent_time, timeline

    def _contractor_workers_assignment(self, head_nodes: list[GraphNode], contractors: list[Contractor],
                                       worker_pool: WorkerContractorPool, spec: ScheduleSpec = ScheduleSpec()
                                       ) -> dict[str, int]:
        # counter for contractors assignments to the works
        contractors_assignments_count = np.ones_like(contractors)
        # mapper of nodes and assigned workers
        self._node_id2workers = {}
        # mapper of nodes and estimated duration
        node_id2duration = {}
        for node in head_nodes:
            work_unit = node.work_unit
            # get contractors that can perform this work and workers amounts for them
            accepted_contractors, workers_amounts = get_contractors_and_workers_amounts_for_work(work_unit,
                                                                                                 contractors,
                                                                                                 spec,
                                                                                                 worker_pool)

            # estimate chain durations for each accepted contractor
            durations = np.array([get_chain_duration(node, amounts, self.work_estimator)
                                  for amounts in workers_amounts])

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

            # assign the received duration to the node
            node_id2duration[node.id] = durations[contractor_index]

        return node_id2duration

    def _get_contractor_index(self, scores: np.ndarray) -> int:
        return np.argmax(scores)


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
        self._prioritization = partial(lft_prioritization, rand=self._random, core_f=lft_randomized_prioritization_core)

    def _get_contractor_index(self, scores: np.ndarray) -> int:
        return self._random.choices(np.arange(len(scores)), weights=scores)[0] if scores.size > 1 else 0
