import random
import numpy as np
from functools import partial
from typing import Type, Iterable

from sampo.scheduler.base import Scheduler, SchedulerType
from sampo.scheduler.timeline.momentum_timeline import MomentumTimeline
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator
from sampo.scheduler.lft.prioritization import lft_prioritization, lft_randomized_prioritization
from sampo.scheduler.lft.time_computaion import work_duration

from sampo.scheduler.timeline.base import Timeline
from sampo.schemas.contractor import Contractor, get_worker_contractor_pool, WorkerContractorPool
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.resources import Worker
from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.scheduled_work import ScheduledWork
from sampo.schemas.time import Time
from sampo.utilities.validation import validate_schedule

from sampo.schemas.exceptions import IncorrectAmountOfWorker, NoSufficientContractorError


class LFTScheduler(Scheduler):
    """
    Scheduler, which assigns contractors evenly, allocates maximum resources
    and schedules works in MIN-LFT priority rule order
    """

    def __init__(self,
                 scheduler_type: SchedulerType = SchedulerType.HEFTAddEnd,
                 timeline_type: Type = MomentumTimeline,
                 work_estimator: WorkTimeEstimator = DefaultWorkEstimator()):
        super().__init__(scheduler_type, None, work_estimator)
        self._timeline_type = timeline_type
        self.prioritization = lft_prioritization

    def schedule_with_cache(self,
                            wg: WorkGraph,
                            contractors: list[Contractor],
                            landscape: LandscapeConfiguration() = LandscapeConfiguration(),
                            spec: ScheduleSpec = ScheduleSpec(),
                            validate: bool = False,
                            assigned_parent_time: Time = Time(0),
                            timeline: Timeline | None = None) \
            -> tuple[Schedule, Time, Timeline, list[GraphNode]]:
        worker_pool = get_worker_contractor_pool(contractors)

        node_id2workers, node_id2duration = self._contractor_workers_assignment(wg, contractors, worker_pool, spec)

        ordered_nodes = self.prioritization(wg, node_id2duration)

        if not isinstance(timeline, self._timeline_type):
            timeline = self._timeline_type(worker_pool, landscape)

        schedule, schedule_start_time, timeline = self.build_scheduler(ordered_nodes, worker_pool, node_id2workers,
                                                                       landscape, spec, self.work_estimator,
                                                                       assigned_parent_time, timeline)
        schedule = Schedule.from_scheduled_works(
            schedule,
            wg
        )

        if validate:
            validate_schedule(schedule, wg, contractors)

        return schedule, schedule_start_time, timeline, ordered_nodes

    def _contractor_workers_assignment(self, wg: WorkGraph, contractors: list[Contractor],
                                       worker_pool: WorkerContractorPool, spec: ScheduleSpec = ScheduleSpec()
                                       ) -> tuple[dict[str, tuple[Contractor, list[Worker]]], dict[str, int]]:
        nodes = [node for node in wg.nodes if not node.is_inseparable_son()]
        contractors_assignments_count = np.zeros_like(contractors)
        node_id2workers = {}
        node_id2duration = {}
        for node in nodes:
            work_unit = node.work_unit
            work_reqs = work_unit.worker_reqs
            work_spec = spec.get_work_spec(work_unit.id)
            work_spec_amounts = np.array([work_spec.assigned_workers.get(req.kind, -1) for req in work_reqs])
            workers_mask = work_spec_amounts != -1

            min_req_amounts = np.array([req.min_count for req in work_reqs])
            if (work_spec_amounts[workers_mask] < min_req_amounts[workers_mask]).any():
                raise IncorrectAmountOfWorker(f"ScheduleSpec assigns not enough workers for work {node.id}")

            max_req_amounts = np.array([req.max_count for req in work_reqs])
            if (work_spec_amounts[workers_mask] > max_req_amounts[workers_mask]).any():
                raise IncorrectAmountOfWorker(f"ScheduleSpec assigns too many workers for work {node.id}")

            contractors_amounts = np.array([[worker_pool[req.kind][contractor.id].count
                                             if contractor.id in worker_pool[req.kind] else -1
                                             for req in work_reqs]
                                            for contractor in contractors])

            contractors_mask = ((contractors_amounts >= min_req_amounts) & (contractors_amounts != -1)).all(axis=1)
            contractors_mask &= (contractors_amounts[:, workers_mask] >= work_spec_amounts[workers_mask]).all(axis=1)
            if not any(contractors_mask):
                raise NoSufficientContractorError(f'There is no contractor that can satisfy given search; contractors: '
                                                  f'{contractors}')

            accepted_contractors = [contractor for contractor, is_accepted in zip(contractors, contractors_mask)
                                    if is_accepted]
            if workers_mask.all():
                assigned_amounts = np.broadcast_to(work_spec_amounts, (len(accepted_contractors),
                                                                       len(work_spec_amounts)))
            else:
                max_amounts = contractors_amounts[contractors_mask]
                max_amounts = np.stack(np.broadcast_arrays(max_amounts, max_req_amounts), axis=0).min(axis=0)
                assigned_amounts = max_amounts
                assigned_amounts[:, workers_mask] = work_spec_amounts[workers_mask]

            durations_for_chain = [work_duration(node, amounts, self.work_estimator) for amounts in assigned_amounts]
            durations = np.array([sum(chain_durations) for chain_durations in durations_for_chain])

            if durations.size == 1:
                contractor_index = 0
            else:
                min_duration = durations.min()
                max_duration = durations.max()
                scores = (durations - min_duration) / (max_duration - min_duration)
                scores = scores + contractors_assignments_count / contractors_assignments_count.sum()
                contractor_index = self._get_contractor_index(scores)

            assigned_amount = assigned_amounts[contractor_index]
            assigned_contractor = accepted_contractors[contractor_index]
            contractors_assignments_count[contractor_index] += 1

            workers = [worker_pool[req.kind][assigned_contractor.id].copy().with_count(amount)
                       for req, amount in zip(work_reqs, assigned_amount)]
            node_id2workers[node.id] = (assigned_contractor, workers)
            for duration, dep_node in zip(durations_for_chain[contractor_index], node.get_inseparable_chain_with_self()):
                node_id2duration[dep_node.id] = duration

        return node_id2workers, node_id2duration

    def _get_contractor_index(self, scores: np.ndarray) -> int:
        return np.argmin(scores)

    def build_scheduler(self,
                        ordered_nodes: list[GraphNode],
                        worker_pool: WorkerContractorPool,
                        node_id2workers: dict[str, tuple[Contractor, list[Worker]]],
                        landscape: LandscapeConfiguration = LandscapeConfiguration(),
                        spec: ScheduleSpec = ScheduleSpec(),
                        work_estimator: WorkTimeEstimator = DefaultWorkEstimator(),
                        assigned_parent_time: Time = Time(0),
                        timeline: Timeline | None = None) \
            -> tuple[Iterable[ScheduledWork], Time, Timeline]:
        """
        Schedule works with assigned order, contractors and workers

        :param landscape: landscape
        :param worker_pool: mapper of workers and amount at the contractor
        :param node_id2workers: mapper of node and assigned contractor, workers
        :param spec: spec for current scheduling
        :param ordered_nodes: sequence of nodes in scheduling order
        :param timeline: the previous used timeline can be specified to handle previously scheduled works
        :param assigned_parent_time: start time of the whole schedule(time shift)
        :param work_estimator: estimate time of work with assigned workers
        :return:
        """
        # dict for writing parameters of completed_jobs
        node2swork: dict[GraphNode, ScheduledWork] = {}
        # list for support the queue of workers
        if not isinstance(timeline, self._timeline_type):
            timeline = self._timeline_type(worker_pool, landscape)

        for index, node in enumerate(reversed(ordered_nodes)):  # the tasks with the highest rank will be done first
            contractor, workers = node_id2workers[node.id]
            work_spec = spec.get_work_spec(node.id)

            start_time, finish_time, _ = timeline.find_min_start_time_with_additional(node, workers, node2swork,
                                                                                      work_spec,
                                                                                      None, assigned_parent_time,
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
            timeline.schedule(node, node2swork, workers, contractor, work_spec,
                              start_time, work_spec.assigned_time, assigned_parent_time, work_estimator)

            if index == len(ordered_nodes) - 1:  # we are scheduling the work `end of the project`
                node2swork[node].zones_pre = finalizing_zones

        return node2swork.values(), assigned_parent_time, timeline


class RandomizedLFTScheduler(LFTScheduler):
    """
    Scheduler, which assigns contractors evenly with stochasticity, allocates maximum resources
    and schedules works in order sampled by MIN-LFT and MIN-LST priority rules
    """

    def __init__(self,
                 scheduler_type: SchedulerType = SchedulerType.HEFTAddEnd,
                 timeline_type: Type = MomentumTimeline,
                 work_estimator: WorkTimeEstimator = DefaultWorkEstimator(),
                 rand: random.Random = random.Random()):
        super().__init__(scheduler_type, timeline_type, work_estimator)
        self._random = rand
        self.prioritization = partial(lft_randomized_prioritization, rand=self._random)

    def _get_contractor_index(self, scores: np.ndarray) -> int:
        indexes = np.arange(len(scores))
        scores = 2 - scores
        return self._random.choices(indexes, weights=scores)[0]
