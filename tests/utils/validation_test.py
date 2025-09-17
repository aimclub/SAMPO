from copy import deepcopy
from enum import Enum
from operator import attrgetter
from typing import Optional

from sampo.scheduler.utils import get_worker_contractor_pool, WorkerContractorPool
from sampo.schemas.graph import WorkGraph
from sampo.schemas.schedule import Schedule
from sampo.utilities.collections_util import build_index
from sampo.utilities.validation import _check_all_tasks_scheduled, _check_parent_dependencies, \
    _check_all_workers_correspond_to_worker_reqs, \
    _check_all_allocated_workers_do_not_exceed_capacity_of_contractors


class BreakType(Enum):
    OrderBrokenDependencies = -1
    ResourcesNotEnoughToWorkReq = 1
    ResourcesTooBigToWorkReq = 2
    ResourcesTooBigToSupplyByThisContractor = 3

    def is_order_break(self) -> bool:
        return self.value < 0

    def is_resources_break(self) -> bool:
        return not self.is_order_break()


def test_check_order_validity_right(setup_default_schedules):
    (setup_wg, _, _, _, _), setup_default_schedules = setup_default_schedules

    for scheduler, (schedule, _, _, _) in setup_default_schedules.items():
        try:
            _check_all_tasks_scheduled(schedule, setup_wg)
            _check_parent_dependencies(schedule, setup_wg)
        except AssertionError as e:
            raise AssertionError(f'Scheduler {scheduler} failed validation', e)


def test_check_order_validity_wrong(setup_default_schedules):
    (setup_wg, _, _, _, _), setup_default_schedules = setup_default_schedules

    for (schedule, _, _, _) in setup_default_schedules.values():
        for break_type in BreakType:
            if break_type.is_order_break():
                broken = break_schedule(break_type, schedule, setup_wg)
                thrown = False
                try:
                    _check_all_tasks_scheduled(broken, setup_wg)
                    _check_parent_dependencies(broken, setup_wg)
                except AssertionError:
                    thrown = True

                assert thrown


def test_check_resources_validity_right(setup_default_schedules):
    (setup_wg, setup_contractors, _, _, _), setup_default_schedules = setup_default_schedules

    for scheduler, (schedule, _, _, _) in setup_default_schedules.items():
        try:
            _check_all_workers_correspond_to_worker_reqs(setup_wg, schedule)
            _check_all_allocated_workers_do_not_exceed_capacity_of_contractors(schedule, setup_contractors)
        except AssertionError as e:
            raise AssertionError(f'Scheduler {scheduler} failed validation', e)


def test_check_resources_validity_wrong(setup_default_schedules):
    (setup_wg, setup_contractors, _, _, _), setup_default_schedules = setup_default_schedules
    setup_worker_pool = get_worker_contractor_pool(setup_contractors)

    for (schedule, _, _, _) in setup_default_schedules.values():
        for break_type in BreakType:
            if break_type.is_resources_break():
                broken = break_schedule(break_type, schedule, setup_wg, setup_worker_pool)
                thrown = False
                try:
                    _check_all_workers_correspond_to_worker_reqs(setup_wg, broken)
                    _check_all_allocated_workers_do_not_exceed_capacity_of_contractors(broken, setup_contractors)
                except AssertionError:
                    thrown = True

                assert thrown


def break_schedule(break_type: BreakType, schedule: Schedule, wg: WorkGraph,
                   agents: Optional[WorkerContractorPool] = None) -> Schedule:
    broken = deepcopy(schedule.to_schedule_work_dict)

    if break_type == BreakType.OrderBrokenDependencies:
        for swork in broken.values():
            parents = [broken[parent.work_unit.id] for parent in wg[swork.id].parents]
            if not parents or swork.start_time == 0:
                continue
            parent = parents[0]
            swork_duration = swork.duration
            parent_duration = parent.duration

            swork.start_time, parent.start_time = parent.start_time, swork.start_time
            swork.finish_time = swork.start_time + swork_duration
            parent.finish_time = parent.start_time + parent_duration
            break
    elif break_type == BreakType.ResourcesNotEnoughToWorkReq:
        for swork in broken.values():
            for worker in swork.workers:
                worker.count = -1
    elif break_type == BreakType.ResourcesTooBigToWorkReq:
        for swork in broken.values():
            worker2req = build_index(wg[swork.id].work_unit.worker_reqs, attrgetter('kind'))
            for worker in swork.workers:
                worker.count = worker2req[worker.model_name].max_count + 1000000
    elif break_type == BreakType.ResourcesTooBigToSupplyByThisContractor:
        for swork in broken.values():
            for worker in swork.workers:
                worker.count = agents[worker.model_name][worker.contractor_id].count + 1000000

    # print(f'Original: {list(schedule.to_schedule_work_dict.values())}')
    # print(f'Broken  : {list(broken.values())}')

    # this is here because without it validation tests may fail if you are running all tests
    # print()

    return Schedule.from_scheduled_works(broken.values(), wg)
