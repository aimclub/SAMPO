from copy import deepcopy
from operator import attrgetter, itemgetter

from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph
from sampo.schemas.schedule import ScheduledWork, Schedule
from sampo.schemas.time import Time
from sampo.utilities.collections_util import build_index


def validate_schedule(schedule: Schedule, wg: WorkGraph, contractors: list[Contractor]) -> None:
    """
    Checks if the schedule is correct and can be executed.
    If there is an error, this function raises AssertException with an appropriate message
    If it finishes without any exception, it means successful passing of the verification

    :param contractors:
    :param wg:
    :param schedule: to apply verification
    """
    # checking preconditions
    # check_all_workers_have_same_qualification(schedule.workGraph, schedule.resourcePools)

    if schedule.execution_time == Time.inf():
        return

    # checking the schedule itself
    _check_all_tasks_scheduled(schedule, wg)
    _check_parent_dependencies(schedule, wg)
    _check_all_tasks_have_valid_duration(schedule)
    _check_all_workers_correspond_to_worker_reqs(wg, schedule)
    _check_all_allocated_workers_do_not_exceed_capacity_of_contractors(schedule, contractors)


def _check_all_tasks_scheduled(schedule: Schedule, wg: WorkGraph) -> None:
    # 1. each task of the graph is scheduled
    scheduled_works = {work.id: work for work in schedule.works}
    absent_works = [node for node in wg.nodes if node.work_unit.id not in scheduled_works]

    assert len(absent_works) == 0, \
        f'Found works that are not scheduled:\n' \
        f'\t{[work.work_unit.id for work in absent_works]}'


def _check_parent_dependencies(schedule: Schedule, wg: WorkGraph) -> None:
    scheduled_works: dict[str, ScheduledWork] = {work.id: work for work in schedule.works}

    for node in wg.nodes:
        start, end = scheduled_works[node.work_unit.id].start_end_time
        for pnode in node.parents:
            pstart, pend = scheduled_works[pnode.work_unit.id].start_end_time
            assert pstart <= pend <= start <= end


def _check_all_tasks_have_valid_duration(schedule: Schedule) -> None:
    # 3. check if all tasks have duration appropriate for their working hours
    service_works_with_incorrect_duration = [
        work for work in schedule.works
        if work.is_service_unit and work.duration != 0
    ]

    assert len(service_works_with_incorrect_duration) == 0, \
        f'Found service works that have non-zero duration:\n' \
        f'\t{[work.id for work in service_works_with_incorrect_duration]}'

    # # TODO: make correct duration calculation
    # works_with_incorrect_duration = [
    #     work for work in schedule.works
    #     if not work.work_unit.is_service_unit and work.duration <= 0
    # ]
    #
    # assert len(works_with_incorrect_duration) == 0, \
    #     f'Found works that have incorrect duration:\n' \
    #     f'\t{[work.work_unit.id for work in works_with_incorrect_duration]}'


def _check_all_allocated_workers_do_not_exceed_capacity_of_contractors(schedule: Schedule,
                                                                       contractors: list[Contractor]) -> None:
    # dict[contractor_id, dict[worker_name, worker_count]]
    initial_contractors_state: dict[str, dict[str, int]] = {}
    for contractor in contractors:
        initial_contractors_state[contractor.id] = {}
        for w in contractor.workers.values():
            if w.name in initial_contractors_state[contractor.id].keys():
                initial_contractors_state[contractor.id][w.name] += w.count
            else:
                initial_contractors_state[contractor.id][w.name] = w.count
    contractors_state = deepcopy(initial_contractors_state)

    check_all_allocated_workers_do_not_exceed_capacity_of_contractors(schedule,
                                                                      initial_contractors_state,
                                                                      contractors_state)


def check_all_allocated_workers_do_not_exceed_capacity_of_contractors(schedule: Schedule,
                                                                      initial_worker_pool: dict[str, dict[str, int]],
                                                                      cur_worker_pool: dict[str, dict[str, int]]) \
        -> dict[str, dict[str, int]]:
    # 4. at each moment sum of allocated workers of all tasks for the same contractor
    # does not exceed capacity of this contractor
    ordered_start_end_events = sorted(
        (el for work in schedule.works
         for el in [('start', work.start_time, work), ('end', work.finish_time, work)]),
        key=itemgetter(1)
    )

    moment_pool: dict[str, dict[str, int]] = {}
    moment = Time(0)
    for index, (event_type, time, work) in enumerate(ordered_start_end_events):
        if len(work.workers) == 0:
            continue
        cont = work.workers[0].contractor_id
        cpool = moment_pool.get(cont, {})
        if len(cpool) == 0:
            moment_pool[cont] = cpool

        # if next equivalency class, check validity of resources distribution
        if time != moment:
            for contractor_id, contractor_pool in moment_pool.items():
                for worker_name, worker_count in contractor_pool.items():
                    available = cur_worker_pool[contractor_id][worker_name]
                    # check
                    assert available + worker_count >= 0, \
                        f'Overuse of workers (event type {event_type} ' \
                        f'at [index={index} of {len(ordered_start_end_events)}, time={time}]) ' \
                        f'for contractor "{contractor_id}" ' \
                        f'and worker type "{worker_name}": available {available},' \
                        f' while being allocated {-worker_count}'
                    assert available + worker_count <= initial_worker_pool[contractor_id][worker_name], \
                        f'Excessive workers appear for contractor "{contractor_id}" and worker type "{worker_name}":' \
                        f'available {available}, being returned {worker_count}, ' \
                        f'initial contractor capacity {initial_worker_pool[contractor_id][worker_name]}'

                    # update
                    cur_worker_pool[contractor_id][worker_name] = available + worker_count
            moment_pool.clear()
            moment = time

        cpool = moment_pool.get(cont, {})
        if len(cpool) == 0:
            moment_pool[cont] = cpool

        # grab event to the current equivalency class
        if event_type == 'start':
            for w in work.workers:
                cpool[w.name] = cpool.get(w.name, 0) - w.count
        elif event_type == 'end':
            for w in work.workers:
                cpool[w.name] = cpool.get(w.name, 0) + w.count
        else:
            raise ValueError(f'Incorrect event type: {event_type}. Only "start" and "end" are supported.')

    return cur_worker_pool


def _check_all_workers_correspond_to_worker_reqs(wg: WorkGraph, schedule: Schedule):
    for swork in schedule.works:
        worker2req = build_index(wg[swork.id].work_unit.worker_reqs, attrgetter('kind'))
        for worker in swork.workers:
            req = worker2req[worker.name]
            assert req.min_count <= worker.count <= req.max_count
