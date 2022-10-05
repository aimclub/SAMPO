from copy import deepcopy
from typing import Dict, List

from schemas.contractor import Contractor
from schemas.schedule import ScheduledWork, Schedule
from schemas.resources import Worker
from schemas.graph import WorkGraph


def validate_schedule(schedule: Schedule, wg: WorkGraph, contractors: List[Contractor]) -> None:
    """
    Checks if schedule is correct and can be executed.
    If there is an error, this function raises AssertException with an appropriate message
    If it finishes without any exception, it means successful passing of the verification
    :param contractors:
    :param wg:
    :param schedule: to apply verification to
    """
    # checking preconditions
    # check_all_workers_have_same_qualification(schedule.workGraph, schedule.resourcePools)

    # checking the schedule itself
    _check_all_tasks_scheduled(schedule, wg)
    _check_parent_dependencies(schedule, wg)
    _check_all_tasks_have_allocated_workers(schedule)
    _check_all_tasks_have_valid_duration(schedule)
    _check_all_allocated_workers_do_not_exceed_capacity_of_contractors(schedule, contractors)


def _check_all_tasks_scheduled(schedule: Schedule, wg: WorkGraph) -> None:
    # 1. each task of the graph is scheduled
    scheduled_works = {work.work_unit.id: work for work in schedule.works}
    absent_works = [node for node in wg.nodes if node.work_unit.id not in scheduled_works]

    assert len(absent_works) == 0, \
        f"Found works that are not scheduled:\n" \
        f"\t{[work.work_unit.id for work in absent_works]}"


def _check_parent_dependencies(schedule: Schedule, wg: WorkGraph) -> None:
    scheduled_works: Dict[str, ScheduledWork] = {work.work_unit.id: work for work in schedule.works}

    for node in wg.nodes:
        start, end = scheduled_works[node.work_unit.id].start_end_time
        for p_node in node.parent_nodes:
            p_start, p_end = scheduled_works[p_node.work_unit.id].start_end_time
            assert p_start <= p_end <= start <= end


def _check_all_tasks_have_allocated_workers(schedule: Schedule) -> None:
    # 2. each task has allocated workers (except for 'service' tasks) that satisfies at least its minimal requirements
    works_with_insufficient_allocated_workers = []
    for work in schedule.works:
        if work.work_unit.is_service_unit:
            assert len(work.workers) == 0, f"Service work unit {work.work_unit.id} cannot have assigned workers"
            continue

        if len(work.workers) == 0:
            # any non-service work unit should have assigned workers
            works_with_insufficient_allocated_workers.append(work)
            continue

        cat2worker: Dict[str, Worker] = {w.name: w for w in work.workers}
        # if any(cat2worker[w_req.req_worker].count < w_req.min_commands_count * w_req.command_member_count
        if any(cat2worker[w_req.type].count < w_req.min_count
               for w_req in work.work_unit.worker_reqs):
            works_with_insufficient_allocated_workers.append(work)

    assert len(works_with_insufficient_allocated_workers) == 0, \
        f"Found works that don't have sufficient amount of assigned workers:\n" \
        f"\t{[work.work_unit.id for work in works_with_insufficient_allocated_workers]}"


def _check_all_tasks_have_valid_duration(schedule: Schedule) -> None:
    # 3. check if all tasks have duration appropriate for their working hours
    service_works_with_incorrect_duration = [
        work for work in schedule.works
        if work.work_unit.is_service_unit and work.duration != 0
    ]

    assert len(service_works_with_incorrect_duration) == 0, \
        f"Found service works that have non-zero duration:\n" \
        f"\t{[work.work_unit.id for work in service_works_with_incorrect_duration]}"

    # TODO: make correct duration calculation
    works_with_incorrect_duration = [
        work for work in schedule.works
        if not work.work_unit.is_service_unit and work.duration <= 0
    ]

    assert len(works_with_incorrect_duration) == 0, \
        f"Found works that have incorrect duration:\n" \
        f"\t{[work.work_unit.id for work in works_with_incorrect_duration]}"


def _check_all_allocated_workers_do_not_exceed_capacity_of_contractors(schedule: Schedule,
                                                                       contractors: List[Contractor]) -> None:
    # 4. at each moment sum of allocated workers of all tasks for the same contractor
    # does not exceed capacity of this contractor
    ordered_start_end_events = sorted(
        (el for work in schedule.works
         for el in [("start", work.start_time, work), ("end", work.finish_time, work)]),
        key=lambda x: x[1]
    )

    # Dict[contractor_id, Dict[worker_name, worker_count]]
    initial_contractors_state: Dict[str, Dict[str, int]] = {}
    #     contractor.id: {w.name: w.count for _, w in contractor.workers.items()}
    #     for contractor in schedule.resourcePools
    # }
    for contractor in contractors:
        initial_contractors_state[contractor.id] = {}
        for w in contractor.workers.values():
            if w.name in initial_contractors_state[contractor.id].keys():
                initial_contractors_state[contractor.id][w.name] += w.count
            else:
                initial_contractors_state[contractor.id][w.name] = w.count

    contractors_state = deepcopy(initial_contractors_state)

    for event_type, time, work in ordered_start_end_events:
        if event_type == "start":
            for w in work.workers:
                assert contractors_state[w.contractor_id][w.name] >= w.count, \
                    f"Overuse of workers (event type {event_type} at [{time}]) for contractor '{w.contractor_id}' " \
                    f"and worker type '{w.name}': available {contractors_state[w.contractor_id][w.name]}, " \
                    f"while being allocated {w.count}"
                contractors_state[w.contractor_id][w.name] -= w.count
        elif event_type == "end":
            for w in work.workers:
                max_workers_count = initial_contractors_state[w.contractor_id][w.name]
                assert contractors_state[w.contractor_id][w.name] + w.count <= max_workers_count, \
                    f"Excessive workers appear for contractor '{w.contractor_id}' and worker type '{w.name}':" \
                    f"available {contractors_state[w.contractor_id][w.name]}, being returned {w.count}, " \
                    f"initial contractor capacity {initial_contractors_state[w.contractor_id][w.name]}"
                contractors_state[w.contractor_id][w.name] += w.count
        else:
            raise ValueError(f"Incorrect event type: {event_type}. Only 'start' and 'end' are supported.")
