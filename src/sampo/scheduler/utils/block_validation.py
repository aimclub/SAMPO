from copy import deepcopy
from typing import Iterable

from sampo.scheduler.utils.block_graph import BlockGraph
from sampo.scheduler.utils.multi_agency import ScheduledBlock
from sampo.schemas.contractor import Contractor
from sampo.utilities.validation import validate_schedule, \
    check_all_allocated_workers_do_not_exceed_capacity_of_contractors


def validate_block_schedule(bg: BlockGraph, schedule: dict[str, ScheduledBlock]):
    _check_block_dependencies(bg, schedule)
    _check_blocks_separately(schedule.values())

    contractors = set([contractor for sblock in schedule.values() for contractor in sblock.agent.contractors])
    _check_blocks_with_global_timelines(schedule.values(), contractors)


def _check_block_dependencies(bg: BlockGraph, schedule: dict[str, ScheduledBlock]):
    for sblock in schedule.values():
        for p in bg[sblock.id].blocks_from:
            parent_sblock = schedule[p.id]
            assert parent_sblock.start_time <= parent_sblock.end_time <= sblock.start_time <= sblock.end_time


def _check_blocks_separately(sblocks: Iterable[ScheduledBlock]):
    for sblock in sblocks:
        validate_schedule(sblock.schedule, sblock.wg, sblock.agent.contractors)


def _check_blocks_with_global_timelines(sblocks: Iterable[ScheduledBlock], contractors: Iterable[Contractor]):
    """
    Checks that no agent's contractor uses more resources that can supply.

    Note that this should fail if there is a shared contractor between agents, but this
    term is, of course, unsupported.

    :param sblocks: scheduled blocks of works
    :param contractors: global scope of contractors(collected from all agents used to construct sblocks)
    """
    initial_worker_pool: dict[str, dict[str, int]] = {}
    for contractor in contractors:
        initial_worker_pool[contractor.id] = {}
        for w in contractor.workers.values():
            if w.name in initial_worker_pool[contractor.id].keys():
                initial_worker_pool[contractor.id][w.name] += w.count
            else:
                initial_worker_pool[contractor.id][w.name] = w.count
    cur_worker_pool = deepcopy(initial_worker_pool)

    for sblock in sblocks:
        check_all_allocated_workers_do_not_exceed_capacity_of_contractors(sblock.schedule,
                                                                          initial_worker_pool,
                                                                          cur_worker_pool)
