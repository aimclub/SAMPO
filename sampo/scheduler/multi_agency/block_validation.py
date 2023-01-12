from copy import deepcopy
from typing import Iterable

from sampo.scheduler.multi_agency.block_graph import BlockGraph
from sampo.scheduler.multi_agency.multi_agency import ScheduledBlock, Agent
from sampo.schemas.contractor import Contractor
from sampo.utilities.validation import validate_schedule, \
    check_all_allocated_workers_do_not_exceed_capacity_of_contractors


def validate_block_schedule(bg: BlockGraph, schedule: dict[str, ScheduledBlock], agents: Iterable[Agent]):
    contractors = [contractor for agent in agents for contractor in agent.contractors]
    contractors_set = set(contractors)

    assert len(contractors) == len(contractors_set), \
        f'There are contractor collisions between agents: ' \
        f'{[c.id for c in contractors]} != {[c.id for c in contractors_set]}'

    _check_block_dependencies(bg, schedule)
    _check_blocks_separately(schedule.values())

    # TODO Fix. To fully validate resources usage in the whole multi-agent appearance,
    #    we should union all agent's ScheduleEvents and go through it in sorted way.
    #    Now we think that this validation phase is not extremely need.
    # _check_blocks_with_global_timelines(schedule.values(), contractors)


def _check_block_dependencies(bg: BlockGraph, schedule: dict[str, ScheduledBlock]):
    for sblock in schedule.values():
        for p in bg[sblock.id].blocks_from:
            parent_sblock = schedule[p.id]
            assert parent_sblock.start_time <= parent_sblock.end_time <= sblock.start_time <= sblock.end_time


def _check_blocks_separately(sblocks: Iterable[ScheduledBlock]):
    for sblock in sblocks:
        try:
            validate_schedule(sblock.schedule, sblock.wg, sblock.agent.contractors)
        except AssertionError as e:
            raise AssertionError(f'Agent {sblock.agent} supplied an invalid schedule', e)


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
