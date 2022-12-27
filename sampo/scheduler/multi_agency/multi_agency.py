from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Callable

from sampo.scheduler.base import Scheduler
from sampo.scheduler.multi_agency.block_graph import BlockGraph
from sampo.scheduler.timeline.base import Timeline
from sampo.scheduler.utils.obstruction import Obstruction
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph
from sampo.schemas.schedule import Schedule
from sampo.schemas.time import Time


class Agent:

    def __init__(self, name: str, scheduler: Scheduler, contractors: list[Contractor]):
        self.name = name
        self._timeline = None
        self._scheduler = scheduler
        self._contractors = contractors
        self._last_task_executed = Time(0)
        self._downtime = Time(0)

    def offer(self, wg: WorkGraph, parent_time: Time) -> tuple[Time, Time, Schedule, Timeline]:
        """
        Computes the offer from agent to manager. Handles all works from given wg.

        :param wg: the given block of tasks
        :param parent_time: max end time of parent blocks
        :return: offered start time, end time, resulting schedule and timeline before offering

        To apply returned offer, use `Agent#confirm`.
        """
        schedule, start_time, timeline = \
            self._scheduler.schedule_with_cache(wg, self._contractors,
                                                assigned_parent_time=parent_time, timeline=deepcopy(self._timeline))
        return start_time, schedule.execution_time, schedule, timeline

    def confirm(self, timeline: Timeline, start: Time, end: Time):
        """
        Applies the given offer.

        :param timeline: timeline returned from corresponding `Agent#offer`
        :param start: global start time of confirmed block
        :param end: global end time of confirmed block
        """
        self._timeline = timeline
        self.update_stat(start)
        # update last task statistic
        self._last_task_executed = end

    def update_stat(self, start: Time):
        # count last iteration downtime
        # if given start time is lower than last executed task
        #   then this downtime are already in self_downtime
        self._downtime += max(Time(0), start - self._last_task_executed)

    def __str__(self):
        return f'Agent(name={self.name}, scheduler={self._scheduler}, downtime={self._downtime})'

    def __repr__(self):
        return str(self)

    @property
    def downtime(self) -> Time:
        return self._downtime

    @property
    def contractors(self):
        return self._contractors


@dataclass
class ScheduledBlock:
    """
    An object represents scheduled graph block(group of works).

    Contains all data used in scheduling, the agent and resulting information.
    """
    wg: WorkGraph
    schedule: Schedule
    agent: Agent
    start_time: Time
    end_time: Time

    @property
    def id(self):
        return self.wg.start.id

    @property
    def duration(self):
        return self.end_time - self.start_time

    def __str__(self):
        return f'ScheduledBlock(start_time={self.start_time}, end_time={self.end_time}, agent={self.agent})'

    def __repr__(self):
        return str(self)


class Manager:
    def __init__(self, agents: list[Agent]):
        if len(agents) == 0:
            raise Exception("Manager can't work with empty list of agents")
        self._agents = agents

    # TODO Upgrade to supply the best parallelism
    def manage_blocks(self, bg: BlockGraph, logger: Callable[[str], None] = None) -> Dict[str, ScheduledBlock]:
        """
        Runs multi-agent system based on auction on given BlockGraph.
        
        :param bg: 
        :param logger:
        :return: an index of resulting `ScheduledBlock`s built by ids of corresponding `WorkGraph`s
        """
        id2sblock = {}
        for i, block in enumerate(bg.toposort()):
            max_parent_time = max((id2sblock[parent.id].end_time for parent in block.blocks_from), default=Time(0))
            start_time, end_time, agent_schedule, agent \
                = self.run_auction_with_obstructions(block.wg, max_parent_time, block.obstruction)

            assert start_time >= max_parent_time, f'Scheduler {agent._scheduler} does not handle parent_time!'

            if logger:
                logger(f'{agent._scheduler}')
            sblock = ScheduledBlock(wg=block.wg, agent=agent, schedule=agent_schedule,
                                    start_time=start_time,
                                    end_time=end_time)
            id2sblock[sblock.id] = sblock

        return id2sblock

    def run_auction_with_obstructions(self, wg: WorkGraph, parent_time: Time = Time(0),
                                      obstruction: Obstruction | None = None):
        if obstruction:
            obstruction.generate(wg)
        return self.run_auction(wg, parent_time)

    def run_auction(self, wg: WorkGraph, parent_time: Time = Time(0)) -> (Time, Time, Schedule, Agent):
        """
        Runs the auction on the given `WorkGraph`.

        :param wg: target `WorkGraph`
        :param parent_time: max parent time of given block
        :return: best start time, end time and the agent that is able to support this working time
        """
        best_start_time = 0
        best_end_time = Time.inf()
        best_schedule = None
        best_timeline = None
        best_agent = None

        offers = [(agent, agent.offer(wg, parent_time)) for agent in self._agents]

        for offered_agent, (offered_start_time, offered_end_time, offered_schedule, offered_timeline) in offers:
            if offered_end_time < best_end_time:
                best_start_time = offered_start_time
                best_end_time = offered_end_time
                best_schedule = offered_schedule
                best_timeline = offered_timeline
                best_agent = offered_agent
        best_agent.confirm(best_timeline, best_start_time, best_end_time)
        for agent in self._agents:
            if agent.name != best_agent.name:
                agent.update_stat(best_start_time)

        return best_start_time, best_end_time, best_schedule, best_agent
