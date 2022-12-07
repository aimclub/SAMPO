from dataclasses import dataclass

from sampo.scheduler.base import Scheduler
from sampo.scheduler.timeline.base import Timeline
from sampo.scheduler.utils.block_graph import BlockGraph
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph
from sampo.schemas.time import Time


class Agent:

    def __init__(self, name: str, scheduler: Scheduler, contractors: list[Contractor]):
        self.name = name
        self._timeline = None
        self._scheduler = scheduler
        self._contractors = contractors

    def offer(self, wg: WorkGraph) -> tuple[Time, Time, Timeline]:
        """
        Computes the offer from agent to manager. Handles all works from given wg.

        :param wg: the given block of tasks
        :return: offered start time, end time and timeline before offering

        To apply returned offer, use `Agent#confirm`.
        """
        schedule, start_time, timeline = \
            self._scheduler.schedule_with_cache(wg, self._contractors, timeline=self._timeline)
        return start_time, start_time + schedule.execution_time, timeline

    def confirm(self, timeline: Timeline):
        """
        Applies the given offer.

        :param timeline: timeline returned from corresponding `Agent#offer`
        """
        self._timeline = timeline


@dataclass
class ScheduledBlock:
    wg: WorkGraph
    agent: Agent
    start_time: Time
    end_time: Time

    @property
    def id(self):
        return self.wg.start.id

    @property
    def duration(self):
        return self.end_time - self.start_time


class Manager:
    def __init__(self, agents: list[Agent]):
        if len(agents) == 0:
            raise Exception("Manager can't work with empty list of agents")
        self._agents = agents

    # TODO Upgrade to supply the best parallelism
    def manage_blocks(self, bg: BlockGraph) -> dict[str, ScheduledBlock]:
        id2sblock = {}
        for block in bg.nodes:
            agent_start_time, agent_end_time, agent = self.run_auction(block.wg)
            max_parent_time = max((id2sblock[parent.id].end_time for parent in block.blocks_from), default=Time(0))
            start_time = max(max_parent_time, agent_start_time)
            delta = start_time - agent_start_time
            sblock = ScheduledBlock(wg=block.wg, agent=agent,
                                    start_time=agent_start_time + delta,
                                    end_time=agent_end_time + delta)
            id2sblock[sblock.id] = sblock

        return id2sblock

    def run_auction(self, wg: WorkGraph) -> tuple[Time, Time, Agent]:
        """
        Runs the auction on the given `WorkGraph`.

        :param wg: target `WorkGraph`
        :return: best start time, end time and the agent that is able to support this working time
        """
        best_start_time = 0
        best_end_time = Time.inf()
        best_timeline = None
        best_agent = None

        offers = [(agent, agent.offer(wg)) for agent in self._agents]

        for offered_agent, (offered_start_time, offered_end_time, offered_timeline) in offers:
            if offered_end_time - offered_start_time < best_end_time - best_start_time:
                best_start_time = offered_start_time
                best_end_time = offered_end_time
                best_timeline = offered_timeline
                best_agent = offered_agent
        best_agent.confirm(best_timeline)

        return best_start_time, best_end_time, best_agent
