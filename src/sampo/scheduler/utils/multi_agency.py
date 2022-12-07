from sampo.scheduler.base import Scheduler
from sampo.scheduler.timeline.base import Timeline
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph
from sampo.schemas.time import Time


class Agent:

    def __init__(self, name: str, scheduler: Scheduler, contractors: list[Contractor]):
        self.name = name
        self._timeline = None
        self._scheduler = scheduler
        self._contractors = contractors

    def offer(self, wg: WorkGraph) -> tuple[Time, Timeline]:
        """
        Computes the offer from agent to manager. Handles all works from given wg.

        :param wg: the given block of tasks
        :return: offered working time and timeline before offering

        To apply returned offer, use `Agent#confirm`.
        """
        schedule, timeline = self._scheduler.schedule_with_cache(wg, self._contractors, timeline=self._timeline)
        return schedule.execution_time, timeline

    def confirm(self, timeline: Timeline):
        """
        Applies the given offer.

        :param timeline: timeline returned from corresponding `Agent#offer`
        """
        self._timeline = timeline


class Manager:

    def __init__(self, agents: list[Agent]):
        if len(agents) == 0:
            raise Exception("Manager can't work with empty list of agents")
        self._agents = agents

    # TODO Decide about this
    # def manage_sequense(self):

    def run_auction(self, wg: WorkGraph) -> tuple[Time, Agent]:
        """
        Runs the auction on the given `WorkGraph`.

        :param wg: target `WorkGraph`
        :return: best working time and the agent that is able to support this time
        """
        best_time = Time.inf()
        best_timeline = None
        best_agent = None

        offers = [(agent, agent.offer(wg)) for agent in self._agents]

        for offered_agent, (offered_time, offered_timeline) in offers:
            if offered_time < best_time:
                best_time = offered_time
                best_timeline = offered_timeline
                best_agent = offered_agent
        best_agent.confirm(best_timeline)

        return best_time, best_agent
