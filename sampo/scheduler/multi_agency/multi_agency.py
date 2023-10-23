from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from sampo.scheduler.base import Scheduler
from sampo.scheduler.generic import GenericScheduler
from sampo.scheduler.multi_agency.block_graph import BlockGraph, BlockNode
from sampo.scheduler.multi_agency.exception import NoSufficientAgents
from sampo.scheduler.selection.neural_net import NeuralNetTrainer
from sampo.scheduler.timeline.base import Timeline
from sampo.scheduler.utils.obstruction import Obstruction
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph
from sampo.schemas.schedule import Schedule
from sampo.schemas.time import Time


class Agent:
    """
    Agent entity representation in the multi-agent model
    Agent have 2 actions: give offer and accept offer
    """

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

        schedule, start_time, timeline, _ = \
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
        """
        Count last iteration downtime.
        If given start time is lower, than the last executed task, then this downtime are already in self_downtime

        :param start: global start time of confirmed block
        """
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

    @property
    def end_time(self) -> Time:
        return self._last_task_executed

    @property
    def scheduler(self):
        return self._scheduler


@dataclass
class ScheduledBlock:
    """
    An object represents a scheduled graph block(group of works).

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
    """
    Manager entity representation in the multi-agent model
    Manager interact with agents

    :param agents: list of agents that has own scheduling algorithm and set of contractors
    """
    def __init__(self, agents: list[Agent]):
        if len(agents) == 0:
            raise NoSufficientAgents('Manager can not work with empty list of agents')
        self._agents = agents

    # TODO Upgrade to supply the best parallelism
    def manage_blocks(self, bg: BlockGraph, logger: Callable[[str], None] = None) -> dict[str, ScheduledBlock]:
        """
        Runs the multi-agent system based on auction on given BlockGraph.
        
        :param bg: 
        :param logger:
        :return: an index of resulting `ScheduledBlock`s built by ids of corresponding `WorkGraph`s
        """
        id2sblock = {}
        for i, block in enumerate(bg.toposort()):
            max_parent_time = max((id2sblock[parent.id].end_time for parent in block.blocks_from), default=Time(0)) + 1
            start_time, end_time, agent_schedule, agent \
                = self.run_auction_with_obstructions(block.wg, max_parent_time, block.obstruction)

            assert start_time >= max_parent_time, f'Scheduler {agent._scheduler} does not handle parent_time!'

            if logger and not block.is_service():
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

        print(best_agent.name)
        return best_start_time, best_end_time, best_schedule, best_agent


class NeuralManager:
    """
    Manager entity representation in the multi-agent model, that uses neural networks
    Neural manager interact with agents
    Neural manager uses neural network as the method of the most suitable agent for each work graph

    :param agents: list of agents that has own scheduling algorithm and set of contractors
    :param algo_trainer: work with a neural network that predicts the most suitable scheduling algorithm
    :param contractor_trainer: work with a neural network that predicts a set of resources
    that the most suitable contractor should have
    :param algorithms: list of unique scheduling algorithms that agents have
    :param blocks: list of blocks of input Block Graph in topological order
    :param encoding_blocks: list of graph block embeddings
    """

    def __init__(self, agents: list[Agent],
                 algo_trainer: NeuralNetTrainer,
                 contractor_trainer: NeuralNetTrainer,
                 algorithms: list[GenericScheduler],
                 blocks: list[BlockNode],
                 encoding_blocks: list[torch.Tensor]):
        if len(agents) == 0:
            raise NoSufficientAgents('Manager can not work with empty list of agents')
        self._agents = agents
        self.algo_trainer = algo_trainer
        self.contractor_trainer = contractor_trainer
        self.algorithms = algorithms
        self.blocks = blocks
        self.encoding_blocks = encoding_blocks

    # TODO Upgrade to supply the best parallelism
    def manage_blocks(self, logger: Callable[[str], None] = None) -> dict[str, ScheduledBlock]:
        """
        Runs the multi-agent system based on auction on given BlockGraph.

        :param logger:
        :return: an index of resulting `ScheduledBlock`s built by ids of corresponding `WorkGraph`s
        """
        id2sblock = {}
        for i, block in enumerate(self.blocks):
            max_parent_time = max((id2sblock[parent.id].end_time for parent in block.blocks_from), default=Time(0)) + 1
            start_time, end_time, agent_schedule, agent \
                = self.run_auction_with_obstructions(block.wg, i, max_parent_time, block.obstruction)

            assert start_time >= max_parent_time, f'Scheduler {agent._scheduler} does not handle parent_time!'

            if logger and not block.is_service():
                logger(f'{agent._scheduler}')
            sblock = ScheduledBlock(wg=block.wg, agent=agent, schedule=agent_schedule,
                                    start_time=start_time,
                                    end_time=end_time)
            id2sblock[sblock.id] = sblock

        return id2sblock

    def run_auction_with_obstructions(self, wg: WorkGraph,
                                      index: int,
                                      parent_time: Time = Time(0),
                                      obstruction: Obstruction | None = None):
        if obstruction:
            obstruction.generate(wg)
        return self.run_auction(wg, index, parent_time)

    def run_auction(self, wg: WorkGraph, index: int, parent_time: Time = Time(0)) -> (Time, Time, Schedule, Agent):
        """
        Runs the auction on the given `WorkGraph`.

        :param index: index of agent from the list of agents
        :param wg: target `WorkGraph`
        :param parent_time: max parent time of given block
        :return: best start time, end time and the agent that is able to support this working time
        """

        wg_encoding = [self.encoding_blocks[index]]
        predicted = self.algo_trainer.predict(wg_encoding)
        predict_proba = self.algo_trainer.predict_proba(wg_encoding)
        best_algo = type(self.algorithms[int(predicted)])
        best_contractor = np.asarray(self.contractor_trainer.predict([self.encoding_blocks[index]]))[0]

        time_algo_agents = []
        not_time_algo_agents = []
        time_not_algo_agents = []
        not_time_not_algo_agents = []

        for agent in self._agents:
            if parent_time > agent.end_time and isinstance(agent.scheduler, best_algo):
                time_algo_agents.append(agent)
            elif not isinstance(agent.scheduler, best_algo) and parent_time > agent.end_time:
                time_not_algo_agents.append(agent)
            elif parent_time <= agent.end_time and isinstance(agent.scheduler, best_algo):
                not_time_algo_agents.append(agent)
            else:
                not_time_not_algo_agents.append(agent)

        def auction(agents: list[Agent]) -> Agent:
            less_mse = 10**9
            best_agent = None

            for agent in agents:
                resources = []
                for worker in agent.contractors[0].workers.values():
                    resources.append(worker.count)
                resources = np.asarray(resources)
                mse = sum((resources - best_contractor) ** 2)
                if less_mse > mse:
                    less_mse = mse
                    best_agent = agent

            return best_agent

        if len(time_algo_agents) > 0:
            best_agent = auction(time_algo_agents)
        elif len(time_not_algo_agents) > 0:
            best_agent = auction(time_not_algo_agents)
        elif len(not_time_algo_agents) > 0:
            best_agent = auction(not_time_algo_agents)
        else:
            best_agent = auction(not_time_not_algo_agents)

        best_start_time, best_end_time, best_schedule, best_timeline = best_agent.offer(wg, parent_time)

        best_agent.confirm(best_timeline, best_start_time, best_end_time)
        for agent in self._agents:
            if agent.name != best_agent.name:
                agent.update_stat(best_start_time)

        print(best_agent.name)
        return best_start_time, best_end_time, best_schedule, best_agent
