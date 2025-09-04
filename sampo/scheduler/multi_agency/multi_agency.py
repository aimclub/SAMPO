"""Multi-agent scheduling framework with agents, managers and auctions.

Многоагентный механизм планирования с агентами, менеджерами и аукционами.
"""

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
from sampo.schemas import WorkerProductivityMode
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph
from sampo.schemas.schedule import Schedule
from sampo.schemas.time import Time


class Agent:
    """Represents an agent capable of bidding on blocks.

    Представляет агента, способного делать ставки на блоки.
    """

    def __init__(self, name: str, scheduler: Scheduler, contractors: list[Contractor]):
        """Initialize agent with scheduler and contractors.

        Инициализировать агента планировщиком и подрядчиками.

        Args:
            name: Agent name.
                Имя агента.
            scheduler: Scheduling algorithm used by the agent.
                Алгоритм планирования, используемый агентом.
            contractors: Available contractors for the agent.
                Подрядчики, доступные агенту.
        """

        self.name = name
        self._timeline = None
        self._scheduler = scheduler
        self._contractors = contractors
        self._last_task_executed = Time(0)
        self._downtime = Time(0)

    def offer(self, wg: WorkGraph, parent_time: Time) -> tuple[Time, Time, Schedule, Timeline]:
        """Compute an offer for a work graph.

        Вычислить предложение для графа работ.

        Args:
            wg: Block of tasks to schedule.
                Блок задач для планирования.
            parent_time: Maximum end time of parent blocks.
                Максимальное время окончания родительских блоков.

        Returns:
            tuple[Time, Time, Schedule, Timeline]: Start time, end time,
                resulting schedule and timeline before offering.
                Начальное время, конечное время, полученный график и
                временная шкала до предложения.

        To apply returned offer, use :meth:`Agent.confirm`.

        Чтобы применить предложение, используйте :meth:`Agent.confirm`.
        """
        schedule, start_time, timeline, _ = \
            self._scheduler.schedule_with_cache(wg, self._contractors,
                                                assigned_parent_time=parent_time, timeline=deepcopy(self._timeline))[0]
        return start_time, schedule.execution_time, schedule, timeline

    def confirm(self, timeline: Timeline, start: Time, end: Time):
        """Apply the given offer.

        Применить указанное предложение.

        Args:
            timeline: Timeline returned by :meth:`Agent.offer`.
                Временная шкала, возвращённая :meth:`Agent.offer`.
            start: Global start time of the block.
                Глобальное время начала блока.
            end: Global end time of the block.
                Глобальное время окончания блока.
        """
        self._timeline = timeline
        self.update_stat(start)
        # update last task statistic
        self._last_task_executed = end

    def update_stat(self, start: Time):
        """Count downtime before the given start time.

        Подсчитать время простоя до указанного времени начала.

        Args:
            start: Global start time of confirmed block.
                Глобальное время начала подтверждённого блока.
        """
        self._downtime += max(Time(0), start - self._last_task_executed)

    def __str__(self) -> str:
        return f'Agent(name={self.name}, scheduler={self._scheduler}, downtime={self._downtime})'

    def __repr__(self) -> str:
        return str(self)

    @property
    def downtime(self) -> Time:
        """Return accumulated downtime.

        Вернуть накопленное время простоя.
        """

        return self._downtime

    @property
    def contractors(self) -> list[Contractor]:
        """Return available contractors.

        Вернуть доступных подрядчиков.
        """

        return self._contractors

    @property
    def end_time(self) -> Time:
        """Return end time of last executed task.

        Вернуть время окончания последней выполненной задачи.
        """

        return self._last_task_executed

    @property
    def scheduler(self) -> Scheduler:
        """Return scheduler used by the agent.

        Вернуть планировщик, используемый агентом.
        """

        return self._scheduler


@dataclass
class ScheduledBlock:
    """Result of scheduling a block of works.

    Результат планирования блока работ.

    Attributes:
        wg: Scheduled work graph.
            Запланированный граф работ.
        schedule: Schedule produced for the block.
            График, построенный для блока.
        agent: Agent that scheduled the block.
            Агент, который запланировал блок.
        start_time: Global start time of the block.
            Глобальное время начала блока.
        end_time: Global end time of the block.
            Глобальное время окончания блока.
    """
    wg: WorkGraph
    schedule: Schedule
    agent: Agent
    start_time: Time
    end_time: Time

    @property
    def id(self) -> str:
        """Return identifier of underlying work graph.

        Вернуть идентификатор базового графа работ.
        """

        return self.wg.start.id

    @property
    def duration(self) -> Time:
        """Return block duration.

        Вернуть длительность блока.
        """

        return self.end_time - self.start_time

    def __str__(self) -> str:
        return f'ScheduledBlock(start_time={self.start_time}, end_time={self.end_time}, agent={self.agent})'

    def __repr__(self) -> str:
        return str(self)


class Manager:
    """Manager that orchestrates agents.

    Менеджер, который координирует агентов.
    """

    def __init__(self, agents: list[Agent]):
        """Initialize manager with agents.

        Инициализировать менеджера агентами.

        Args:
            agents: Agents each with its scheduler and contractors.
                Агенты, каждый со своим планировщиком и подрядчиками.
        """

        if len(agents) == 0:
            raise NoSufficientAgents('Manager can not work with empty list of agents')
        self._agents = agents

    # TODO Upgrade to supply the best parallelism
    def manage_blocks(self, bg: BlockGraph, logger: Callable[[str], None] = None) -> dict[str, ScheduledBlock]:
        """Run auction-based scheduling on a block graph.

        Запустить аукционное планирование на графе блоков.

        Args:
            bg: Block graph to schedule.
                Граф блоков для планирования.
            logger: Optional logging function.
                Необязательная функция логирования.

        Returns:
            dict[str, ScheduledBlock]: Mapping from work graph id to scheduled block.
                Отображение от идентификатора графа работ к запланированному блоку.
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
        """Run auction on a work graph after applying obstruction.

        Запустить аукцион на графе работ после применения препятствия.

        Args:
            wg: Target work graph.
                Целевой граф работ.
            parent_time: Maximum parent time of the block.
                Максимальное время окончания родительского блока.
            obstruction: Optional obstruction to apply.
                Необязательное препятствие для применения.

        Returns:
            tuple[Time, Time, Schedule, Agent]: Auction results from
                :meth:`run_auction`.
                Результаты аукциона из :meth:`run_auction`.
        """

        if obstruction:
            obstruction.generate(wg)
        return self.run_auction(wg, parent_time)

    def run_auction(self, wg: WorkGraph, parent_time: Time = Time(0)) -> tuple[Time, Time, Schedule, Agent]:
        """Run an auction on the given work graph.

        Запустить аукцион на заданном графе работ.

        Args:
            wg: Target work graph.
                Целевой граф работ.
            parent_time: Maximum parent time of the block.
                Максимальное время окончания родительского блока.

        Returns:
            tuple[Time, Time, Schedule, Agent]: Best start time, end time, schedule
                and winning agent.
                Лучшее время начала, время окончания, график и победивший агент.
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
        # To simulate stochastic behaviour
        best_start_time, best_end_time, best_schedule, best_timeline = best_agent.offer(wg, parent_time)
        best_agent.confirm(best_timeline, best_start_time, best_end_time)
        for agent in self._agents:
            if agent.name != best_agent.name:
                agent.update_stat(best_start_time)

        # print(best_agent.name)
        return best_start_time, best_end_time, best_schedule, best_agent


class StochasticManager(Manager):
    """Manager using confidence levels to adjust agent offers.

    Менеджер, использующий уровни доверия для коррекции предложений агентов.
    """

    def __init__(self, agents: list[Agent]):
        super().__init__(agents)
        self._confidence = {agent.name: 1 for agent in agents}

    def run_auction(self, wg: WorkGraph, parent_time: Time = Time(0)) -> tuple[Time, Time, Schedule, Agent]:
        """Run auction with stochastic adjustment.

        Запустить аукцион со стохастической корректировкой.

        Args:
            wg: Target work graph.
                Целевой граф работ.
            parent_time: Maximum parent time of the block.
                Максимальное время окончания родительского блока.

        Returns:
            tuple[Time, Time, Schedule, Agent]: Best start time, end time, schedule
                and winning agent.
                Лучшее время начала, время окончания, график и победивший агент.
        """
        best_end_time = Time.inf()
        best_agent = None

        def get_offer(agent: Agent):
            agent._scheduler.work_estimator.set_productivity_mode(WorkerProductivityMode.Static)
            return agent, agent.offer(wg, parent_time)

        offers = [get_offer(agent) for agent in self._agents]

        for offered_agent, (offered_start_time, offered_end_time, _, _) in offers:
            offered_end_time = offered_start_time + (offered_end_time - offered_start_time) * self._confidence[offered_agent.name]
            if offered_end_time < best_end_time:
                best_end_time = offered_end_time
                best_agent = offered_agent

        best_agent._scheduler.work_estimator.set_productivity_mode(WorkerProductivityMode.Stochastic)

        old_best_end_time = best_end_time
        best_start_time, best_end_time, best_schedule, best_timeline = best_agent.offer(wg, parent_time)
        modified_end_time = best_start_time + (best_end_time - best_start_time) * self._confidence[
            best_agent.name]
        best_agent.confirm(best_timeline, best_start_time, best_end_time)

        if modified_end_time > old_best_end_time:
            # agent supplied worse time than predicted, lower its confidence
            self._confidence[best_agent.name] += 0.1

        for agent in self._agents:
            if agent.name != best_agent.name:
                agent.update_stat(best_start_time)

        return best_start_time, best_end_time, best_schedule, best_agent


class NeuralManager:
    """Manager that selects agents using neural networks.

    Менеджер, выбирающий агентов с помощью нейронных сетей.

    Args:
        agents: Agents each with its scheduler and contractors.
            Агенты, каждый со своим планировщиком и подрядчиками.
        algo_trainer: Neural network predicting the best scheduling algorithm.
            Нейросеть, предсказывающая наилучший алгоритм планирования.
        contractor_trainer: Neural network predicting contractor resources.
            Нейросеть, предсказывающая ресурсы подрядчиков.
        algorithms: List of unique schedulers used by agents.
            Список уникальных планировщиков, используемых агентами.
        blocks: Blocks of the input block graph in topological order.
            Блоки входного графа блоков в топологическом порядке.
        encoding_blocks: Embeddings of graph blocks.
            Векторные представления блоков графа.
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
        """Run auction-based scheduling using neural predictions.

        Запустить аукционное планирование с использованием нейронных предсказаний.

        Args:
            logger: Optional logging function.
                Необязательная функция логирования.

        Returns:
            dict[str, ScheduledBlock]: Mapping from work graph id to scheduled block.
                Отображение от идентификатора графа работ к запланированному блоку.
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
        """Run auction with neural guidance after applying obstruction.

        Запустить аукцион с нейронным выбором после применения препятствия.

        Args:
            wg: Target work graph.
                Целевой граф работ.
            index: Index of block in the block list.
                Индекс блока в списке блоков.
            parent_time: Maximum parent time of the block.
                Максимальное время окончания родительского блока.
            obstruction: Optional obstruction to apply.
                Необязательное препятствие для применения.

        Returns:
            tuple[Time, Time, Schedule, Agent]: Auction results from
                :meth:`run_auction`.
                Результаты аукциона из :meth:`run_auction`.
        """

        if obstruction:
            obstruction.generate(wg)
        return self.run_auction(wg, index, parent_time)

    def run_auction(self, wg: WorkGraph, index: int, parent_time: Time = Time(0)) -> tuple[Time, Time, Schedule, Agent]:
        """Run auction selecting agent via neural networks.

        Запустить аукцион, выбирая агента с помощью нейронных сетей.

        Args:
            index: Index of the block among the list of blocks.
                Индекс блока среди списка блоков.
            wg: Target work graph.
                Целевой граф работ.
            parent_time: Maximum parent time of the block.
                Максимальное время окончания родительского блока.

        Returns:
            tuple[Time, Time, Schedule, Agent]: Best start time, end time, schedule
                and winning agent.
                Лучшее время начала, время окончания, график и победивший агент.
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
