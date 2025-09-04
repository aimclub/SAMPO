"""Local optimization helpers for scheduling.

Инструменты локальной оптимизации для планирования.
"""

from abc import ABC, abstractmethod
from operator import attrgetter
from typing import Iterable

from sampo.scheduler.timeline.base import Timeline
from sampo.scheduler.utils import WorkerContractorPool
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import GraphNode
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.requirements import WorkerReq
from sampo.schemas.resources import Worker
from sampo.schemas.schedule import ScheduledWork
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.utilities.collections_util import build_index

PRIORITY_SHUFFLE_RADIUS = 0.5


class OrderLocalOptimizer(ABC):
    """Base interface for node order optimizers.

    Базовый интерфейс оптимизаторов порядка узлов.
    """

    @abstractmethod
    def optimize(self, node_order: list[GraphNode], area: range) -> list[GraphNode]:
        """Reorder nodes within given area.

        Изменяет порядок узлов в указанной области.

        Args:
            node_order: Sequence of nodes.
                Последовательность узлов.
            area: Range to optimize.
                Диапазон для оптимизации.

        Returns:
            Optimized node order.
            Оптимизированный порядок узлов.
        """
        ...


class ScheduleLocalOptimizer(ABC):
    """Base class for schedule-level local optimization.

    Базовый класс для локальной оптимизации расписаний.
    """

    def __init__(self, timeline_type: type[Timeline]):
        """Store timeline type for recalculation.

        Сохраняет тип временной шкалы для перерасчёта.

        Args:
            timeline_type: Timeline class used for rescheduling.
                Класс временной шкалы, используемый для пересчёта.
        """
        self._timeline_type = timeline_type

    @abstractmethod
    def optimize(self, scheduled_works: dict[GraphNode, ScheduledWork], node_order: list[GraphNode],
                 contractors: list[Contractor], landscape_config: LandscapeConfiguration,
                 spec: ScheduleSpec, worker_pool: WorkerContractorPool,
                 work_estimator: WorkTimeEstimator, assigned_parent_time: Time,
                 area: range) -> dict[GraphNode, ScheduledWork]:
        """Optimize subset of scheduled works.

        Оптимизирует подмножество запланированных работ.

        Args:
            scheduled_works: Mapping of nodes to schedules.
                Отображение узлов к расписаниям.
            node_order: Order of nodes.
                Порядок узлов.
            contractors: Available contractors.
                Доступные подрядчики.
            landscape_config: Landscape configuration.
                Конфигурация местности.
            spec: Schedule specification.
                Спецификация расписания.
            worker_pool: Pool of workers by contractor.
                Пул работников по подрядчикам.
            work_estimator: Work time estimator.
                Оценщик времени работы.
            assigned_parent_time: Start time of parent context.
                Время начала родительского контекста.
            area: Range to optimize.
                Диапазон для оптимизации.

        Returns:
            Updated mapping of scheduled works.
            Обновлённое отображение запланированных работ.
        """
        ...


def get_swap_candidates(node: GraphNode,
                        node_index: int,
                        candidates: Iterable[GraphNode],
                        node2ind: dict[GraphNode, int],
                        processed: set[GraphNode]) -> list[GraphNode]:
    """Find nodes swappable with target without breaking order.

    Находит узлы, которые можно обменять с целевым, не нарушая порядок.

    Args:
        node: Target node.
            Целевой узел.
        node_index: Index of target node in sequence.
            Индекс целевого узла в последовательности.
        candidates: Iterable of nodes to try.
            Перечень кандидатов для обмена.
        node2ind: Mapping from node to its index.
            Отображение узла в индекс.
        processed: Set of nodes to skip.
            Множество узлов, которые нужно пропустить.

    Returns:
        List of acceptable swap candidates.
        Список подходящих кандидатов для обмена.
    """
    cur_children: set[GraphNode] = node.children_set

    def is_candidate_accepted(candidate: GraphNode) -> bool:
        if candidate in cur_children or candidate in processed:
            return False
        candidate_ind = node2ind[candidate]
        for child in cur_children:
            if node2ind.get(child, 0) >= candidate_ind:  # we have a child between us and candidate
                return False
        candidate_parents = candidate.parents
        for parent in candidate_parents:
            if node2ind.get(parent, 0) <= node_index:  # candidate has a parent between us and candidate
                return False
        return True

    return [candidate for candidate in candidates if is_candidate_accepted(candidate)]


class SwapOrderLocalOptimizer(OrderLocalOptimizer):
    """Shuffle nodes without violating topological order.

    Переставляет узлы, не нарушая топологического порядка.
    """

    def optimize(self, node_order: list[GraphNode], area: range) -> list[GraphNode]:
        """Swap nodes to slightly change order.

        Меняет местами узлы для небольшого изменения порядка.

        Args:
            node_order: Sequence of nodes.
                Последовательность узлов.
            area: Range to process.
                Диапазон для обработки.

        Returns:
            Modified node order.
            Изменённый порядок узлов.
        """
        if node_order is None:
            return

        start_index = area.start
        end_index = area.stop

        # TODO Examine what is better: perform shuffling in nearly placed sub-seq or in whole sequence
        # node2cost = {node: work_priority(node, calculate_working_time_cascade, work_estimator) for node in sub_seq}

        # preprocessing
        node2ind: dict[GraphNode, int] = {node: start_index + ind for ind, node in
                                          enumerate(node_order[start_index:end_index])}

        # temporary for usability measurement
        swapped = 0

        processed: set[GraphNode] = set()
        for i in reversed(area):
            node = node_order[i]
            if node in processed:
                continue
            # cur_cost = node2cost[node]
            chain_candidates: list[GraphNode] = node_order[start_index:i]

            accepted_candidates = get_swap_candidates(node, i, chain_candidates, node2ind, processed)

            if accepted_candidates:
                chain_candidate = accepted_candidates[0]
                swap_idx = node2ind[chain_candidate]
                node_order[i], node_order[swap_idx] = node_order[swap_idx], node_order[i]
                # print(f'Swapped {i} and {swap_idx}')
                processed.add(chain_candidate)
                node2ind[chain_candidate] = i
                node2ind[node] = swap_idx
                swapped += 1

            processed.add(node)
        print(f'Swapped {swapped} times!')

        return node_order


class ParallelizeScheduleLocalOptimizer(ScheduleLocalOptimizer):
    """Make nearby works execute in parallel.

    Заставляет близкие работы выполняться параллельно.
    """

    def __init__(self, timeline_type: type[Timeline]):
        """Initialize optimizer.

        Инициализирует оптимизатор.

        Args:
            timeline_type: Timeline class for scheduling.
                Класс временной шкалы для планирования.
        """
        super().__init__(timeline_type)

    def recalc_schedule(self,
                        node_order: Iterable[GraphNode],
                        contractors: list[Contractor],
                        landscape_config: LandscapeConfiguration,
                        spec: ScheduleSpec,
                        node2swork: dict[GraphNode, ScheduledWork],
                        worker_pool: WorkerContractorPool,
                        assigned_parent_time: Time,
                        work_estimator: WorkTimeEstimator) -> dict[GraphNode, ScheduledWork]:
        """Recalculate durations and times for sequence.

        Пересчитывает длительности и времена для последовательности.

        Args:
            node_order: Scheduled works to process.
                Запланированные работы для обработки.
            contractors: Available contractors.
                Доступные подрядчики.
            landscape_config: Landscape configuration.
                Конфигурация местности.
            spec: Schedule specification.
                Спецификация расписания.
            node2swork: Mapping of nodes to scheduled works.
                Отображение узлов к запланированным работам.
            worker_pool: Pool of workers.
                Пул работников.
            assigned_parent_time: Start time of parent.
                Время начала родителя.
            work_estimator: Work time estimator.
                Оценщик времени работы.

        Returns:
            Mapping of nodes to recalculated works.
            Отображение узлов к пересчитанным работам.
        """

        timeline = self._timeline_type(worker_pool, landscape_config)
        node2swork_new: dict[GraphNode, ScheduledWork] = {}

        id2contractor = build_index(contractors, attrgetter('name'))

        for index, node in enumerate(node_order):
            node_schedule = node2swork[node]
            work_spec = spec.get_work_spec(node.id)
            # st = timeline.find_min_start_time(node, node_schedule.workers, node2swork_new)
            # ft = st + node_schedule.get_actual_duration(work_estimator)
            timeline.schedule(node, node2swork_new, node_schedule.workers,
                              id2contractor[node_schedule.contractor], work_spec, None, work_spec.assigned_time,
                              assigned_parent_time, work_estimator)
            # node_schedule.start_end_time = (st, ft)
            node2swork_new[node] = node_schedule

        return node2swork_new

    def optimize(self, scheduled_works: dict[GraphNode, ScheduledWork], node_order: list[GraphNode],
                 contractors: list[Contractor], landscape_config: LandscapeConfiguration,
                 spec: ScheduleSpec, worker_pool: WorkerContractorPool,
                 work_estimator: WorkTimeEstimator, assigned_parent_time: Time,
                 area: range) -> dict[GraphNode, ScheduledWork]:
        """Turn nearby works into parallel execution.

        Преобразует близкие работы в параллельное выполнение.

        Args:
            scheduled_works: Mapping of nodes to scheduled works.
                Отображение узлов к запланированным работам.
            node_order: Order of nodes.
                Порядок узлов.
            contractors: Available contractors.
                Доступные подрядчики.
            landscape_config: Landscape configuration.
                Конфигурация местности.
            spec: Schedule specification.
                Спецификация расписания.
            worker_pool: Pool of workers.
                Пул работников.
            work_estimator: Time estimator.
                Оценщик времени.
            assigned_parent_time: Start time of parent context.
                Время начала родительского контекста.
            area: Range of indices to process.
                Диапазон индексов для обработки.

        Returns:
            Mapping of nodes to updated schedules.
            Отображение узлов к обновлённым расписаниям.
        """
        start_index = area.start
        end_index = area.stop

        # preprocessing
        node2ind: dict[GraphNode, int] = {node: start_index + ind for ind, node in
                                          enumerate(node_order[start_index:end_index])}

        processed: set[GraphNode] = set()

        for i in reversed(area):
            node = node_order[i]
            if node in processed:
                continue
            chain_candidates: list[GraphNode] = node_order[start_index:i]
            accepted_candidates = get_swap_candidates(node, i, chain_candidates, node2ind, processed)

            my_schedule: ScheduledWork = scheduled_works[node]
            my_workers: dict[str, Worker] = build_index(my_schedule.workers, attrgetter('name'))
            my_schedule_reqs: dict[str, WorkerReq] = build_index(node.work_unit.worker_reqs, attrgetter('kind'))

            new_my_workers = {}

            # now accepted_candidates is a list of nodes that can(according to dependencies) run in parallel
            for candidate in accepted_candidates:
                candidate_schedule = scheduled_works[candidate]

                candidate_schedule_reqs: dict[str, WorkerReq] = build_index(candidate.work_unit.worker_reqs,
                                                                            attrgetter('kind'))

                new_candidate_workers: dict[str, int] = {}

                satisfy = True

                for candidate_worker in candidate_schedule.workers:
                    my_worker = my_workers.get(candidate_worker.name, None)
                    if my_worker is None:  # these two works are not compete for this worker
                        continue

                    need_me = my_workers[candidate_worker.name].count
                    need_candidate = candidate_worker.count

                    total = need_me + need_candidate
                    my_req = my_schedule_reqs[candidate_worker.name]
                    candidate_req = candidate_schedule_reqs[candidate_worker.name]
                    needed_min = my_req.min_count + candidate_req.min_count

                    if needed_min > total:  # these two works can't run in parallel
                        satisfy = False
                        break

                    candidate_worker_count = candidate_req.min_count
                    my_worker_count = my_req.min_count
                    total -= needed_min

                    add_me = min(my_req.max_count, total // 2)
                    add_candidate = min(candidate_req.max_count, total - add_me)

                    my_worker_count += add_me
                    candidate_worker_count += add_candidate

                    new_my_workers[candidate_worker.name] = my_worker_count
                    new_candidate_workers[candidate_worker.name] = candidate_worker_count

                if satisfy:  # replacement found, apply changes and leave candidates bruteforce
                    print(f'Found! {candidate.work_unit.name} {node.work_unit.name}')
                    for worker in my_schedule.workers:
                        worker_count = new_my_workers.get(worker.name, None)
                        if worker_count is not None:
                            worker.count = worker_count
                    for worker in candidate_schedule.workers:
                        worker_count = new_candidate_workers.get(worker.name, None)
                        if worker_count is not None:
                            worker.count = worker_count
                    # candidate_schedule.start_time = my_schedule.start_time
                    break

        return self.recalc_schedule(reversed(node_order), contractors, landscape_config, spec, scheduled_works,
                                    worker_pool, assigned_parent_time, work_estimator)


def optimize_local_sequence(seq: list[GraphNode],
                            start_ind: int,
                            end_ind: int,
                            work_estimator: WorkTimeEstimator):
    """Experimental local sequence optimizer.

    Экспериментальный оптимизатор локальной последовательности.

    Args:
        seq: Sequence of nodes.
            Последовательность узлов.
        start_ind: Start index.
            Начальный индекс.
        end_ind: End index.
            Конечный индекс.
        work_estimator: Work time estimator.
            Оценщик времени работы.

    TODO: Try to find sets of works with similar resources and parallelize.
    TODO: Попробовать находить работы с похожими ресурсами для параллелизации.
    """
    pass
