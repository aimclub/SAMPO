"""Structures for representing and manipulating block graphs.

Структуры для представления и обработки графов блоков.
"""

from operator import attrgetter
from typing import Callable

from sampo.scheduler.utils.obstruction import Obstruction
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.works import WorkUnit
from sampo.utilities.collections_util import build_index


class BlockNode:
    """Node of a block graph with its work graph and relations.

    Узел графа блоков с его графом работ и связями.

    Attributes:
        wg: Work graph contained in the node.
            Рабочий граф, содержащийся в узле.
        obstruction: Optional obstruction for the block.
            Необязательное препятствие для блока.
        blocks_from: Predecessor block nodes.
            Узлы-предшественники.
        blocks_to: Successor block nodes.
            Узлы-потомки.
    """

    def __init__(self, wg: WorkGraph, obstruction: Obstruction | None = None):
        """Initialize block node.

        Инициализировать узел блока.

        Args:
            wg: Work graph contained in the node.
                Рабочий граф, содержащийся в узле.
            obstruction: Optional obstruction for the block.
                Необязательное препятствие для блока.
        """

        self.wg = wg
        self.obstruction = obstruction
        self.blocks_from: list[BlockNode] = []
        self.blocks_to: list[BlockNode] = []

    @property
    def id(self) -> str:
        """Return identifier of the node.

        Вернуть идентификатор узла.
        """

        return self.wg.start.id

    def is_service(self) -> bool:
        """Check whether the node represents a service block.

        Проверить, представляет ли узел сервисный блок.
        """

        return self.wg.vertex_count == 2

    def __hash__(self) -> int:
        return hash(self.id)


class BlockGraph:
    """Graph composed of work blocks connected by finish-start edges.

    Граф, составленный из блоков работ, связанных зависимостями типа
    "конец-начало".
    """

    def __init__(self, nodes: list[BlockNode]):
        """Initialize block graph with given nodes.

        Инициализировать граф блоков заданными узлами.

        Args:
            nodes: Nodes forming the graph.
                Узлы, формирующие граф.
        """

        self.nodes = nodes
        self.node_dict = build_index(self.nodes, attrgetter('id'))

    @staticmethod
    def pure(nodes: list[WorkGraph], obstruction_getter: Callable[[int], Obstruction | None] = lambda _: None) -> 'BlockGraph':
        """Build a block graph from work graphs without edges.

        Построить граф блоков из графов работ без рёбер.

        Args:
            nodes: Work graphs to convert into blocks.
                Графы работ для преобразования в блоки.
            obstruction_getter: Function providing an optional obstruction.
                Функция, предоставляющая необязательное препятствие.

        Returns:
            BlockGraph: Block graph without edges.
                Граф блоков без рёбер.
        """
        return BlockGraph([BlockNode(node, obstruction_getter(i)) for i, node in enumerate(nodes)])

    def __getitem__(self, item) -> BlockNode:
        """Return block node by identifier.

        Вернуть узел блока по идентификатору.
        """

        return self.node_dict[item]

    def __len__(self) -> int:
        """Return number of nodes in the graph.

        Вернуть количество узлов в графе.
        """

        return len(self.nodes)

    def __copy__(self) -> 'BlockGraph':
        """Create a shallow copy of the block graph.

        Создать неглубокую копию графа блоков.
        """

        parent_ids = [[parent.id for parent in node.blocks_to] for node in self.nodes]
        nodes = [BlockNode(old_node.wg, old_node.obstruction) for old_node in self.nodes]
        nodes_index = build_index(nodes, attrgetter('id'))

        for node, parents in zip(nodes, parent_ids):
            for parent in parents:
                BlockGraph.add_edge(nodes_index[parent], node)

        return BlockGraph(nodes)

    def to_work_graph(self) -> WorkGraph:
        """Convert block graph into an equivalent work graph.

        Преобразовать граф блоков в эквивалентный граф работ.
        """
        copied_graph = self.__copy__()
        copied_nodes = copied_graph.nodes

        # nodes are copied as follows, since recursion occurs when attempting to copy WorkGraph normally
        for node in copied_nodes:
            node.wg = WorkGraph._deserialize(node.wg._serialize())

        nodes_without_children = []

        global_start = GraphNode(WorkUnit('start', 'start', is_service_unit=True), [])

        for end in copied_nodes:
            end.wg.start.add_parents([start.wg.finish for start in end.blocks_to])
            if len(end.wg.start.parents) == 0:
                end.wg.start.add_parents([global_start])
            if len(end.wg.finish.children) == 0:
                nodes_without_children.append(end.wg.finish)

        global_end = GraphNode(WorkUnit('finish', 'finish', is_service_unit=True), nodes_without_children)

        # global_start: GraphNode = [node for node in copied_nodes if len(node.blocks_to) == 0][0].wg.start
        # global_end:   GraphNode = [node for node in copied_nodes if len(node.blocks_from) == 0][0].wg.finish

        return WorkGraph(global_start, global_end)

    def toposort(self) -> list[BlockNode]:
        """Sort block graph nodes in topological order.

        Отсортировать узлы графа блоков в топологическом порядке.

        Returns:
            list[BlockNode]: Ordered list of block nodes.
                Упорядоченный список узлов блоков.
        """
        visited = set()
        ans = []

        def dfs(u: BlockNode):
            visited.add(u)
            for v in u.blocks_to:
                if v not in visited:
                    dfs(v)
            ans.append(u)

        for node in self.nodes:
            if node not in visited:
                dfs(node)
        ans.reverse()

        return ans

    @staticmethod
    def add_edge(start: BlockNode, end: BlockNode):
        """Add directed edge from ``start`` to ``end``.

        Добавить направленное ребро от ``start`` к ``end``.

        Args:
            start: Source block node.
                Исходный узел блока.
            end: Destination block node.
                Конечный узел блока.
        """

        start.blocks_to.append(end)
        end.blocks_from.append(start)
