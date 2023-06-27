from uuid import uuid4

from _pytest.fixtures import fixture

from sampo.schemas.graph import EdgeType
from sampo.schemas.works import WorkUnit
from sampo.structurator.insert_wu import insert_work_unit


@fixture()
def setup_wu() -> WorkUnit:
    return WorkUnit(str(uuid4()), 'test_wu')


# TODO add checking without id change
# TODO docstring documentation ??
def test_insert_work_unit(setup_wg, setup_wu):
    parents_edges = [(setup_wg.start, -1, EdgeType.FinishStart)]
    children_edges = [(setup_wg.finish, -1, EdgeType.FinishStart)]

    wg_with_inserted_wu = insert_work_unit(setup_wg, setup_wu, parents_edges, children_edges)

    is_wu_child_in_wg = False
    for child in wg_with_inserted_wu.start.children:
        if child.id == setup_wu.id:
            is_wu_child_in_wg = True

    assert is_wu_child_in_wg, 'Given node is not reachable from start'

    is_wu_parent_in_wg = False
    for parent in wg_with_inserted_wu.finish.parents:
        if parent.id == setup_wu.id:
            is_wu_parent_in_wg = True

    assert is_wu_parent_in_wg, 'Finish is not reachable from given node'
