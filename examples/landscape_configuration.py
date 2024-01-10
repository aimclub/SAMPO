import uuid
from itertools import chain

from sampo.generator import SimpleSynthetic
from sampo.generator.environment import get_contractor_by_wg
from sampo.pipeline import SchedulingPipeline
from sampo.scheduler import HEFTScheduler
from sampo.schemas import LandscapeConfiguration, ResourceHolder, Material, MaterialReq, EdgeType, WorkGraph
from sampo.schemas.landscape import Vehicle
from sampo.schemas.landscape_graph import LandGraphNode, ResourceStorageUnit, LandGraph
from sampo.utilities.sampler import Sampler
from sampo.utilities.visualization import VisualizationMode


def setup_lg():
    platform1 = LandGraphNode(str(uuid.uuid4()), 'platform1',
                              ResourceStorageUnit({
                                  'mat1': 100,
                                  'mat2': 150,
                                  'mat3': 120
                              }))
    platform2 = LandGraphNode(str(uuid.uuid4()), 'platform2',
                              ResourceStorageUnit({
                                  'mat1': 70,
                                  'mat2': 80,
                                  'mat3': 90
                              }))
    platform3 = LandGraphNode(str(uuid.uuid4()), 'platform3',
                              ResourceStorageUnit({
                                  'mat1': 110,
                                  'mat2': 130,
                                  'mat3': 170
                              }))
    platform4 = LandGraphNode(str(uuid.uuid4()), 'platform4',
                              ResourceStorageUnit({
                                  'mat1': 180,
                                  'mat2': 190,
                                  'mat3': 200
                              }))
    holder1 = LandGraphNode(str(uuid.uuid4()), 'holder1',
                            ResourceStorageUnit({
                                'mat1': 500,
                                'mat2': 500,
                                'mat3': 500
                            }))
    holder2 = LandGraphNode(str(uuid.uuid4()), 'holder2',
                            ResourceStorageUnit({
                                'mat1': 705,
                                'mat2': 750,
                                'mat3': 800
                            }))
    platform1.add_neighbours([(platform3, 1.0, 2)])
    platform2.add_neighbours([(platform4, 2.0, 1)])
    platform3.add_neighbours([(holder1, 4.0, 2), (holder2, 3.0, 3)])
    platform4.add_neighbours([(holder1, 5.0, 1), (holder2, 7.0, 2)])
    holder1.add_neighbours([(holder2, 6.0, 2)])

    return LandGraph(nodes=[platform1, platform2, platform3, platform4, holder1, holder2]), [holder1, holder2]


def setup_landscape_many_holders(lg_info):
    lg, holders = lg_info
    return LandscapeConfiguration(holders=[
        ResourceHolder(str(uuid.uuid4()), 'holder1',
                       [
                           Vehicle(str(uuid.uuid4()), 'vehicle1', [
                               Material('111', 'mat1', 100),
                               Material('222', 'mat2', 100),
                               Material('333', 'mat3', 100)
                           ]),
                           Vehicle(str(uuid.uuid4()), 'vehicle2', [
                               Material('111', 'mat1', 150),
                               Material('222', 'mat2', 150),
                               Material('333', 'mat3', 150)
                           ])
                       ],
                       holders[0]),
        ResourceHolder(str(uuid.uuid4()), 'holder2',
                       [
                           Vehicle(str(uuid.uuid4()), 'vehicle1', [
                               Material('111', 'mat1', 120),
                               Material('222', 'mat2', 120),
                               Material('333', 'mat3', 120)
                           ]),
                           Vehicle(str(uuid.uuid4()), 'vehicle2', [
                               Material('111', 'mat1', 140),
                               Material('222', 'mat2', 140),
                               Material('333', 'mat3', 140)
                           ])
                       ],
                       holders[1]),
        ], lg=lg)

def setup_wg(lg_info):
    landscape = lg_info
    platforms = landscape.lg.nodes
    sr = Sampler(1e-1)

    l1n1 = sr.graph_node('l1n1', [], group='0', work_id='000001')
    l1n1.platform = platforms[0]
    l1n1.work_unit.material_reqs = [MaterialReq('mat1', 50)]
    l1n2 = sr.graph_node('l1n2', [], group='0', work_id='000002')
    l1n2.platform = platforms[0]
    l1n2.work_unit.material_reqs = [MaterialReq('mat1', 50)]

    l2n1 = sr.graph_node('l2n1', [(l1n1, 0, EdgeType.FinishStart)], group='1', work_id='000011')
    l2n1.platform = platforms[1]
    l2n1.work_unit.material_reqs = [MaterialReq('mat1', 50)]
    l2n2 = sr.graph_node('l2n2', [(l1n1, 0, EdgeType.FinishStart),
                                  (l1n2, 0, EdgeType.FinishStart)], group='1', work_id='000012')
    l2n2.platform = platforms[1]
    l2n2.work_unit.material_reqs = [MaterialReq('mat1', 50)]
    l2n3 = sr.graph_node('l2n3', [(l1n2, 1, EdgeType.LagFinishStart)], group='1', work_id='000013')
    l2n3.platform = platforms[2]
    l2n3.work_unit.material_reqs = [MaterialReq('mat1', 50)]

    l3n1 = sr.graph_node('l3n1', [(l2n1, 0, EdgeType.FinishStart),
                                  (l2n2, 0, EdgeType.FinishStart)], group='2', work_id='000021')
    l3n1.platform = platforms[2]
    l3n1.work_unit.material_reqs = [MaterialReq('mat1', 50)]
    l3n2 = sr.graph_node('l3n2', [(l2n2, 0, EdgeType.FinishStart)], group='2', work_id='000022')
    l3n2.platform = platforms[3]
    l3n2.work_unit.material_reqs = [MaterialReq('mat1', 50)]
    l3n3 = sr.graph_node('l3n3', [(l2n3, 1, EdgeType.LagFinishStart),
                                  (l2n2, 0, EdgeType.FinishStart)], group='2', work_id='000023')
    l3n3.platform = platforms[3]
    l3n3.work_unit.material_reqs = [MaterialReq('mat1', 50)]

    return WorkGraph.from_nodes([l1n1, l1n2, l2n1, l2n2, l2n3, l3n1, l3n2, l3n3]), landscape

# Set up attributes for the generated synthetic graph
synth_works_top_border = 2000
synth_unique_works = 300
synth_resources = 100

# Set up scheduling algorithm and project's start date
scheduler = HEFTScheduler()
start_date = "2023-01-01"

# Set up visualization mode (ShowFig or SaveFig) and the gant chart file's name (if SaveFig mode is chosen)
visualization_mode = VisualizationMode.ShowFig
gant_chart_filename = './output/synth_schedule_gant_chart.png'

# Generate synthetic graph with the given approximate works count,
# number of unique works names and number of unique resources
srand = SimpleSynthetic(rand=31)
wg, landscape = setup_wg(setup_landscape_many_holders(setup_lg()))

# Get information about created WorkGraph's attributes
works_count = len(wg.nodes)
work_names_count = len(set(n.work_unit.name for n in wg.nodes))
res_kind_count = len(set(req.kind for req in chain(*[n.work_unit.worker_reqs for n in wg.nodes])))
print(works_count, work_names_count, res_kind_count)

# Check the validity of the WorkGraph's attributes
assert (works_count <= synth_works_top_border * 1.1)
assert (work_names_count <= synth_works_top_border)
assert (res_kind_count <= synth_works_top_border)

# Get list with the Contractor object, which can satisfy the created WorkGraph's resources requirements
contractors = [get_contractor_by_wg(wg)]

project = SchedulingPipeline.create() \
    .wg(wg) \
    .contractors(contractors) \
    .landscape(landscape) \
    .schedule(scheduler) \
    .finish()
