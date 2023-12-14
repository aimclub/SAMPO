import uuid

from sampo.scheduler.timeline.material_timeline import SupplyTimeline
from sampo.schemas.landscape import LandscapeConfiguration, ResourceHolder
from sampo.schemas.time import Time


def test_lg_init(setup_lg):
    print()
    lg, holders = setup_lg
    print([node.name for node in lg.nodes])


def test_landscape_init(setup_lg):
    lg, holders = setup_lg
    holders = [ResourceHolder(str(uuid.uuid4()),
                              holder.name,
                              vehicles=[],
                              node=holder) for holder in holders]
    landscape = LandscapeConfiguration(lg=lg, holders=holders)
    landscape.build_landscape()
    timeline = SupplyTimeline(landscape)
    print()
    for i in range(lg.vertex_count):
        print(f'From {lg.nodes[i].name}')
        for j in range(lg.vertex_count):
            print(f'    to {lg.nodes[j].name} through {lg.nodes[landscape.path_mx[i][j]].name}')
            print(f'    The last vertex to achieve target - {landscape.path_mx[i][j]}')
        print()


def test_building_routes(setup_lg):
    lg, holders = setup_lg
    holders = [ResourceHolder(str(uuid.uuid4()),
                              holder.name,
                              vehicles=[],
                              node=holder) for holder in holders]
    landscape = LandscapeConfiguration(lg=lg, holders=holders)
    landscape.build_landscape()
    timeline = SupplyTimeline(landscape)
    timeline._get_route_time(4, 0, [], landscape, Time(0))

