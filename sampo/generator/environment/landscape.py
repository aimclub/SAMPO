import math
import random
import uuid
from collections import defaultdict

from sampo.schemas import Material, WorkGraph
from sampo.schemas.landscape import ResourceHolder, Vehicle, LandscapeConfiguration
from sampo.schemas.landscape_graph import LandGraphNode, ResourceStorageUnit, LandGraph


def setup_landscape(platforms_info: dict[str, dict[str, int]],
                    warehouses_info: dict[str, list[dict[str, int], list[tuple[str, dict[str, int]]]]],
                    roads_info: dict[str, list[tuple[str, float, int]]]) -> LandscapeConfiguration:
    """
    Build landscape configuration based on the provided information with structure as below.
    Attributes:
           Platform_info structure:
                {platform_name:
                    {material_name: material_count}
                }
           Warehouse_info structure:
                {holder_name:
                    [
                        {material_name: material_count},
                        [(vehicle_name, {vehicle_material_name: vehicle_material_count})]
                    ]
                }
           Roads_info structure:
                {platform_name:
                    [(neighbour_name, road_length, road_workload)]
                }
    :return: landscape configuration
    """
    name2platform: dict[str, LandGraphNode] = {}
    holders: list[ResourceHolder] = []
    for platform, platform_info in platforms_info.items():
        node = LandGraphNode(str(uuid.uuid4()), platform, ResourceStorageUnit(
                {name: count for name, count in platform_info.items()}
            ))
        name2platform[platform] = node

    for holder_name, holder_info in warehouses_info.items():
        materials = holder_info[0]
        vehicles = holder_info[1]
        holder_node = LandGraphNode(
            str(uuid.uuid4()), holder_name, ResourceStorageUnit(
                {name: count for name, count in materials.items()}
            ))
        name2platform[holder_name] = holder_node
        holders.append(ResourceHolder(
            str(uuid.uuid4()), holder_name,
            [
                Vehicle(str(uuid.uuid4()), name, [
                    Material(str(uuid.uuid4()), mat_name, mat_count)
                    for mat_name, mat_count in vehicle_mat_info.items()
                    ])
                for name, vehicle_mat_info in vehicles
            ],
            holder_node
        ))

    for from_node, adj_list in roads_info.items():
        name2platform[from_node].add_neighbours([(name2platform[node], length, workload)
                                                 for node, length, workload in adj_list])

    platforms: list[LandGraphNode] = list(name2platform.values())

    return LandscapeConfiguration(
        holders=holders,
        lg=LandGraph(nodes=platforms)
    )


def get_landscape_by_wg(wg: WorkGraph, rnd: random.Random) -> LandscapeConfiguration:
    nodes = wg.nodes
    max_materials = defaultdict(int)

    for node in nodes:
        for mat in node.work_unit.need_materials():
            if mat.name not in max_materials:
                max_materials[mat.name] = mat.count
            else:
                max_materials[mat.name] = max(max_materials[mat.name], mat.count)

    platforms_number = math.ceil(math.log(wg.vertex_count))
    platforms = []
    for i in range(platforms_number):
        materials_name = list(max_materials.keys())
        platforms.append(LandGraphNode(str(uuid.uuid4()), f'platform{i}',
                                       ResourceStorageUnit(
                                           {
                                               name: rnd.randint(max(max_materials[name], 1),
                                                                 3 * max(max_materials[name], 1))
                                               for name in materials_name
                                           }
                                       )))

    for i, platform in enumerate(platforms):
        if i == platforms_number - 1:
            continue
        neighbour_platforms = rnd.choices(platforms[i + 1:], k=rnd.randint(1, math.ceil(len(platforms[i + 1:]) / 3)))

        neighbour_platforms_tmp = neighbour_platforms.copy()
        for neighbour in neighbour_platforms:
            if neighbour in platform.neighbours:
                neighbour_platforms_tmp.remove(neighbour)
        neighbour_platforms = neighbour_platforms_tmp

        neighbour_edges = [(neighbour, rnd.uniform(1.0, 10.0), rnd.randint(30, 50))
                           for neighbour in neighbour_platforms]
        platform.add_neighbours(neighbour_edges)

    # nodes_without_ins_child = [node for node in nodes if node.inseparable_parent is None]
    # number_of_nodes_without_ins_child = len(nodes_without_ins_child)

    platforms_tmp = ((wg.vertex_count // platforms_number) * platforms +
                     platforms[:wg.vertex_count % platforms_number])
    rnd.shuffle(platforms_tmp)

    for node, platform in zip(wg.nodes, platforms_tmp):
        if node.edges_to and node.edges_from:
            platform.add_works(node)

    holders_number = math.ceil(math.sqrt(math.log(wg.vertex_count)))
    holders_node = []
    holders = []

    for i in range(holders_number):
        if not max_materials:
            materials_name = []
        else:
            materials_name = rnd.choices(list(max_materials.keys()), k=rnd.randint(min(len(max_materials), 1), max(len(max_materials), 1)))
        holders_node.append(LandGraphNode(str(uuid.uuid4()), f'holder{i}',
                                          ResourceStorageUnit(
                                              {
                                                  name: max(max_materials[name], 1) * wg.vertex_count
                                                  for name in materials_name
                                              }
                                          )))
        neighbour_platforms = rnd.choices(holders_node[:-1] + platforms, k=rnd.randint(1, len(holders_node[:-1] + platforms)))

        neighbour_platforms_tmp = neighbour_platforms.copy()
        for neighbour in neighbour_platforms:
            if neighbour in holders_node[-1].neighbours:
                neighbour_platforms_tmp.remove(neighbour)
        neighbour_platforms = neighbour_platforms_tmp

        neighbour_edges = [(neighbour, rnd.uniform(1.0, 10.0), rnd.randint(30, 50))
                           for neighbour in neighbour_platforms]
        holders_node[-1].add_neighbours(neighbour_edges)

        vehicles_number = rnd.randint(10, 20)
        holders.append(ResourceHolder(str(uuid.uuid4()), holders_node[-1].name,
                                      vehicles=[
                                          Vehicle(str(uuid.uuid4()), f'vehicle{j}',
                                                  [Material(name, name, math.ceil(math.sqrt(count)))
                                                   for name, count in max_materials.items()])
                                          for j in range(vehicles_number)
                                      ], node=holders_node[-1]))

    lg = LandGraph(nodes=platforms + holders_node)
    return LandscapeConfiguration(holders, lg)