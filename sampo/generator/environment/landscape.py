import uuid

from sampo.schemas import Material
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
