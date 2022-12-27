from random import Random

import sampo.generator.config.gen_counts as gen_c
import sampo.generator.config.worker_req as wr
from sampo.generator.config.worker_req import scale_reqs
from sampo.schemas.graph import GraphNode, EdgeType
from sampo.schemas.interval import IntervalUniform
from sampo.schemas.utils import uuid_str
from sampo.schemas.works import WorkUnit


def _add_addition_work(probability: float, rand: Random | None = None) -> bool:
    return IntervalUniform(0, 1).rand_float(rand) <= probability


def _get_roads(parents: list[GraphNode], cluster_name: str, dist: float,
               rand: Random | None = None) -> dict[str, GraphNode]:
    road_nodes = dict()
    min_r = WorkUnit(uuid_str(rand), "minimal road",
                     scale_reqs(wr.MIN_ROAD, dist), group=f"{cluster_name}:road", volume=dist, volume_type="km")
    road_nodes['min'] = GraphNode(min_r, parents)
    temp_r = WorkUnit(uuid_str(rand), "temporary road",
                      scale_reqs(wr.TEMP_ROAD, dist), group=f"{cluster_name}:road", volume=dist, volume_type="km")
    road_nodes['temp'] = GraphNode(temp_r, [(road_nodes['min'], wr.ATOMIC_ROAD_LEN, EdgeType.LagFinishStart)])

    final_r = WorkUnit(uuid_str(rand), "final road", scale_reqs(wr.FINAL_ROAD, dist), group=f"{cluster_name}:road",
                       volume=dist, volume_type="km")
    road_nodes['final'] = GraphNode(final_r, [(road_nodes['temp'], wr.ATOMIC_ROAD_LEN, EdgeType.LagFinishStart)])
    return road_nodes


def _get_engineering_preparation(parents: list[GraphNode], cluster_name: str, boreholes_count: int,
                                 rand: Random | None = None) -> GraphNode:
    worker_req = wr.mul_borehole_volume(wr.ENGINEERING_PREPARATION, boreholes_count, wr.ENGINEERING_PREPARATION_BASE)
    work = WorkUnit(uuid_str(rand), "engineering preparation", worker_req, 
                    group=f"{cluster_name}:engineering",
                    volume=wr.get_borehole_volume(boreholes_count, wr.ENGINEERING_PREPARATION_BASE))
    node = GraphNode(work, parents)
    return node


def _get_power_lines(parents: list[GraphNode], cluster_name: str, dist_line: float,
                     dist_high_line: float | None = None, rand: Random | None = None) -> list[GraphNode]:
    worker_req = wr.scale_reqs(wr.POWER_LINE, dist_line)
    power_line_1 = WorkUnit(uuid_str(rand), "power line", worker_req,
                            group=f"{cluster_name}:electricity",
                            volume=dist_line, volume_type="km")
    power_line_2 = WorkUnit(uuid_str(rand), "power line", worker_req,
                            group=f"{cluster_name}:electricity",
                            volume=dist_line, volume_type="km")

    power_lines = [
        GraphNode(power_line_1, parents),
        GraphNode(power_line_2, parents),
    ]
    if dist_high_line is not None:
        worker_req_high = wr.scale_reqs(wr.POWER_LINE, dist_high_line)
        high_power_line = WorkUnit(uuid_str(rand), "high power line", worker_req_high,
                                   group=f"{cluster_name}:electricity", volume=dist_high_line, volume_type="km")
        power_lines.append(GraphNode(high_power_line, parents))

    return power_lines


def _get_pipe_lines(parents: list[GraphNode], cluster_name: str, pipe_dists: list[float],
                    rand: Random | None = None) -> list[GraphNode]:
    worker_req_pipe = wr.scale_reqs(wr.PIPE_LINE, pipe_dists[0])
    first_pipe = WorkUnit(uuid_str(rand), "pipe", worker_req_pipe, group=f"{cluster_name}:oil_gas_long_pipes",
                          volume=pipe_dists[0], volume_type="km")

    graph_nodes = [GraphNode(first_pipe, parents)]
    for i in range(1, len(pipe_dists)):
        node_work = WorkUnit(uuid_str(rand), "node", wr.PIPE_NODE,
                             group=f"{cluster_name}:oil_gas_long_pipes")
        graph_nodes.append(GraphNode(node_work, parents))
        worker_req_pipe = wr.scale_reqs(wr.PIPE_LINE, pipe_dists[i])
        pipe_work = WorkUnit(uuid_str(rand), "pipe", worker_req_pipe,
                             group=f"{cluster_name}:oil_gas_long_pipes",
                             volume=pipe_dists[i], volume_type="km")
        graph_nodes.append(GraphNode(pipe_work, parents))

    worker_req_loop = wr.scale_reqs(wr.PIPE_LINE, pipe_dists[0])
    looping = WorkUnit(uuid_str(rand), "looping", worker_req_loop, group=f"{cluster_name}:oil_gas_long_pipes",
                       volume=pipe_dists[0], volume_type="km")
    graph_nodes.append(GraphNode(looping, graph_nodes[0:1]))
    return graph_nodes


def _get_boreholes_equipment_group(parents: list[GraphNode], cluster_name: str, group_ind: int, borehole_count: int,
                                   rand: Random | None = None) -> list[GraphNode]:
    metering_install = WorkUnit(uuid_str(rand), "metering installation",
                                wr.METERING_INSTALL, group=f"{cluster_name}:borehole_env")
    worker_req_ktp_nep = wr.mul_borehole_volume(wr.KTP_NEP, borehole_count, wr.KTP_NEP_BASE)
    ktp_nep = WorkUnit(uuid_str(rand), "KTP and NEP",
                       worker_req_ktp_nep, group=f"{cluster_name}:borehole_env",
                       volume=wr.get_borehole_volume(borehole_count, wr.KTP_NEP_BASE))
    worker_req_tank = wr.mul_borehole_volume(wr.DRAINAGE_TANK, borehole_count, wr.DRAINAGE_TANK_BASE)
    drainage_tank = WorkUnit(uuid_str(rand), "drainage tank",
                             worker_req_tank, group=f"{cluster_name}:borehole_env",
                             volume=wr.get_borehole_volume(borehole_count, wr.DRAINAGE_TANK_BASE))
    nodes = [
        GraphNode(metering_install, parents),
        GraphNode(ktp_nep, parents),
        GraphNode(drainage_tank, parents),
    ]
    return nodes


def _get_boreholes_equipment_shared(parents: list[GraphNode], cluster_name: str,
                                    rand: Random | None = None) -> list[GraphNode]:
    water_block = WorkUnit(uuid_str(rand), "block water distribution", wr.WATER_BLOCK,
                           group=f"{cluster_name}:borehole_env")
    automation_block = WorkUnit(uuid_str(rand), "block local automation", wr.AUTOMATION_BLOCK,
                                group=f"{cluster_name}:borehole_env")
    block_dosage = WorkUnit(uuid_str(rand), "block dosage inhibitor", wr.BLOCK_DOSAGE,
                            group=f"{cluster_name}:borehole_env")
    start_filters = WorkUnit(uuid_str(rand), "start filters system", wr.START_FILTER,
                             group=f"{cluster_name}:borehole_env")
    firewall = WorkUnit(uuid_str(rand), "firewall tank", wr.FIREWALL, group=f"{cluster_name}:borehole_env")
    nodes = [
        GraphNode(water_block, parents),
        GraphNode(automation_block, parents),
        GraphNode(block_dosage, parents),
        GraphNode(start_filters, parents),
        GraphNode(firewall, parents),
    ]
    return nodes


def _get_boreholes(parents: list[GraphNode], cluster_name: str, group_ind: int, borehole_count: int,
                   rand: Random | None = None) -> list[GraphNode]:
    nodes = []
    for i in range(borehole_count):
        borehole_work = WorkUnit(uuid_str(rand), "borehole",
                                 wr.BOREHOLE, group=f"{cluster_name}:borehole_groups")
        nodes.append(GraphNode(borehole_work, parents))
    return nodes


def _get_boreholes_equipment_general(parents: list[GraphNode], cluster_name: str, pipes_count: int, masts_count: int,
                                     rand: Random | None = None) -> list[GraphNode]:
    nodes = []
    dists_sum = 0
    for i in range(pipes_count):
        dist = gen_c.DIST_BETWEEN_BOREHOLES.rand_float(rand)
        dists_sum += dist
        worker_req_pipe = scale_reqs(wr.POWER_NETWORK, dist)
        pipe_net_work = WorkUnit(uuid_str(rand), "elem of pipe_network", worker_req_pipe,
                                 group=f"{cluster_name}:oil_gas_pipe_net", volume=dist, volume_type="km")
        nodes.append(GraphNode(pipe_net_work, parents))

    worker_req_power = scale_reqs(wr.POWER_NETWORK, dists_sum)
    power_net_work = WorkUnit(uuid_str(rand), "power network", worker_req_power, group=f"{cluster_name}:electricity",
                              volume=dists_sum, volume_type="km")
    nodes.append(GraphNode(power_net_work, parents))

    for i in range(masts_count):
        light_mast_work = WorkUnit(uuid_str(rand), "mast", wr.LIGHT_MAST,
                                   group=f"{cluster_name}:light_masts")
        nodes.append(GraphNode(light_mast_work, parents))
    return nodes


def _get_handing_stage(parents: list[GraphNode], cluster_name: str, borehole_count: int,
                       rand: Random | None = None) -> GraphNode:
    worker_req = wr.mul_borehole_volume(wr.HANDING_STAGE, borehole_count, wr.HANDING_STAGE_BASE)
    work = WorkUnit(uuid_str(rand), "cluster handing", worker_req, group=f"{cluster_name}:handing_stage",
                    volume=wr.get_borehole_volume(borehole_count, wr.HANDING_STAGE_BASE))
    node = GraphNode(work, parents)
    return node


def get_cluster_works(root_node: GraphNode, cluster_name: str, pipe_nodes_count: int,
                      pipe_net_count: int, light_masts_count: int, borehole_counts: list[int],
                      roads: dict[str, GraphNode] | None = None,
                      rand: Random | None = None) -> (GraphNode, dict[str, GraphNode]):
    """
    Creates works on the development of the field on one object, i.e. a group of boreholes
    :param root_node: The parent node of the work graph for this object
    :param cluster_name: object name
    :param pipe_nodes_count: Number of pipeline segments from this field to other fields
    :param pipe_net_count: Number of pipes connecting boreholes
    :param light_masts_count: Number of floodlight masts
    :param borehole_counts: Number of boreholes
    :param roads: If the object is not connected to the central node, the road to which it is connected, otherwise None
    :param rand: Number generator with a fixed seed, or None for no fixed seed
    :return:
    """
    is_slave = roads is not None
    dist_to_parent = None
    if not is_slave:
        dist_to_parent = gen_c.DIST_TO_PARENT.rand_float(rand)
        roads = _get_roads([root_node], cluster_name, dist_to_parent, rand)
    engineering_preparation = _get_engineering_preparation([roads['min']], cluster_name, sum(borehole_counts), rand)
    preparation_stage = [roads['temp'], engineering_preparation]

    dist_line = gen_c.ONE_SECTION_PIPE.rand_float(rand)
    power_lines = _get_power_lines(preparation_stage, cluster_name, dist_line, dist_high_line=dist_to_parent, rand=rand)

    pipe_dists = [gen_c.ONE_SECTION_PIPE.rand_float(rand) for _ in range(pipe_nodes_count + 1)]
    pipe_lines_and_nodes = _get_pipe_lines(preparation_stage, cluster_name, pipe_dists, rand)

    boreholes_eq_shared = _get_boreholes_equipment_shared(preparation_stage, cluster_name, rand)
    boreholes_eq_general = _get_boreholes_equipment_general(preparation_stage, cluster_name,
                                                            pipes_count=pipe_net_count, masts_count=light_masts_count,
                                                            rand=rand)
    boreholes_all = []
    for i in range(len(borehole_counts)):
        boreholes_eq = _get_boreholes_equipment_group(preparation_stage, cluster_name,
                                                      group_ind=i, borehole_count=borehole_counts[i], rand=rand)
        boreholes = _get_boreholes(boreholes_eq + boreholes_eq_shared, cluster_name,
                                   group_ind=i, borehole_count=borehole_counts[i], rand=rand)
        boreholes_all += boreholes

    all_stages = [roads['final']] + power_lines + pipe_lines_and_nodes + boreholes_all + boreholes_eq_general
    handing_stage = _get_handing_stage(all_stages, cluster_name, sum(borehole_counts), rand=rand)
    return handing_stage, roads
