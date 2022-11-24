from random import Random
from typing import Dict, List, Optional, Tuple

from sampo.generator.config.worker_req import scale_reqs
from sampo.schemas.interval import IntervalUniform
from sampo.schemas.utils import uuid_str
from sampo.schemas.works import WorkUnit
from sampo.schemas.graph import GraphNode, EdgeType

import sampo.generator.config.worker_req as wr
import sampo.generator.config.gen_counts as gen_c


def get_start_stage(work_id: Optional[str] = "", rand: Optional[Random] = None) -> GraphNode:
    work_id = work_id or uuid_str(rand)
    work = WorkUnit(work_id, f"start of project", wr.START_PROJECT, group="start", is_service_unit=True)
    node = GraphNode(work, [])
    return node


def get_finish_stage(parents: List[GraphNode or Tuple[GraphNode, float, EdgeType]], work_id: Optional[str] = "",
                     rand: Optional[Random] = None) -> GraphNode:
    work_id = work_id or uuid_str(rand)
    work = WorkUnit(str(work_id), f"finish of project", wr.END_PROJECT, group="finish", is_service_unit=True)
    node = GraphNode(work, parents)
    return node


def add_addition_work(probability: float, rand: Optional[Random] = None) -> bool:
    return IntervalUniform(0, 1).rand_float(rand) <= probability


def get_roads(parents: List[GraphNode], cluster_name: str, dist: float,
              rand: Optional[Random] = None) -> Dict[str, GraphNode]:
    road_nodes = dict()
    min_r = WorkUnit(uuid_str(rand), f"minimal road `{cluster_name}`",
                     scale_reqs(wr.MIN_ROAD, dist), group="road", volume=dist, volume_type="km")
    road_nodes['min'] = GraphNode(min_r, parents)
    temp_r = WorkUnit(uuid_str(rand), f"temporary road `{cluster_name}`",
                      scale_reqs(wr.TEMP_ROAD, dist), group="road", volume=dist, volume_type="km")
    road_nodes['temp'] = GraphNode(temp_r, [(road_nodes['min'], wr.ATOMIC_ROAD_LEN, EdgeType.LagFinishStart)])

    final_r = WorkUnit(uuid_str(rand), f"final road `{cluster_name}`", scale_reqs(wr.FINAL_ROAD, dist), group="road",
                       volume=dist, volume_type="km")
    road_nodes['final'] = GraphNode(final_r, [(road_nodes['temp'], wr.ATOMIC_ROAD_LEN, EdgeType.LagFinishStart)])
    return road_nodes


def get_engineering_preparation(parents: List[GraphNode], cluster_name: str, boreholes_count: int,
                                rand: Optional[Random] = None) -> GraphNode:
    worker_req = wr.mul_borehole_volume(wr.ENGINEERING_PREPARATION, boreholes_count, wr.ENGINEERING_PREPARATION_BASE)
    work = WorkUnit(uuid_str(rand), f"engineering preparation `{cluster_name}`", worker_req, group="engineering",
                    volume=wr.get_borehole_volume(boreholes_count, wr.ENGINEERING_PREPARATION_BASE))
    node = GraphNode(work, parents)
    return node


def get_power_lines(parents: List[GraphNode], cluster_name: str, dist_line: float,
                    dist_high_line: Optional[float] = None, rand: Optional[Random] = None) -> List[GraphNode]:
    worker_req = wr.scale_reqs(wr.POWER_LINE, dist_line)
    power_line_1 = WorkUnit(uuid_str(rand), f"power line 1 `{cluster_name}`", worker_req, group="electricity",
                            volume=dist_line, volume_type="km")
    power_line_2 = WorkUnit(uuid_str(rand), f"power line 2 `{cluster_name}`", worker_req, group="electricity",
                            volume=dist_line, volume_type="km")

    power_lines = [
        GraphNode(power_line_1, parents),
        GraphNode(power_line_2, parents),
    ]
    if dist_high_line is not None:
        worker_req_high = wr.scale_reqs(wr.POWER_LINE, dist_high_line)
        high_power_line = WorkUnit(uuid_str(rand), f"high power line `{cluster_name}`", worker_req_high,
                                   group="electricity", volume=dist_high_line, volume_type="km")
        power_lines.append(GraphNode(high_power_line, parents))

    return power_lines


def get_pipe_lines(parents: List[GraphNode], cluster_name: str, pipe_dists: List[float],
                   rand: Optional[Random] = None) -> List[GraphNode]:
    worker_req_pipe = wr.scale_reqs(wr.PIPE_LINE, pipe_dists[0])
    first_pipe = WorkUnit(uuid_str(rand), f"pipe `{cluster_name}`-0", worker_req_pipe, group="oil_gas_pipe",
                          volume=pipe_dists[0], volume_type="km")

    graph_nodes = [GraphNode(first_pipe, parents)]
    for i in range(1, len(pipe_dists)):
        node_work = WorkUnit(uuid_str(rand), f"node `{cluster_name}`-{i - 1}", wr.PIPE_NODE, group="oil_gas_net")
        graph_nodes.append(GraphNode(node_work, parents))
        worker_req_pipe = wr.scale_reqs(wr.PIPE_LINE, pipe_dists[i])
        pipe_work = WorkUnit(uuid_str(rand), f"pipe `{cluster_name}`-{i}", worker_req_pipe, group="oil_gas_net",
                             volume=pipe_dists[i], volume_type="km")
        graph_nodes.append(GraphNode(pipe_work, parents))

    worker_req_loop = wr.scale_reqs(wr.PIPE_LINE, pipe_dists[0])
    looping = WorkUnit(uuid_str(rand), f"looping `{cluster_name}`", worker_req_loop, group="oil_gas_net",
                       volume=pipe_dists[0], volume_type="km")
    graph_nodes.append(GraphNode(looping, graph_nodes[0:1]))
    return graph_nodes


def get_boreholes_equipment_group(parents: List[GraphNode], cluster_name: str, group_ind: int, borehole_count: int,
                                  rand: Optional[Random] = None) -> List[GraphNode]:
    metering_install = WorkUnit(uuid_str(rand), f"metering installation `{cluster_name}`-{group_ind}-{borehole_count}",
                                wr.METERING_INSTALL, group="borehole_env")
    worker_req_ktp_nep = wr.mul_borehole_volume(wr.KTP_NEP, borehole_count, wr.KTP_NEP_BASE)
    ktp_nep = WorkUnit(uuid_str(rand), f"KTP and NEP `{cluster_name}`-{group_ind}-{borehole_count}",
                       worker_req_ktp_nep, group="borehole_env",
                       volume=wr.get_borehole_volume(borehole_count, wr.KTP_NEP_BASE))
    worker_req_tank = wr.mul_borehole_volume(wr.DRAINAGE_TANK, borehole_count, wr.DRAINAGE_TANK_BASE)
    drainage_tank = WorkUnit(uuid_str(rand), f"drainage tank `{cluster_name}`-{group_ind}-{borehole_count}",
                             worker_req_tank, group="borehole_env",
                             volume=wr.get_borehole_volume(borehole_count, wr.DRAINAGE_TANK_BASE))
    nodes = [
        GraphNode(metering_install, parents),
        GraphNode(ktp_nep, parents),
        GraphNode(drainage_tank, parents),
    ]
    return nodes


def get_boreholes_equipment_shared(parents: List[GraphNode], cluster_name: str,
                                   rand: Optional[Random] = None) -> List[GraphNode]:
    water_block = WorkUnit(uuid_str(rand), f"block water distribution `{cluster_name}`", wr.WATER_BLOCK,
                           group="borehole_env")
    automation_block = WorkUnit(uuid_str(rand), f"block local automation `{cluster_name}`", wr.AUTOMATION_BLOCK,
                                group="borehole_env")
    block_dosage = WorkUnit(uuid_str(rand), f"block dosage inhibitor `{cluster_name}`", wr.BLOCK_DOSAGE,
                            group="borehole_env")
    start_filters = WorkUnit(uuid_str(rand), f"start filters system `{cluster_name}`", wr.START_FILTER,
                             group="borehole_env")
    firewall = WorkUnit(uuid_str(rand), f"firewall tank `{cluster_name}`", wr.FIREWALL, group="borehole_env")
    nodes = [
        GraphNode(water_block, parents),
        GraphNode(automation_block, parents),
        GraphNode(block_dosage, parents),
        GraphNode(start_filters, parents),
        GraphNode(firewall, parents),
    ]
    return nodes


def get_boreholes(parents: List[GraphNode], cluster_name: str, group_ind: int, borehole_count: int,
                  rand: Optional[Random] = None) -> List[GraphNode]:
    nodes = []
    for i in range(borehole_count):
        borehole_work = WorkUnit(uuid_str(rand), f"borehole `{cluster_name}`-{group_ind}-{i}",
                                 wr.BOREHOLE, group="borehole")
        nodes.append(GraphNode(borehole_work, parents))
    return nodes


def get_boreholes_equipment_general(parents: List[GraphNode], cluster_name: str, pipes_count: int, masts_count: int,
                                    rand: Optional[Random] = None) -> List[GraphNode]:
    nodes = []
    dists_sum = 0
    for i in range(pipes_count):
        dist = gen_c.DIST_BETWEEN_BOREHOLES.rand_float(rand)
        dists_sum += dist
        worker_req_pipe = scale_reqs(wr.POWER_NETWORK, dist)
        pipe_net_work = WorkUnit(uuid_str(rand), f"elem of pipe_network `{cluster_name}`-{i}", worker_req_pipe,
                                 group="oil_gas_net", volume=dist, volume_type="km")
        nodes.append(GraphNode(pipe_net_work, parents))

    worker_req_power = scale_reqs(wr.POWER_NETWORK, dists_sum)
    power_net_work = WorkUnit(uuid_str(rand), f"power network `{cluster_name}`", worker_req_power, group="electricity",
                              volume=dists_sum, volume_type="km")
    nodes.append(GraphNode(power_net_work, parents))

    for i in range(masts_count):
        light_mast_work = WorkUnit(uuid_str(rand), f"floodlight mast `{cluster_name}`-{i}", wr.LIGHT_MAST,
                                   group="borehole_env")
        nodes.append(GraphNode(light_mast_work, parents))
    return nodes


def get_handing_stage(parents: List[GraphNode], cluster_name: str, borehole_count: int,
                      rand: Optional[Random] = None) -> GraphNode:
    worker_req = wr.mul_borehole_volume(wr.HANDING_STAGE, borehole_count, wr.HANDING_STAGE_BASE)
    work = WorkUnit(uuid_str(rand), f"cluster handing `{cluster_name}`", worker_req, group="handing_stage",
                    volume=wr.get_borehole_volume(borehole_count, wr.HANDING_STAGE_BASE))
    node = GraphNode(work, parents)
    return node


def get_cluster_works(root_node: GraphNode, cluster_name: str, pipe_nodes_count: int,
                      pipe_net_count: int, light_masts_count: int, borehole_counts: List[int],
                      roads: Optional[Dict[str, GraphNode]] = None,
                      rand: Optional[Random] = None) -> (GraphNode, Dict[str, GraphNode]):
    is_slave = roads is not None
    dist_to_parent = None
    if not is_slave:
        dist_to_parent = gen_c.DIST_TO_PARENT.rand_float(rand)
        roads = get_roads([root_node], cluster_name, dist_to_parent, rand)
    engineering_preparation = get_engineering_preparation([roads['min']], cluster_name, sum(borehole_counts), rand)
    preparation_stage = [roads['temp'], engineering_preparation]

    dist_line = gen_c.ONE_SECTION_PIPE.rand_float(rand)
    power_lines = get_power_lines(preparation_stage, cluster_name, dist_line, dist_high_line=dist_to_parent, rand=rand)

    pipe_dists = [gen_c.ONE_SECTION_PIPE.rand_float(rand) for _ in range(pipe_nodes_count + 1)]
    pipe_lines_and_nodes = get_pipe_lines(preparation_stage, cluster_name, pipe_dists, rand)

    boreholes_eq_shared = get_boreholes_equipment_shared(preparation_stage, cluster_name, rand)
    boreholes_eq_general = get_boreholes_equipment_general(preparation_stage, cluster_name,
                                                           pipes_count=pipe_net_count, masts_count=light_masts_count,
                                                           rand=rand)
    boreholes_all = []
    for i in range(len(borehole_counts)):
        boreholes_eq = get_boreholes_equipment_group(preparation_stage, cluster_name,
                                                     group_ind=i, borehole_count=borehole_counts[i], rand=rand)
        boreholes = get_boreholes(boreholes_eq + boreholes_eq_shared, cluster_name,
                                  group_ind=i, borehole_count=borehole_counts[i], rand=rand)
        boreholes_all += boreholes

    all_stages = [roads['final']] + power_lines + pipe_lines_and_nodes + boreholes_all + boreholes_eq_general
    handing_stage = get_handing_stage(all_stages, cluster_name, sum(borehole_counts), rand=rand)
    return handing_stage, roads
