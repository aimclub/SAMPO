import math
import uuid
from collections import defaultdict

from sortedcontainers import SortedList

from sampo.schemas.exceptions import NotEnoughMaterialsInDepots, NoDepots
from sampo.schemas.graph import GraphNode
from sampo.schemas.landscape import LandscapeConfiguration, ResourceHolder, Vehicle, Road, MaterialDelivery
from sampo.schemas.landscape_graph import LandGraphNode
from sampo.schemas.resources import Material
from sampo.schemas.time import Time
from sampo.schemas.types import ScheduleEvent, EventType


class SupplyTimeline:
    def __init__(self, landscape_config: LandscapeConfiguration):

        def event_cmp(event: ScheduleEvent | Time | tuple[Time, int, int]) -> tuple[Time, int, int]:
            if isinstance(event, ScheduleEvent):
                if event.event_type is EventType.INITIAL:
                    return Time(-1), -1, event.event_type.priority

                return event.time, event.seq_id, event.event_type.priority

            if isinstance(event, Time):
                # instances of Time must be greater than almost all ScheduleEvents with same time point
                return event, Time.inf().value, 2

            if isinstance(event, tuple):
                return event

            raise ValueError(f'Incorrect type of value: {type(event)}')

        self._timeline: dict[str, dict[str, SortedList[ScheduleEvent]]] = {}
        self._node_id2holder: dict[str, ResourceHolder] = landscape_config.holder_node_id2resource_holder
        self._holder_id2holder: dict[str, ResourceHolder] = {holder.id: holder for holder in landscape_config.holders}
        for resource in landscape_config.get_all_resources():
            for mat_id, mat_dict in resource.items():
                self._timeline[mat_id] = {
                    mat[0]: SortedList(iterable=(ScheduleEvent(-1, EventType.INITIAL, Time(0), None, mat[1]),),
                                       key=event_cmp)
                    for mat in mat_dict.items()
                }

    def _get_vehicle_info(self, depot_id: str, materials: list[Material]):
        vehicle_capacity = self._holder_id2holder[depot_id].vehicles[0].capacity
        need_mat = {mat.name: mat.count for mat in materials}
        return (self._timeline[depot_id]['vehicles'],
                max(math.ceil(count / mat.count) for name, count in need_mat.items()
                    for mat in vehicle_capacity
                    if mat.name == name))

    def can_schedule_at_the_moment(self, node: GraphNode, landscape: LandscapeConfiguration, start_time: Time,
                                   materials: list[Material], exec_time: Time) -> bool:
        """
        1) получить список подходящих складов
        2) пробежаться по этим складам и посмотреть, что в момент времени "start_time + exec_time" будут доступны все требуемые материалы
            2.1) что будет доступно достаточное кол-во машин
            2.2) что будут доступны дороги, по которым можно будет доставить ресурсы
        """
        holders = self._find_best_holders_by_dist(landscape, node.platform, materials)

        for depot_id in holders:
            depot_state = self._timeline[depot_id]

            not_enough_mat = False
            # check whether remaining materials volume is enough for request
            for mat in materials:
                right_index = depot_state[mat.name].bisect_right(start_time + exec_time)
                if depot_state[mat.name][right_index] < mat.count:
                    not_enough_mat = True
                    break
            if not_enough_mat:
                continue

            vehicle_state, vehicle_count_need = self._get_vehicle_info(depot_id, materials)
            start_ind = vehicle_state.bisect_right(start_time)
            finish_ind = vehicle_state.bisect_right(start_time + exec_time)

            not_enough_vehicles = False
            for ind in range(start_ind, finish_ind + 1):
                if vehicle_state[ind].available_workers_count < vehicle_count_need:
                    not_enough_vehicles = True
                    break
            if not_enough_vehicles:
                continue

            road_available = []
            for road in landscape.roads:
                road_state = self._timeline[road.id]['vehicles']
                start_ind = road_state.bisect_right(start_time)
                right_index = road_state.bisect_right(start_time + exec_time)
                if start_ind != right_index:
                    road_available.append(road)

            if not landscape.construct_route(self._holder_id2holder[depot_id].node, node.platform, road_available):
                continue
            return True
        return False

    def _request_materials(self, node: GraphNode, materials: list[Material], start_time: Time) -> list[Material]:
        request: list[Material] = []
        for need_mat in materials:
            ind = self._timeline[node.platform.id][need_mat.name].bisect_right(start_time) - 1
            available_count_material = self._timeline[node.platform.id][need_mat.name][ind].available_workers_count
            if available_count_material < need_mat.count:
                request.append(
                    Material(str(uuid.uuid4()), need_mat.name, node.platform.resource_storage_unit.capacity[need_mat.name] - available_count_material))
            else:
                request.append(
                    Material(str(uuid.uuid4()), need_mat.name, 0)
                )
        return request

    def find_min_material_time(self, node: GraphNode, landscape: LandscapeConfiguration, start_time: Time,
                               materials: list[Material]) -> Time:
        mat_request = self._request_materials(node, materials, start_time)

        if sum(mat.count for mat in mat_request) == 0:
            return start_time

        _, time = self.supply_resources(node, landscape, start_time, materials)
        return time

    def _find_best_holders_by_dist(self, landscape: LandscapeConfiguration, node: LandGraphNode,
                                   materials: list[Material]) -> list[str]:
        """
        Get the best depot, that is the closest and the most early resource-supply available.
        :param landscape: landscape
        :param node_ind: id of GraphNode, that initializes resource delivery
        :param material: required material
        :param count: the volume of required material
        :param deadline: time-deadline for resource delivery
        :return: best depot, time
        """
        holders = [self._node_id2holder[holder_id].id for dist, holder_id in landscape.get_sorted_holders(node)]

        if not holders:
            raise NoDepots(
                f"Schedule can not be built. There is no any resource holder")

        count = 0
        for holder_id in holders:
            material_available = [False] * len(materials)
            for mat_ind in range(len(materials)):
                if not self._timeline[holder_id].get(materials[mat_ind].name, None) is None:
                    if self._timeline[holder_id][materials[mat_ind].name][-1].available_workers_count >= materials[mat_ind].count:
                        material_available[mat_ind] = True
            count += 1 if all(material_available) else 0

        if not count:
            raise NotEnoughMaterialsInDepots(
                f'Schedule can not be built. There is no resource holder, that has required materials')

        depots_result: list[str] = [depot_id for depot_id in holders if
                                    all([True if self._timeline[depot_id][mat.name][-1].available_workers_count >= mat.count else False for mat in materials])]

        return depots_result

    def supply_resources(self, node: GraphNode,
                         landscape: LandscapeConfiguration,
                         deadline: Time,
                         materials: list[Material],
                         update: bool = False) -> tuple[MaterialDelivery, Time]:
        """
        Finds minimal time that given materials can be supplied, greater than given start time

        :param update: Should timeline be updated?
        It is necessary when materials are supplied, otherwise, timeline should not be updated
        :param node: GraphNode that initializes the resource-delivery
        :param landscape: landscape
        :param deadline: the time work starts
        :param materials: material resources that are required to start
        :return: material deliveries, the time when resources are ready
        """
        delivery = MaterialDelivery(node.id)
        mat_request = self._request_materials(node, materials, deadline)
        land_node = node.platform
        # if a job doesn't need materials
        if not mat_request or sum([mat.count for mat in mat_request]) == 0:
            if update and not node.platform is None:
                update_timeline_info: dict[str, list[tuple[str, int, Time, Time]]] = defaultdict(list)
                for mat in materials:
                    update_timeline_info[node.platform.id].append((mat.name, mat.count, deadline, Time.inf()))

                self._update_timeline(update_timeline_info)

            return MaterialDelivery(node.id), deadline

        # get the best depot that has enough materials and get its access start time (absolute value)
        depots = self._find_best_holders_by_dist(landscape, land_node, mat_request)
        depot_id = None
        min_depot_time = Time.inf()

        vehicles = []
        finish_vehicle_time = Time(0)
        finish_delivery_time = Time(0)

        road_deliveries = []

        # find time, that depot could provide resources
        for depot in depots:
            depot_time = Time(0)
            for mat in mat_request:
                depot_time = max(self._find_earliest_start_time(self._timeline[depot][mat.name],
                                                                mat.count,
                                                                Time(0),
                                                                Time.inf()),
                                 depot_time)
            # get vehicles from the depot
            vehicle_state, vehicle_count_need = self._get_vehicle_info(depot, mat_request)
            vehicles: list[Vehicle] = self._holder_id2holder[depot].vehicles[:vehicle_count_need]

            delivery_time, deliveries, exec_return_time = \
                self._get_route_time(self._holder_id2holder[depot].node,
                                     land_node, vehicles, landscape, depot_time)

            depot_time = max(depot_time, self._find_earliest_start_time(vehicle_state,
                                                                        vehicle_count_need,
                                                                        depot_time,
                                                                        delivery_time + exec_return_time))
            if min_depot_time > depot_time:
                finish_vehicle_time = delivery_time + exec_return_time
                min_depot_time = depot_time
                finish_delivery_time = delivery_time
                depot_id = depot

                road_deliveries = deliveries

        for mat in mat_request:
            delivery.add_delivery(mat.name, mat.count, min_depot_time, finish_delivery_time)

        if not update:
            return delivery, finish_delivery_time

        update_timeline_info: dict[str, list[tuple[str, int, Time, Time]]] = defaultdict(list)

        # add update info about depot
        update_timeline_info[depot_id] = [(mat.name, mat.count, min_depot_time, min_depot_time + Time.inf()) for mat in
                                          mat_request]
        update_timeline_info[depot_id].append(('vehicles', len(vehicles), min_depot_time, finish_vehicle_time))

        # add update info about roads
        for delivery_dict in road_deliveries:
            for road_id, res_info in delivery_dict.items():
                update_timeline_info[road_id].append(res_info)

        # add update info about platform
        platform_state = self._timeline[node.platform.id]
        for mat in materials:
            ind = platform_state[mat.name].bisect_right(finish_delivery_time) - 1
            available_mat = platform_state[mat.name][ind].available_workers_count
            if mat.name not in set(m.name for m in mat_request):
                update_timeline_info[node.platform.id].append((mat.name, mat.count,
                                                               finish_delivery_time, Time.inf()))
            else:
                update_timeline_info[node.platform.id].append((mat.name,
                                                               available_mat - node.platform.resource_storage_unit.capacity[mat.name] + mat.count,
                                                               finish_delivery_time, Time.inf()))

        self._update_timeline(update_timeline_info)

        return delivery, finish_delivery_time

    @staticmethod
    def _find_earliest_start_time(state: SortedList[ScheduleEvent],
                                  required_resources: int,
                                  parent_time: Time,
                                  exec_time: Time) -> Time:
        current_state_time = parent_time
        base_ind = state.bisect_right(parent_time) - 1

        while state[base_ind:]:
            end_ind = state.bisect_right(current_state_time + exec_time)

            not_enough_resources = False
            for idx in range(end_ind - 1, base_ind - 2, -1):
                if state[idx].available_workers_count < required_resources:
                    base_ind = max(idx, base_ind) + 1
                    not_enough_resources = True
                    break

            if not not_enough_resources:
                break

            if base_ind >= len(state):
                current_state_time = max(parent_time, state[-1].time + 1)
                break

            current_state_time = state[base_ind].time

        return current_state_time

    def _get_route_time(self, holder_node: LandGraphNode, node: LandGraphNode, vehicles: list[Vehicle],
                        landscape: LandscapeConfiguration,
                        start_holder_time: Time) -> tuple[Time, list[dict[str, tuple[str, int, Time, Time]]], Time]:
        def move_vehicles(from_node: LandGraphNode,
                          to_node: LandGraphNode,
                          _parent_time: Time,
                          batch_size: int):
            road_delivery: dict[str, tuple[str, int, Time, Time]] = {}
            parent_time = _parent_time

            route = landscape.get_route(from_node, to_node)

            # check time availability of each part of 'route'
            for road_id in route:
                road_overcome_time = _id2road[road_id].overcome_time
                for i in range(len(vehicles) // batch_size):
                    start_time = self._find_earliest_start_time(state=self._timeline[road_id]['vehicles'],
                                                                required_resources=batch_size,
                                                                parent_time=parent_time,
                                                                exec_time=road_overcome_time)
                    road_delivery[road_id] = ('vehicles', batch_size,
                                              start_time, start_time + Time(road_overcome_time))
                    parent_time = start_time + road_overcome_time

            return parent_time, road_delivery, parent_time - _parent_time

        _id2road: dict[str, Road] = {road.id: road for road in landscape.roads}

        # | ------------ from holder to platform ------------ |
        finish_delivery_time, delivery, exec_time_ahead = move_vehicles(holder_node, node, start_holder_time,
                                                                        len(vehicles))

        # | ------------ from platform to holder ------------ |
        # compute the return time for vehicles
        # TODO: adapt move_vehicles() for batch_size = 1. Now the method don't allow to save deliveries of each (batch_size = 1) vehicle (similar with roads' deliveries)
        return_time, return_delivery, exec_time_return = move_vehicles(node, holder_node, finish_delivery_time, len(vehicles))

        return finish_delivery_time, [delivery, return_delivery], exec_time_return

    def _update_timeline(self, update_timeline_info: dict[str, list[tuple[str, int, Time, Time]]]) -> None:

        for res_holder_id, res_holder_info in update_timeline_info.items():
            res_holder_state = self._timeline[res_holder_id]

            for res_info in res_holder_info:
                res_name, res_count, start_time, end_time = res_info
                res_state = res_holder_state[res_name]
                start_idx = res_state.bisect_right(start_time)
                end_idx = res_state.bisect_right(end_time)
                available_res_count = res_state[start_idx - 1].available_workers_count

                for event in res_state[start_idx: end_idx + 1]:
                    assert event.available_workers_count >= res_count
                    event.available_workers_count -= res_count

                assert available_res_count >= res_count

                if start_idx < end_idx:
                    event: ScheduleEvent = res_state[end_idx - 1]
                    end_count = event.available_workers_count + res_count
                else:
                    assert res_state[0].available_workers_count >= available_res_count
                    end_count = available_res_count

                res_state.add(
                    ScheduleEvent(int(start_idx), EventType.START, start_time, None, available_res_count - res_count))
                res_state.add(ScheduleEvent(int(end_time), EventType.END, end_time, None, end_count))
