import math
import uuid
from collections import defaultdict

from sortedcontainers import SortedList

from sampo.schemas.exceptions import NotEnoughMaterialsInDepots, NoDepots, NoAvailableResources
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
        self._task_index = 0
        self._node_id2holder: dict[str, ResourceHolder] = landscape_config.holder_node_id2resource_holder
        self._holder_id2holder: dict[str, ResourceHolder] = {holder.id: holder for holder in landscape_config.holders}
        self._landscape = landscape_config
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

    @staticmethod
    def _check_resource_availability(state: SortedList[ScheduleEvent],
                                     required_resources: int,
                                     start_ind: int,
                                     finish_ind: int) -> bool:
        for idx in range(finish_ind - 1, start_ind - 1, -1):
            if state[idx].available_workers_count < required_resources:
                return False
        return True

    def _can_deliver_to_time(self, platform: LandGraphNode, finish_delivery_time: Time, materials: list[Material]) -> bool:
        holders = self._find_best_holders_by_dist(platform, materials)
        start_delivery_time = Time(0)
        for depot_id in holders:
            depot_state = self._timeline[depot_id]

            not_enough_mat = False
            # check whether remaining materials volume is enough for request
            for mat in materials:
                right_index = depot_state[mat.name].bisect_right(finish_delivery_time) - 1
                if depot_state[mat.name][right_index].available_workers_count < mat.count:
                    not_enough_mat = True
                    break
            if not_enough_mat:
                continue

            vehicle_state, vehicle_count_need = self._get_vehicle_info(depot_id, materials)
            start_ind = vehicle_state.bisect_right(start_delivery_time) - 1
            finish_ind = vehicle_state.bisect_right(finish_delivery_time)

            if not self._check_resource_availability(vehicle_state, vehicle_count_need, start_ind, finish_ind):
                continue

            road_available = []
            for road in self._landscape.roads:
                road_state = self._timeline[road.id]['vehicles']
                start_ind = road_state.bisect_right(start_delivery_time) - 1
                finish_ind = road_state.bisect_right(finish_delivery_time)
                if self._check_resource_availability(road_state, vehicle_count_need, start_ind, finish_ind):
                    road_available.append(road)

            if not self._landscape.construct_route(self._holder_id2holder[depot_id].node, platform, road_available):
                continue

            return True
        return False

    def can_schedule_at_the_moment(self, node: GraphNode, start_time: Time,
                                   materials: list[Material], exec_time: Time) -> bool:
        if node.work_unit.is_service_unit or not node.work_unit.need_materials():
            return True
        platform = self._landscape.works2platform[node]
        if not self._check_platform_availability(platform, materials, start_time, exec_time):
            return False
        start_delivery_time = Time(0)
        finish_delivery_time = start_time
        mat_request = self._request_materials(node, materials, finish_delivery_time)
        platform_state = self._timeline[platform.id]
        # if not mat_request:
        #     # TODO: make it more smart. Make check, if we can deliver materials to the start time of next (by the time) but the previous (by the prioritization) work, then we can schedule here
        #     # TODO: we need to have previous (by the prioritization) work
        #     for mat in mat_request:
        #         start_ind = platform_state[mat.name].bisect_right(finish_delivery_time) - 1
        #         finish_ind = platform_state[mat.name].bisect_left(Time.inf())
        #         if not self._check_resource_availability(platform_state[mat.name], mat.count, start_ind, finish_ind):
        #             return False
        #     return True

        return self._can_deliver_to_time(platform, finish_delivery_time, mat_request)

        # holders = self._find_best_holders_by_dist(self._landscape.works2platform[node], mat_request)
        #
        # for depot_id in holders:
        #     depot_state = self._timeline[depot_id]
        #
        #     not_enough_mat = False
        #     # check whether remaining materials volume is enough for request
        #     for mat in mat_request:
        #         right_index = depot_state[mat.name].bisect_right(finish_delivery_time) - 1
        #         if depot_state[mat.name][right_index].available_workers_count < mat.count:
        #             not_enough_mat = True
        #             break
        #     if not_enough_mat:
        #         continue
        #
        #     vehicle_state, vehicle_count_need = self._get_vehicle_info(depot_id, mat_request)
        #     start_ind = vehicle_state.bisect_right(start_delivery_time) - 1
        #     finish_ind = vehicle_state.bisect_right(finish_delivery_time)
        #
        #     if not self._check_resource_availability(vehicle_state, vehicle_count_need, start_ind, finish_ind):
        #         continue
        #
        #     road_available = []
        #     for road in self._landscape.roads:
        #         road_state = self._timeline[road.id]['vehicles']
        #         start_ind = road_state.bisect_right(start_delivery_time) - 1
        #         finish_ind = road_state.bisect_right(finish_delivery_time)
        #         if self._check_resource_availability(road_state, vehicle_count_need, start_ind, finish_ind):
        #             road_available.append(road)
        #
        #     if not self._landscape.construct_route(self._holder_id2holder[depot_id].node,
        #                                            self._landscape.works2platform[node], road_available):
        #         continue
        #
        #     return True
        # return False

    def _check_platform_availability(self, platform: LandGraphNode, materials: list[Material], start_time: Time,
                                     exec_time: Time) -> bool:
        platform_state = self._timeline[platform.id]
        mat_request = []
        delivery_finish_time = start_time + exec_time
        material_availability = []

        for mat in materials:
            start = platform_state[mat.name].bisect_right(start_time) - 1
            finish = platform_state[mat.name].bisect_left((Time.inf(), -1, EventType.INITIAL)) - 1

            if not self._check_resource_availability(platform_state[mat.name], mat.count, start, finish):
                if finish - start > 1:
                    mat_request.append(
                        Material(str(uuid.uuid4()), mat.name, mat.count))
                else:
                    material_availability.append(True)

        if all(material_availability) and len(material_availability) == len(materials) or len(mat_request) == 0:
            return True

        return self._can_deliver_to_time(platform, delivery_finish_time, mat_request)

    def _request_materials(self, node: GraphNode, materials: list[Material], work_start_time: Time) \
            -> list[Material]:
        request: list[Material] = []
        platform = self._landscape.works2platform[node]
        platform_state = self._timeline[platform.id]
        for need_mat in materials:
            start = platform_state[need_mat.name].bisect_right(work_start_time) - 1
            available_count_material = platform_state[need_mat.name][start].available_workers_count

            if available_count_material < need_mat.count:
                request.append(
                    Material(str(uuid.uuid4()), need_mat.name,
                             self._landscape.works2platform[node].resource_storage_unit.capacity[need_mat.name] -
                             available_count_material)
                )
            else:
                request.append(Material(str(uuid.uuid4()), need_mat.name, 0))

            return request

    def find_min_material_time(self, node: GraphNode, start_time: Time,
                               materials: list[Material], exec_time: Time) -> Time:
        if node.work_unit.is_service_unit or not materials:
            return start_time
        while not self._check_platform_availability(self._landscape.works2platform[node], materials, start_time,
                                                    exec_time):
            start_time += 1
        mat_request = self._request_materials(node, materials, start_time)

        if sum(mat.count for mat in mat_request) == 0:
            return start_time

        _, time = self._supply_resources(node, start_time, materials)
        # for mat in mat_request:
        #     self._validate(self._timeline[self._landscape.works2platform[node].id][mat.name], start_time, time, mat.count)
        return time

    def _find_best_holders_by_dist(self, node: LandGraphNode,
                                   materials: list[Material]) -> list[str]:
        """
        Get the best depot, that is the closest and the most early resource-supply available.
        :param node: GraphNode, that initializes resource delivery
        :param materials: required materials
        :return: best depot, time
        """
        holders = [self._node_id2holder[holder_id].id for dist, holder_id in self._landscape.get_sorted_holders(node)]

        if not holders:
            raise NoDepots(
                f"Schedule can not be built. There is no any resource holder")

        holders_with_req_mat = []
        count = 0
        for holder_id in holders:
            material_available = [False] * len(materials)
            for mat_ind in range(len(materials)):
                if not self._timeline[holder_id].get(materials[mat_ind].name, None) is None:
                    ind = self._timeline[holder_id][materials[mat_ind].name].bisect_left(Time.inf()) - 1
                    if self._timeline[holder_id][materials[mat_ind].name][ind].available_workers_count >= materials[
                        mat_ind].count:
                        material_available[mat_ind] = True
            if all(material_available):
                count += 1
                holders_with_req_mat.append(holder_id)

        if count == 0:
            raise NotEnoughMaterialsInDepots(
                f'Schedule can not be built. There is no resource holder, that has required materials')

        depots_result: list[str] = [depot_id for depot_id in holders_with_req_mat if
                                    all([True if self._timeline[depot_id][mat.name][
                                                     -1].available_workers_count >= mat.count else False for mat in
                                         materials])]

        return depots_result

    def deliver_resources(self, node: GraphNode,
                          deadline: Time,
                          materials: list[Material],
                          exec_time: Time,
                          update: bool = False) -> tuple[MaterialDelivery, Time]:
        if node.work_unit.is_service_unit or not node.work_unit.need_materials():
            return MaterialDelivery(node.id), deadline
        start_time = deadline
        can_deliver_to_next_node = False
        if self._check_platform_availability(self._landscape.works2platform[node], materials, start_time,
                                             exec_time):
            can_deliver_to_next_node = True
        else:
            while not self._check_platform_availability(self._landscape.works2platform[node], materials, start_time,
                                                        exec_time):
                start_time += 1
        mat_request = self._request_materials(node, materials, start_time)
        platform = self._landscape.works2platform[node]
        # if a job doesn't need materials
        if not mat_request or sum([mat.count for mat in mat_request]) == 0:
            if update and platform is not None:
                update_timeline_info: dict[str, list[tuple[str, int, Time, Time]]] = defaultdict(list)
                for mat in materials:
                    if can_deliver_to_next_node or start_time != deadline:
                        update_timeline_info[platform.id].append((mat.name, mat.count, start_time, start_time))
                    else:
                        update_timeline_info[platform.id].append((mat.name, mat.count, start_time, Time.inf()))

                self._update_timeline(update_timeline_info)

            return MaterialDelivery(node.id), start_time

        return self._supply_resources(node, start_time, mat_request, update)

    def _supply_resources(self, node: GraphNode,
                          deadline: Time,
                          materials: list[Material],
                          update: bool = False) -> tuple[MaterialDelivery, Time]:
        """
        Finds minimal time that given materials can be supplied, greater than given start time

        :param update: Should timeline be updated?
        It is necessary when materials are supplied, otherwise, timeline should not be updated
        :param node: GraphNode that initializes the resource-delivery
        :param deadline: the time work starts
        :param materials: material resources that are required to start
        :return: material deliveries, the time when resources are ready
        """
        delivery = MaterialDelivery(node.id)
        land_node = self._landscape.works2platform[node]

        # get the best depot that has enough materials and get its access start time (absolute value)
        depots = self._find_best_holders_by_dist(land_node, materials)
        depot_id = None
        min_depot_time = Time.inf()

        vehicles = []
        finish_vehicle_time = Time(0)
        start_delivery_time = deadline
        finish_delivery_time = Time(-1)

        road_deliveries = []

        # find time, that depot could provide resources
        local_min_start_time = Time.inf()
        for depot in depots:
            depot_mat_start_time = start_delivery_time
            for mat in materials:
                depot_mat_start_time = max(self._find_earliest_start_time(self._timeline[depot][mat.name],
                                                                          mat.count,
                                                                          depot_mat_start_time,
                                                                          Time.inf()), depot_mat_start_time)
            # get vehicles from the depot
            vehicle_state, vehicle_count_need = self._get_vehicle_info(depot, materials)
            vehicles: list[Vehicle] = self._holder_id2holder[depot].vehicles[:vehicle_count_need]

            road_start_time, deliveries, exec_ahead_time, exec_return_time = \
                self._get_route_time(self._holder_id2holder[depot].node,
                                     land_node, vehicles, depot_mat_start_time)

            depot_vehicle_start_time = self._find_earliest_start_time(vehicle_state,
                                                                      vehicle_count_need,
                                                                      road_start_time,
                                                                      exec_return_time + exec_ahead_time)

            if local_min_start_time > depot_vehicle_start_time:
                finish_vehicle_time = depot_vehicle_start_time + exec_return_time + exec_ahead_time
                min_depot_time = local_min_start_time = depot_vehicle_start_time
                finish_delivery_time = road_start_time + exec_ahead_time
                depot_id = depot

                road_deliveries = deliveries

        for mat in materials:
            delivery.add_delivery(mat.name, mat.count, min_depot_time, finish_delivery_time,
                                  self._holder_id2holder[depot_id].name)

        if not update:
            return delivery, finish_delivery_time

        update_timeline_info: dict[str, list[tuple[str, int, Time, Time]]] = defaultdict(list)

        # add update info about depot
        update_timeline_info[depot_id] = [(mat.name, mat.count, min_depot_time, min_depot_time + Time.inf()) for mat in
                                          materials]
        update_timeline_info[depot_id].append(('vehicles', len(vehicles), min_depot_time, finish_vehicle_time))

        # add update info about roads
        for delivery_dict in road_deliveries:
            for road_id, res_info in delivery_dict.items():
                update_timeline_info[road_id].append(res_info)

        # add update info about platform
        platform_state = self._timeline[land_node.id]
        for mat in materials:
            ind = platform_state[mat.name].bisect_right(finish_delivery_time) - 1
            available_mat = platform_state[mat.name][ind].available_workers_count
            if mat.name not in set(m.name for m in materials):
                update_timeline_info[land_node.id].append((mat.name, mat.count,
                                                           finish_delivery_time, Time.inf()))
            else:
                update_timeline_info[land_node.id].append((mat.name,
                                                           available_mat - land_node.resource_storage_unit.capacity[
                                                               mat.name] + mat.count,
                                                           finish_delivery_time, Time.inf()))

        self._update_timeline(update_timeline_info)

        return delivery, finish_delivery_time
    @staticmethod
    def _find_earliest_start_time(state: SortedList[ScheduleEvent],
                                  required_resources: int,
                                  parent_time: Time,
                                  exec_time: Time) -> Time:
        current_start_time = parent_time
        base_ind = state.bisect_right(parent_time) - 1

        ind = min(state.bisect_left(Time.inf()), len(state) - 1)
        last_time = state[ind].time

        while current_start_time < last_time:
            end_ind = state.bisect_right(current_start_time + exec_time)

            not_enough_resources = False
            for idx in range(end_ind - 1, base_ind - 1, -1):
                if state[idx].available_workers_count < required_resources:
                    base_ind = min(len(state) - 1, idx + 1)
                    not_enough_resources = True
                    break

            if not not_enough_resources:
                break

            current_start_time = state[base_ind].time

        return current_start_time
    def _get_route_time(self, holder_node: LandGraphNode, node: LandGraphNode, vehicles: list[Vehicle],
                        start_holder_time: Time) -> tuple[
        Time, list[dict[str, tuple[str, int, Time, Time]]], Time, Time]:
        def move_vehicles(from_node: LandGraphNode,
                          to_node: LandGraphNode,
                          parent_time: Time,
                          batch_size: int):
            road_delivery: dict[str, tuple[str, int, Time, Time]] = {}
            finish_time = parent_time

            available_roads = [road for road in self._landscape.roads if road.vehicles >= batch_size]
            route = self._landscape.construct_route(from_node, to_node, available_roads)

            if not route:
                raise NoAvailableResources(f'there is no chance to construct route with roads '
                                           f'{[road.name for road in available_roads]}')

            # check time availability of each part of 'route'
            for road_id in route:
                road_overcome_time = _id2road[road_id].overcome_time
                for i in range(len(vehicles) // batch_size):
                    start_time = self._find_earliest_start_time(state=self._timeline[road_id]['vehicles'],
                                                                required_resources=batch_size,
                                                                parent_time=finish_time,
                                                                exec_time=road_overcome_time)
                    road_delivery[road_id] = ('vehicles', batch_size,
                                              start_time, start_time + Time(road_overcome_time))
                    finish_time = start_time + road_overcome_time

            if not road_delivery:
                raise NoAvailableResources(f'there is no resources of available roads '
                                           f'{[road.name for road in available_roads]} '
                                           f'(probably roads have less bandwidth than is required)')

            return finish_time, road_delivery, finish_time - parent_time

        _id2road: dict[str, Road] = {road.id: road for road in self._landscape.roads}

        # | ------------ from holder to platform ------------ |
        finish_delivery_time, delivery, exec_time_ahead = move_vehicles(holder_node, node, start_holder_time,
                                                                        len(vehicles))

        # | ------------ from platform to holder ------------ |
        # compute the return time for vehicles
        # TODO: adapt move_vehicles() for batch_size = 1. Now the method don't allow to save deliveries of each (batch_size = 1) vehicle (similar with roads' deliveries)
        return_time, return_delivery, exec_time_return = move_vehicles(node, holder_node, finish_delivery_time,
                                                                       len(vehicles))

        return finish_delivery_time - exec_time_ahead, [delivery, return_delivery], exec_time_ahead, exec_time_return

    def _update_timeline(self, update_timeline_info: dict[str, list[tuple[str, int, Time, Time]]]) -> None:

        for res_holder_id, res_holder_info in update_timeline_info.items():
            res_holder_state = self._timeline[res_holder_id]

            for res_info in res_holder_info:
                task_index = self._task_index
                self._task_index += 1
                res_name, res_count, start_time, end_time = res_info
                res_state = res_holder_state[res_name]
                start_idx = res_state.bisect_right(start_time)
                end_idx = res_state.bisect_right(end_time)
                available_res_count = res_state[start_idx - 1].available_workers_count

                for event in res_state[start_idx: end_idx]:
                    assert event.available_workers_count >= res_count
                    event.available_workers_count -= res_count

                assert available_res_count >= res_count

                res_state.add(
                    ScheduleEvent(task_index, EventType.START, start_time, None, available_res_count - res_count))

                # end_idx = res_state.bisect_right(end_time) - 1
                if res_state[end_idx].time == end_time:
                    end_count = res_state[end_idx].available_workers_count
                else:
                    end_count = res_state[end_idx].available_workers_count + res_count

                res_state.add(ScheduleEvent(task_index, EventType.END, end_time, None, end_count))
    @staticmethod
    def _validate(state: SortedList[ScheduleEvent], start: Time, finish: Time, res_count_req: int):
        st = state.bisect_right(start)
        ed = state.bisect_right(finish)

        for event in state[st: ed]:
            assert event.available_workers_count >= res_count_req

        assert state[st - 1].available_workers_count >= res_count_req



