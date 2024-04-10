import math
from collections import defaultdict

from sortedcontainers import SortedList

from sampo.scheduler.timeline.base import BaseSupplyTimeline
from sampo.scheduler.timeline.platform_timeline import PlatformTimeline
from sampo.schemas.exceptions import NotEnoughMaterialsInDepots, NoDepots, NoAvailableResources
from sampo.schemas.graph import GraphNode
from sampo.schemas.landscape import LandscapeConfiguration, ResourceHolder, Vehicle, Road, MaterialDelivery
from sampo.schemas.landscape_graph import LandGraphNode
from sampo.schemas.resources import Material
from sampo.schemas.time import Time
from sampo.schemas.types import ScheduleEvent, EventType


class ToStartSupplyTimeline(BaseSupplyTimeline):
    """
    Material Timeline that implements the hybrid approach of resource supply -
    compares the time of resource delivery to work start and the time of delivery starting from the work start
    """

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

        self._platform_timeline = PlatformTimeline(landscape_config)
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

    @staticmethod
    def _get_necessary_vehicles_amount(depot: ResourceHolder, materials: list[Material]) -> int:
        vehicle_capacity = depot.vehicles[0].capacity
        need_mat = {mat.name: mat.count for mat in materials}
        return max(math.ceil(need_mat[material_carry_one_vehicle.name] / material_carry_one_vehicle.count)
                   for material_carry_one_vehicle in vehicle_capacity
                   if material_carry_one_vehicle.name in need_mat)

    def _can_deliver_to_time(self, node: GraphNode, finish_delivery_time: Time, materials: list[Material]) -> bool:
        # if finish_delivery_time < 60:
        #     _, time = self._supply_resources(node, finish_delivery_time, materials)
        # else:
        _, time = self._supply_resources_reversed_v(node, finish_delivery_time, materials)
        # assert time >= finish_delivery_time
        return time == finish_delivery_time

    def can_schedule_at_the_moment(self, node: GraphNode, start_time: Time,
                                   materials: list[Material]) -> bool:
        # if work doesn't need materials, return start time
        if not materials or node.work_unit.is_service_unit:
            return True

        if not self._platform_timeline.can_schedule_at_the_moment(node, start_time, materials):
            return False

        materials_for_delivery = self._platform_timeline.get_material_for_delivery(node, materials, start_time)
        # if there are no materials to be delivered, return start time
        if not materials_for_delivery:
            return True

        return self._can_deliver_to_time(node, start_time, materials_for_delivery)

    @staticmethod
    def _is_resource_available_at_the_moment(resource_state: SortedList, resource_amount: int, start_time: Time, finish_time: Time) -> bool:

        start_idx = resource_state.bisect_right(start_time) - 1

        end_idx = resource_state.bisect_left((finish_time, -1, EventType.INITIAL))

        for event in resource_state[start_idx: end_idx]:
            if event.available_workers_count < resource_amount:
                return False

        return True

    def find_min_material_time(self, node: GraphNode, start_time: Time,
                               materials: list[Material]) -> Time:
        # if work doesn't need materials, return start time
        if node.work_unit.is_service_unit or not materials:
            return start_time

        # first, we need to find the time when materials are ready on the platform and the materials
        # that should be delivered
        start_time, mat_request = self._platform_timeline.find_min_material_time_with_additional(node, start_time,
                                                                                                 materials)

        # if there are no materials to be delivered and the platform could allocate resources, return start time
        if not mat_request:
            return start_time

        # we need to find the optimal time when materials can be supplied
        # we compare the delivery algorithms and choose the best one
        # if start_time < 40:
        #     _, time = self._supply_resources(node, start_time, mat_request)
        # else:
        _, time = self._supply_resources_reversed_v(node, start_time, mat_request)

        return time

    def _find_best_holders_by_dist(self, node: LandGraphNode,
                                   materials: list[Material]) -> list[str]:
        """
        Get depots that have enough materials in sorted order by distance
        :param node: work that initializes resource delivery
        :param materials: required materials to perform the work
        :return: list of depots' ids
        """
        # get holders in sorted order by distance
        sorted_holder_ids = [self._node_id2holder[holder_id].id
                             for dist, holder_id in self._landscape.get_sorted_holders(node)]

        if not sorted_holder_ids:
            raise NoDepots(f'Schedule can not be built. There is no any resource holder')

        holders_can_supply_materials = []
        for holder_id in sorted_holder_ids:
            holder_state = self._timeline[holder_id]
            materials_available = 0

            for material in materials:
                if holder_state.get(material.name, None) is not None:
                    ind = holder_state[material.name].bisect_left((Time.inf(), -1, EventType.INITIAL)) - 1
                    # if the last event in the depot timeline has enough materials
                    # than the material is available
                    if holder_state[material.name][ind].available_workers_count >= material.count:
                        materials_available += 1
            #  if all materials are available, great!
            if materials_available == len(materials):
                holders_can_supply_materials.append(holder_id)

        if not holders_can_supply_materials:
            raise NotEnoughMaterialsInDepots(
                f'Schedule can not be built. There is no resource holder that has required materials')

        return holders_can_supply_materials

    def deliver_resources(self,
                          node: GraphNode,
                          deadline: Time,
                          materials: list[Material]) -> tuple[MaterialDelivery, Time]:
        # if work doesn't need materials, return start time
        if not materials or node.work_unit.is_service_unit:
            return MaterialDelivery(node.id), deadline

        # if the platform could allocate resources, then delivery is not needed and we return start time
        if self._platform_timeline.can_provide_resources(node, deadline, materials):
            return MaterialDelivery(node.id), deadline

        # get the materials that should be delivered to the platform
        materials_for_delivery = self._platform_timeline.get_material_for_delivery(node, materials, deadline)
        # if deadline < 40:
        #     delivery, time = self._supply_resources(node, deadline, materials_for_delivery, True)
        # else:
        delivery, time = self._supply_resources_reversed_v(node, deadline, materials_for_delivery, True)

        return delivery, time

    def _supply_resources_reversed_v(self, node: GraphNode,
                          deadline: Time,
                          materials: list[Material],
                          update: bool = False) -> tuple[MaterialDelivery, Time]:
        """
        Finds minimal time that the materials can be supplied, equal or greater than start time and make the delivery
        :param update: should timeline be updated,
        It is necessary when materials are supplied, otherwise, timeline should not be updated
        :param node: work that initializes the resource delivery
        :param deadline: the proposed time when work could start
        :param materials: material resources that are required to start the work
        :return: material deliveries and the time when resources are ready
        """

        delivery = MaterialDelivery(node.id)

        if not materials:
            return delivery, deadline

        platform = self._landscape.works2platform[node]

        # get the list of depots that have enough materials
        depots = [self._holder_id2holder[depot] for depot in self._find_best_holders_by_dist(platform, materials)]
        # the optimal depot that could supply resources
        selected_depot = None
        min_depot_time = Time.inf()

        continue_search = True

        selected_vehicles = []
        # the time when selected vehicles return back to the depot
        depot_vehicle_finish_time = Time(0)

        # the time when selected vehicles go to the platform and deliver materials
        finish_delivery_time = deadline - 1
        finish_delivery_time_tmp = Time.inf()

        # information (for updating timeline) about roads that are used to deliver materials
        road_deliveries = []

        # find the closest start delivery time to deadline
        # (it's explained by the fact that roads should be free as most time as possible,
        # because others could use them on another time)
        # (the finish delivery time should be equal or greater than work start time)
        amounts = 0
        while continue_search:
            amounts += 1

            finish_delivery_time += 1

            for depot in depots:
                vehicle_count_need = self._get_necessary_vehicles_amount(depot, materials)
                selected_vehicles = depot.vehicles[:vehicle_count_need]

                route_start_time, deliveries, exec_ahead_time, exec_return_time = \
                    self._get_reversed_route_with_additional(depot.node, platform, len(selected_vehicles), finish_delivery_time)

                # assert amounts < 200

                if route_start_time < 0:
                    continue

                if not self._is_resource_available_at_the_moment(self._timeline[depot.id]['vehicles'], vehicle_count_need,
                                                                 route_start_time,
                                                                 route_start_time + exec_ahead_time + exec_return_time):
                    continue

                # if not all([self._is_resource_available_at_the_moment(self._timeline[depot.id][mat.name], mat.count,
                #                                                       route_start_time,
                #                                                       Time.inf()) for mat in materials]):
                #     continue

                if finish_delivery_time_tmp > route_start_time + exec_return_time:
                    continue_search = False
                    depot_vehicle_finish_time = route_start_time + exec_ahead_time + exec_return_time
                    min_depot_time = route_start_time
                    finish_delivery_time_tmp = route_start_time + exec_return_time
                    selected_depot = depot
                    road_deliveries = deliveries

        for mat in materials:
            delivery.add_delivery(mat.name, mat.count, min_depot_time, finish_delivery_time, selected_depot.name)

        if not update:
            return delivery, finish_delivery_time

        update_timeline_info: dict[str, list[tuple[str, int, Time, Time]]] = defaultdict(list)

        # add update info about holder
        update_timeline_info[selected_depot.id] = [(mat.name, mat.count, min_depot_time, min_depot_time + Time.inf())
                                                   for mat in materials]
        update_timeline_info[selected_depot.id].append(('vehicles', len(selected_vehicles),
                                                        min_depot_time, depot_vehicle_finish_time))

        # add update info about roads
        for delivery_dict in road_deliveries:
            for road_id, res_info in delivery_dict.items():
                update_timeline_info[road_id].append(res_info)

        # update the platform timeline
        node_mat_req = {mat.name: mat.count for mat in node.work_unit.need_materials()}
        update_mat_req_info = [(mat.name, node_mat_req[mat.name] - mat.count, finish_delivery_time) for mat in
                               materials]
        self._platform_timeline.update_timeline(platform.id, update_mat_req_info)

        # update the supply timeline
        self._update_timeline(update_timeline_info)

        return delivery, finish_delivery_time

    def _supply_resources(self, node: GraphNode,
                          deadline: Time,
                          materials: list[Material],
                          update: bool = False) -> tuple[MaterialDelivery, Time]:
        """
        Finds minimal time that the materials can be supplied, equal or greater than start time and make the delivery
        :param update: should timeline be updated,
        It is necessary when materials are supplied, otherwise, timeline should not be updated
        :param node: work that initializes the resource delivery
        :param deadline: the proposed time when work could start
        :param materials: material resources that are required to start the work
        :return: material deliveries and the time when resources are ready
        """

        def get_finish_time(start_time: Time):
            """
            Iterates over depots and finds the earliest time when the depot could supply resources
            :return: the earliest time when the depot could supply resources with addition information
            """
            for depot in depots:
                depot_mat_start_time = start_time

                vehicle_count_need = self._get_necessary_vehicles_amount(depot, materials)
                selected_vehicles = depot.vehicles[:vehicle_count_need]

                # get information about route: start time, deliveries,
                # time of vehicles routing to platform and back separately
                route_start_time, deliveries, exec_ahead_time, exec_return_time = \
                    self._get_route_with_additional(depot.node, platform, selected_vehicles, depot_mat_start_time)

                # get the minimum start time that all required vehicles are available
                depot_vehicle_start_time = self._find_earliest_start_time(self._timeline[depot.id]['vehicles'],
                                                                          vehicle_count_need,
                                                                          route_start_time,
                                                                          exec_return_time + exec_ahead_time)

                yield depot_vehicle_start_time, exec_ahead_time, exec_return_time, depot, deliveries

        delivery = MaterialDelivery(node.id)

        if not materials:
            return delivery, deadline

        platform = self._landscape.works2platform[node]

        # get the list of depots that have enough materials
        depots = [self._holder_id2holder[depot] for depot in self._find_best_holders_by_dist(platform, materials)]
        # the optimal depot that could supply resources
        selected_depot = None
        min_depot_time = Time.inf()

        selected_vehicles = []
        # the time when selected vehicles return back to the depot
        depot_vehicle_finish_time = Time(0)

        start_delivery_time = Time(-1)
        # the time when selected vehicles go to the platform and deliver materials
        finish_delivery_time = Time(-1)

        # information (for updating timeline) about roads that are used to deliver materials
        road_deliveries = []

        # find the closest start delivery time to deadline
        # (it's explained by the fact that roads should be free as most time as possible,
        # because others could use them on another time)
        # (the finish delivery time should be equal or greater than work start time)
        while finish_delivery_time < deadline:
            start_delivery_time += 1

            # iterate over depots and find the earliest time when the depot could supply resources
            for depot_vehicle_start_time, exec_ahead_time, exec_return_time, depot, deliveries in get_finish_time(
                    start_delivery_time):
                # choose the current depot if only it could supply resources earlier than the previous one
                if finish_delivery_time > deadline and depot_vehicle_start_time + exec_ahead_time < finish_delivery_time \
                        or finish_delivery_time < deadline:
                    depot_vehicle_finish_time = depot_vehicle_start_time + exec_return_time + exec_ahead_time
                    min_depot_time = depot_vehicle_start_time
                    finish_delivery_time = depot_vehicle_start_time + exec_ahead_time
                    selected_depot = depot

                    road_deliveries = deliveries

        for mat in materials:
            delivery.add_delivery(mat.name, mat.count, min_depot_time, finish_delivery_time, selected_depot.name)

        if not update:
            return delivery, finish_delivery_time

        update_timeline_info: dict[str, list[tuple[str, int, Time, Time]]] = defaultdict(list)

        # add update info about holder
        update_timeline_info[selected_depot.id] = [(mat.name, mat.count, min_depot_time, min_depot_time + Time.inf())
                                                   for mat in materials]
        update_timeline_info[selected_depot.id].append(('vehicles', len(selected_vehicles),
                                                        min_depot_time, depot_vehicle_finish_time))

        # add update info about roads
        for delivery_dict in road_deliveries:
            for road_id, res_info in delivery_dict.items():
                update_timeline_info[road_id].append(res_info)

        # update the platform timeline
        node_mat_req = {mat.name: mat.count for mat in node.work_unit.need_materials()}
        update_mat_req_info = [(mat.name, node_mat_req[mat.name] - mat.count, finish_delivery_time) for mat in
                               materials]
        self._platform_timeline.update_timeline(platform.id, update_mat_req_info)

        # update the supply timeline
        self._update_timeline(update_timeline_info)

        return delivery, finish_delivery_time

    @staticmethod
    def _find_earliest_start_time(state: SortedList[ScheduleEvent],
                                  required_resources: int,
                                  parent_time: Time,
                                  exec_time: Time) -> Time:
        """
        Finds the earliest time when required resources are available
        :param state: the timeline of required resource
        :param required_resources: amount of resources
        :param parent_time: initial time when resource already should be available
        :param exec_time: period of time when resources should be available
        :return: the earliest time when required resources are available during the received period of time
        """
        current_start_time = parent_time
        base_ind = state.bisect_right(parent_time) - 1

        ind = state.bisect_left((Time.inf(), -1, EventType)) - 1
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

        assert current_start_time >= parent_time

        return current_start_time

    @staticmethod
    def _find_reversed_earliest_start_time(state: SortedList[ScheduleEvent],
                                           required_resources: int,
                                           parent_time: Time,
                                           exec_time: Time) -> Time:
        current_start_time = parent_time
        base_ind = state.bisect_right(parent_time) - 1

        first_time = state[0].time
        while current_start_time > first_time:
            end_ind = state.bisect_left(current_start_time - exec_time)

            not_enough_resources = False
            for idx in range(end_ind - 1, base_ind - 1, -1):
                if state[idx].available_workers_count < required_resources:
                    base_ind = max(0, idx - 1)
                    not_enough_resources = True
                    break

            if not not_enough_resources:
                break

            current_start_time = state[base_ind].time

        assert current_start_time <= parent_time

        return current_start_time

    def _get_reversed_route_with_additional(self, holder_node: LandGraphNode, node: LandGraphNode,
                                            vehicle_amount: int, start_reversed_time: Time) \
            -> tuple[Time, list[dict[str, tuple[str, int, Time, Time]]], Time, Time]:
        def move_vehicles_reversed(from_node: LandGraphNode,
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
                for i in range(vehicle_amount // batch_size):
                    start_time = self._find_reversed_earliest_start_time(state=self._timeline[road_id]['vehicles'],
                                                                         required_resources=batch_size,
                                                                         parent_time=finish_time,
                                                                         exec_time=road_overcome_time)
                    # road delivery has the direct form of time, because this info is using in update method
                    road_delivery[road_id] = ('vehicles', batch_size,
                                              start_time - Time(road_overcome_time), start_time)
                    finish_time = start_time - road_overcome_time

            if not road_delivery:
                raise NoAvailableResources(f'there is no resources of available roads '
                                           f'{[road.name for road in available_roads]} '
                                           f'(probably roads have less bandwidth than is required)')

            return finish_time, road_delivery, parent_time - finish_time

        if not vehicle_amount:
            raise

        _id2road: dict[str, Road] = {road.id: road for road in self._landscape.roads}

        # | ------------ from platform to holder ------------ |
        finish_delivery_time, delivery, exec_time_ahead = move_vehicles_reversed(node, holder_node, start_reversed_time,
                                                                        vehicle_amount)

        # | ------------ from holder to platform ------------ |
        # compute the return time for vehicles
        # TODO: adapt move_vehicles() for batch_size = 1. Now the method don't allow to save deliveries of each (batch_size = 1) vehicle (similar with roads' deliveries)
        return_time, return_delivery, exec_time_return = move_vehicles_reversed(holder_node, node, finish_delivery_time,
                                                                       vehicle_amount)

        if return_time < 0:
            finish_delivery_time = -1

        return finish_delivery_time, [delivery, return_delivery], exec_time_ahead, exec_time_return

    def _get_route_with_additional(self, holder_node: LandGraphNode, node: LandGraphNode, vehicles: list[Vehicle],
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
                                                                exec_time=-road_overcome_time)
                    road_delivery[road_id] = ('vehicles', batch_size,
                                              start_time, start_time + Time(road_overcome_time))
                    finish_time = start_time + road_overcome_time

            if not road_delivery:
                raise NoAvailableResources(f'there is no resources of available roads '
                                           f'{[road.name for road in available_roads]} '
                                           f'(probably roads have less bandwidth than is required)')

            return finish_time, road_delivery, finish_time - parent_time

        if not vehicles:
            raise

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

                end_idx = res_state.bisect_left((end_time, -1, EventType.INITIAL))
                available_res_count = res_state[start_idx - 1].available_workers_count

                assert available_res_count >= res_count

                for event in res_state[start_idx: end_idx]:
                    assert event.available_workers_count >= res_count
                    event.available_workers_count -= res_count

                res_state.add(
                    ScheduleEvent(task_index, EventType.START, start_time, None, available_res_count - res_count)
                )

                end_idx = res_state.bisect_right(end_time) - 1

                if res_state[end_idx].time == end_time:
                    end_count = res_state[end_idx].available_workers_count
                else:
                    end_count = res_state[end_idx].available_workers_count + res_count

                res_state.add(ScheduleEvent(task_index, EventType.END, end_time, None, end_count))

    def _validate(self, res_holder_id: str, res_info: tuple[str, int, Time, Time]):
        res_holder_state = self._timeline[res_holder_id]

        res_name, res_count, start_time, end_time = res_info
        res_state = res_holder_state[res_name]
        start_idx = res_state.bisect_right(start_time)

        end_idx = res_state.bisect_left((end_time, -1, EventType.INITIAL))
        available_res_count = res_state[start_idx - 1].available_workers_count

        assert available_res_count >= res_count

        for event in res_state[start_idx: end_idx]:
            assert event.available_workers_count >= res_count
            event.available_workers_count -= res_count
