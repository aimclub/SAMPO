import math
import uuid
from collections import defaultdict
from operator import itemgetter

from sortedcontainers import SortedList

from sampo.schemas.exceptions import NotEnoughMaterialsInDepots, NoAvailableResources
from sampo.schemas.graph import GraphNode
from sampo.schemas.landscape import LandscapeConfiguration, MaterialDelivery, ResourceHolder, Vehicle, Road
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
        self._id2holder: dict[str, ResourceHolder] = landscape_config.holder_id2resource_holder
        for resource in landscape_config.get_all_resources():
            for mat_id, mat_dict in resource.items():
                self._timeline[mat_id] = {
                    mat[0]: SortedList(iterable=(ScheduleEvent(-1, EventType.INITIAL, Time(0), None, mat[1]),),
                                       key=event_cmp)
                    for mat in mat_dict.items()
                }

    def can_schedule_at_the_moment(self, node: GraphNode, landscape: LandscapeConfiguration, start_time: Time,
                                   materials: list[Material], batch_size: int) -> bool:
        return self.find_min_material_time(node, landscape, start_time, materials, batch_size) == start_time

    def find_min_material_time(self, node: GraphNode, landscape: LandscapeConfiguration, start_time: Time,
                               materials: list[Material], batch_size: int) -> Time:
        mat_request: list[Material] = []

        for need_mat in materials:
            available_count_material = \
                self._timeline[node.platform.id][need_mat.name].bisect_right(start_time).available_workers_count
            if available_count_material < need_mat.count:
                mat_request.append(
                    Material(str(uuid.uuid4()), need_mat.name, available_count_material - need_mat.count))

        if not mat_request:
            return start_time

        return self.supply_resources(node, landscape, start_time, materials, True)[1]

    # TODO: delete method
    def deliver_materials(self, node: GraphNode, landscape: LandscapeConfiguration, start_time: Time, finish_time: Time,
                          materials: list[Material], batch_size: int) -> tuple[list[MaterialDelivery], Time, Time]:
        """
        Models material delivery.

        Delivery performed in batches sized by batch_size.

        :return: pair of material-driven minimum start and finish times
        """
        sum_materials = sum([material.count for material in materials])
        ratio = sum_materials / batch_size
        batches = max(1, math.ceil(ratio))

        first_batch = [material.copy().with_count(material.count // batches) for material in materials]
        other_batches = [first_batch for _ in range(batches - 2)]
        if batches > 1:
            other_batches.append([material.copy().with_count(material.count - batch_material.count * (batches - 1))
                                  for material, batch_material in zip(materials, first_batch)])

        deliveries = []
        d, start_time = self.supply_resources(node, landscape, start_time, first_batch, False)
        deliveries.append(d)
        max_finish_time = finish_time
        for batch in other_batches:
            d, finish_time = self.supply_resources(node, landscape, max_finish_time, batch, False, start_time)
            deliveries.append(d)
            max_finish_time = finish_time if finish_time > max_finish_time else max_finish_time

        return deliveries, start_time, max_finish_time

    def _find_best_holder_time(self, landscape: LandscapeConfiguration, node_id: int, materials: list[Material],
                               deadline: Time) -> tuple[str, Time]:
        """
        Get the best depot, that is the closest and the most early resource-supply available.
        :param landscape: landscape
        :param node_id: id of GraphNode, that initializes resource delivery
        :param material: required material
        :param count: the volume of required material
        :param deadline: time-deadline for resource delivery
        :return: best depot, time
        """
        holders = landscape.get_sorted_holders(node_id)

        # has any holder 'materials'
        material_available = [False] * len(materials)
        for holder_id in holders:
            for mat_ind in range(len(materials)):
                if not self._timeline[holder_id].get(materials[mat_ind].name, None) is None:
                    material_available[mat_ind] = True

        if not all(material_available):
            raise NoAvailableResources(
                f'Schedule can not be built. No available resource sources with materials '
                f'{[materials[mat_ind] for mat_ind in range(len(materials)) if not material_available[mat_ind]]}')

        depots = [id for id in holders for mat in materials if self._timeline[id][mat.name] >= mat.count]

        if not depots:
            raise NotEnoughMaterialsInDepots(
                f"Schedule can not be built. No one supplier has enough materials")

        depots_result = []

        for depot_id in depots:
            material_time = Time(0)
            for material in materials:
                material_time = max(material_time, self._timeline[depot_id][material.name].bisect_left(deadline).time)
            depots_result.append((depot_id, material_time))

        depots = [(depot_id, time)
                  for time, depot_id in depots_result]
        depots.sort(key=itemgetter(1, 2))

        return depots[0]

    def supply_resources(self, node: GraphNode,
                         landscape: LandscapeConfiguration,
                         deadline: Time,
                         materials: list[Material],
                         min_supply_start_time: Time = Time(0)) \
            -> tuple[MaterialDelivery, Time]:
        """
        Finds minimal time that given materials can be supplied, greater than given start time

        :param node: GraphNode that initializes the resource-delivery
        :param landscape: landscape
        :param deadline: the time work starts
        :param materials: material resources that are required to start
        :param simulate: should timeline only find minimum supply time and not change timeline
        :param min_supply_start_time:
        :return: material deliveries, the time when resources are ready
        """
        def merge_dicts(a, b):
            c = a.copy()
            c.update(b)
            return c

        assert min_supply_start_time <= deadline
        min_work_start_time = deadline
        # index of LangGraphNode
        node_ind = landscape.lg.node2ind[node.platform]

        # get the best depot that has enough materials and get its access start time (absolute value)
        depot_id, depot_time = self._find_best_holder_time(landscape, node_ind, materials, min_work_start_time)

        # get vehicles from the depot
        sorted_vehicles = SortedList(iterable=self._id2holder[depot_id].vehicles,
                                     key=lambda v: -v.volume)
        need_mat = {mat.name: mat.count for mat in materials}
        dict_mat = need_mat.copy()
        need_volume = sum([count for mat, count in need_mat.items()])
        vehicles: list[Vehicle] = []

        # find set the most suitable vehicles to supply resources
        for vehicle in sorted_vehicles:
            if need_volume > 0:
                vehicles.append(vehicle)
                for name, count in vehicle.resources.items():
                    need_mat[name] -= min(count, need_mat[name])
                need_volume = sum([count for mat, count in need_mat.items()])

        # calculate materials, that are gotten from holder
        load_resources = defaultdict(None)
        for vehicle in vehicles:
            for mat in vehicle.capacity:
                load_resources[mat.name] += mat.count

        finish_time, deliveries = self._get_route_time(landscape.lg.id2ind[depot_id], node_ind, vehicles, landscape, depot_time)

        update_timeline_info = {depot_id: {'start_time': depot_time,
                                           'exec_time': Time(1),
                                           'resource': dict_mat}}

        for delivery_dict in deliveries:
            update_timeline_info = merge_dicts(update_timeline_info, delivery_dict)

        self.update_timeline(update_timeline_info)

        return finish_time

    @staticmethod
    def _find_earliest_start_time(state: SortedList[ScheduleEvent],
                                  required_resources: int,
                                  parent_time: Time,
                                  exec_time: Time):
        current_state_time = parent_time
        base_ind = state.bisect_right(parent_time) - 1

        i = 0
        while state[base_ind:]:
            i += 1
            end_ind = state.bisect_right(current_state_time + exec_time)

            not_enough_resources = False
            for idx in range(end_ind - 1, base_ind - 2, -1):
                if state[idx].available_workers_count < required_resources or state[idx].time < parent_time:
                    base_ind = max(idx, base_ind) - 1
                    not_enough_resources = True
                    break

            if not not_enough_resources:
                break

            if base_ind >= len(state):
                current_state_time = max(parent_time, state[-1].time + 1)
                break

            current_state_time = state[base_ind].time

        return current_state_time

    def _get_route_time(self, holder_ind: int, node_ind: int, vehicles: list[Vehicle],
                        landscape: LandscapeConfiguration,
                        start_holder_time: Time) -> tuple[Time, list[dict[str, dict[str, Time | int]]]]:

        def get_path(path, v, u):
            if landscape.path_mx[v][u] == v:
                return
            get_path(path, v, landscape.path_mx[v][u][v][u])
            path.append(landscape.path_mx[v][u])

        def move_vehicles(from_node: int,
                          to_node: int,
                          _parent_time: Time,
                          batch_size: int):
            road_delivery: dict[str, dict[str, Time | int]] = {}
            parent_time = _parent_time

            path = [from_node]
            get_path(path, from_node, to_node)
            path.append(to_node)

            route = [landscape.road_mx[path[v]][path[v + 1]] for v in range(len(path) - 1)]

            # check time availability of each part of 'route'
            for road_id in route:
                for i in range(len(vehicles) // batch_size):
                    start_time = self._find_earliest_start_time(state=self._timeline[road_id]['vehicles'],
                                                                required_resources=batch_size,
                                                                parent_time=parent_time,
                                                                exec_time=_id2road[road_id].overcome_time)
                    road_delivery[road_id] = {'start_time': start_time,
                                              'exec_time': Time(_id2road[road_id].overcome_time),
                                              'resource': {'vehicles': batch_size}
                                              }
                    parent_time = start_time + _id2road[road_id].overcome_time

            return parent_time, road_delivery

        _id2road: dict[str, Road] = {road.id: road for road in landscape.roads}

        # | ------------ from holder to platform ------------ |
        finish_delivery_time, delivery = move_vehicles(holder_ind, node_ind, start_holder_time, len(vehicles))

        # | ------------ from platform to holder ------------ |
        # compute the return time for vehicles
        return_time, return_delivery = move_vehicles(node_ind, holder_ind, finish_delivery_time, 1)

        return finish_delivery_time, [delivery, return_delivery]

    def update_timeline(self, update_timeline_info: dict[str, dict[str, Time | dict[str, int]]]):
        # 1) establish the number of available resources in holder at the moment 'depot_time'
        # 2) replenish resources in 'platform'
        # 3) occupy chosen vehicles on route time
        # 4) occupy roads of found route on supplying time

        for res_holder_id, res_holder_info in update_timeline_info.items():
            res_holder_state = self._timeline[res_holder_id]
            start_time = res_holder_info['start_info']
            exec_time = res_holder_info['exec_time']

            for res_name, res_count in res_holder_info['resource'].items():
                start_id
