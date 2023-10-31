import math
from operator import itemgetter

from sampo.schemas.exceptions import NotEnoughMaterialsInDepots, NoAvailableResources
from sampo.schemas.landscape import LandscapeConfiguration, MaterialDelivery
from sampo.schemas.resources import Material
from sampo.schemas.sorted_list import ExtendedSortedList
from sampo.schemas.time import Time


class SupplyTimeline:
    def __init__(self, landscape_config: LandscapeConfiguration):
        self._timeline = {}
        self._capacity = {}
        # material -> list of depots, that can supply this type of resource
        self._resource_sources: dict[str, dict[str, int]] = {}
        for landscape in landscape_config.get_all_resources():
            self._timeline[landscape.id] = ExtendedSortedList([(Time(0), landscape.count), (Time.inf(), 0)],
                                                              itemgetter(0))
            self._capacity[landscape.id] = landscape.count
            for count, res in landscape.get_available_resources():
                res_source = self._resource_sources.get(res, None)
                if res_source is None:
                    res_source = {}
                    self._resource_sources[res] = res_source
                res_source[landscape.id] = count

    def can_schedule_at_the_moment(self, id: str, start_time: Time, materials: list[Material], batch_size: int) -> bool:
        return self.find_min_material_time(id, start_time, materials, batch_size) == start_time

    def find_min_material_time(self, id: str, start_time: Time, materials: list[Material], batch_size: int) -> Time:
        sum_materials = sum([material.count for material in materials])
        ratio = sum_materials / batch_size
        batches = max(1, math.ceil(ratio))

        first_batch = [material.copy().with_count(material.count // batches) for material in materials]
        return self.supply_resources(id, start_time, first_batch, True)[1]

    def deliver_materials(self, id: str, start_time: Time, finish_time: Time,
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
        d, start_time = self.supply_resources(id, start_time, first_batch, False)
        deliveries.append(d)
        max_finish_time = finish_time
        for batch in other_batches:
            d, finish_time = self.supply_resources(id, max_finish_time, batch, False, start_time)
            deliveries.append(d)
            max_finish_time = finish_time if finish_time > max_finish_time else max_finish_time

        return deliveries, start_time, max_finish_time

    def _find_best_supply(self, material: str, count: int, deadline: Time) -> str:
        # TODO Make better algorithm
        if self._resource_sources.get(material, None) is None:
            raise NoAvailableResources(
                f'Schedule can not be built. No available resource sources with material {material}')
        depots = [depot_id for depot_id, depot_count in self._resource_sources[material].items()
                  if depot_count >= count]
        if not depots:
            raise NotEnoughMaterialsInDepots(
                f"Schedule can not be built. No one supplier has enough '{material}' material")
        depots = [(depot_id, self._timeline[depot_id].bisect_key_left(deadline), -self._capacity[depot_id])
                  for depot_id in depots]
        depots.sort(key=itemgetter(1, 2))

        return depots[0][0]

    def supply_resources(self, work_id: str, deadline: Time, materials: list[Material], simulate: bool,
                         min_supply_start_time: Time = Time(0)) \
            -> tuple[MaterialDelivery, Time]:
        """
        Finds minimal time that given materials can be supplied, greater than given start time

        :param work_id: work id
        :param deadline: the time work starts
        :param materials: material resources that are required to start
        :param simulate: should timeline only find minimum supply time and not change timeline
        :param min_supply_start_time:
        :return: material deliveries, the time when resources are ready
        """
        assert min_supply_start_time <= deadline
        delivery = MaterialDelivery(work_id)
        min_work_start_time = deadline

        def append_in_material_delivery_list(time: Time, count: int, delivery_list: list[tuple[Time, int]]):
            if not simulate:
                if count > need_count:
                    count = need_count
                delivery_list.append((time, count))

        def update_material_timeline_and_res_sources(timeline: ExtendedSortedList, mat_sources: dict[str, int]):
            for time, count in material_delivery_list:
                mat_sources[depot] -= count
                ind = timeline.bisect_key_left(time)
                timeline_time, timeline_count = timeline[ind]
                if timeline_time == time:
                    timeline[ind] = (time, timeline_count - count)
                else:
                    timeline.add((time, capacity - count))

            time, count = timeline[0]
            if not count:
                ind = 1
                is_zero_count = True
                while ind < len(timeline) - 1 and is_zero_count:
                    next_time, next_count = timeline[ind]
                    if not next_count and next_time == time + 1:
                        ind += 1
                        time = next_time
                    else:
                        is_zero_count = False
                if ind == len(timeline) - 1 or timeline[ind][0] != time + 1:
                    ind -= 1
                del timeline[:ind]

        for material in materials:
            if not material.count:
                continue
            material_sources = self._resource_sources[material.name]
            depot = self._find_best_supply(material.name, material.count, deadline)
            material_timeline = self._timeline[depot]
            capacity = self._capacity[depot]
            need_count = material.count
            idx_left = idx_base = material_timeline.bisect_key_right(deadline) - 1
            cur_time = deadline - 1
            material_delivery_list = [] if not simulate else None

            going_right = False

            while need_count > 0:
                # find current period
                time_left = material_timeline[idx_left][0]
                time_right = material_timeline[idx_left + 1][0]

                if going_right:
                    if cur_time == time_left:
                        time_left_capacity = material_timeline[idx_left][1]
                        if time_left_capacity:
                            append_in_material_delivery_list(cur_time, time_left_capacity, material_delivery_list)
                            need_count -= time_left_capacity
                        cur_time += 1
                    while need_count > 0 and cur_time < time_right:
                        append_in_material_delivery_list(cur_time, capacity, material_delivery_list)
                        need_count -= capacity
                        cur_time += 1
                    if need_count > 0:
                        idx_left += 1
                else:
                    while need_count > 0 and time_left < cur_time and min_supply_start_time <= cur_time:
                        append_in_material_delivery_list(cur_time, capacity, material_delivery_list)
                        need_count -= capacity
                        cur_time -= 1
                    if need_count > 0 and cur_time == time_left and min_supply_start_time <= cur_time:
                        time_left_capacity = material_timeline[idx_left][1]
                        if time_left_capacity:
                            append_in_material_delivery_list(cur_time, time_left_capacity, material_delivery_list)
                            need_count -= time_left_capacity
                        cur_time -= 1
                    if need_count > 0:
                        idx_left -= 1
                        if idx_left < 0 or cur_time < min_supply_start_time:
                            idx_left = idx_base
                            cur_time = deadline
                            going_right = True

            if not simulate:
                update_material_timeline_and_res_sources(material_timeline, material_sources)
                delivery.add_deliveries(material.name, material_delivery_list)

            min_work_start_time = max(min_work_start_time, cur_time)

        return delivery, min_work_start_time

    @property
    def resource_sources(self):
        return self._resource_sources
