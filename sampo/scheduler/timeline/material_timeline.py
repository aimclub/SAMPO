import math
from operator import itemgetter

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

    def find_min_material_time(self, id: str, start_time: Time, materials: list[Material], batch_size: int) -> Time:
        sum_materials = sum([material.count for material in materials])
        ratio = sum_materials / batch_size
        batches = math.ceil(ratio)

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
        batches = math.ceil(ratio)

        first_batch = [material.copy().with_count(material.count // batches) for material in materials]
        other_batches = [first_batch for _ in range(batches - 1)]
        other_batches.append([material.copy().with_count(material.count * (ratio - batches)) for material in materials])

        deliveries = []
        d, start_time = self.supply_resources(id, start_time, first_batch, False)
        deliveries.append(d)
        batch_processing = [self.supply_resources(id, finish_time, batch, False) for batch in other_batches]
        finish_time = max([b[1] for b in batch_processing])
        deliveries.extend([b[0] for b in batch_processing])

        return deliveries, start_time, finish_time

    def _find_best_supply(self, material: str, count: int) -> str:
        # TODO Make better algorithm
        # Return the first depot that can supply given materials
        return [depot_id for depot_id, depot_count in self._resource_sources[material].items() if depot_count >= count][
            0]

    @staticmethod
    def _grab_from_current_area(material_timeline: ExtendedSortedList,
                                cur_time: Time, idx_start: int, need_count: int,
                                capacity: int, going_right: bool, simulate: bool) -> tuple[Time, int]:
        """
        Processes the whole area starts with `idx_start` from `cur_time` moment

        :param material_timeline:
        :param cur_time:
        :param idx_start:
        :param need_count:
        :param capacity:
        :param going_right:
        :param simulate:
        :return: pair of finish time and grabbed amount
        """
        time_start = material_timeline[idx_start]
        time_end = material_timeline[idx_start + 1]
        grabbed = 0

        def process_start_milestone():
            nonlocal need_count, grabbed
            if cur_time == time_start:  # grab from start milestone
                start_count = material_timeline[idx_start][1]
                if need_count >= start_count:
                    need_count -= start_count
                    if not simulate:  # drop start milestone
                        material_timeline[idx_start] = (time_start, 0)
                    grabbed += start_count
                else:
                    if not simulate:  # subtract from start milestone
                        material_timeline[idx_start] = (time_start, start_count - need_count)
                    grabbed += need_count
                    need_count = 0

        process_start_milestone()

        if not going_right:
            return cur_time, grabbed

        # grab from area
        if going_right:
            while need_count > 0 and cur_time < time_end:  # inside area
                need_count -= capacity
                cur_time += 1

            if cur_time >= time_end:
                # we grabbed all the area, so we don't need to insert any milestone
                # our start milestone is already processed upper
                return cur_time, grabbed

            # if we are here, need_count < 0
            # we grabbed not all the area, so insert milestone to cur_time - 1 moment
            # 'cur_time - 1' because cur_time is the moment after the last 'need_count' addition performed
            # '-need_count' is resources count that are left at the last seen time moment
            material_timeline.add((cur_time - 1, -need_count))
        else:
            while need_count > 0 and time_start < cur_time:  # inside area
                need_count -= capacity
                cur_time -= 1

            if cur_time == time_start:
                # we grabbed all the area, so now we are allowed to grab the start milestone
                process_start_milestone()
                return cur_time, grabbed

            # if we are here, need_count < 0
            # we grabbed not all the area, so insert milestone to cur_time + 1 moment
            # 'cur_time + 1' because cur_time is the moment after the last 'need_count' subtraction performed
            # '-need_count' is resources count that are left at the last seen time moment
            material_timeline.add((cur_time + 1, -need_count))

        return cur_time, grabbed

    def supply_resources(self, id: str, deadline: Time, materials: list[Material], simulate: bool) \
            -> tuple[MaterialDelivery, Time]:
        """
        Finds minimal time that given materials can be supplied, greater that given start time

        :param id: work id
        :param deadline: the time work starts
        :param materials: material resources that are required to start
        :param simulate: should timeline only find minimum supply time and not change timeline
        # :param workground_size: material capacity of the workground
        :return: the time when resources are ready
        """
        min_start_time = deadline
        delivery = MaterialDelivery(id)

        grabbed = 0

        def record_delivery(name: str, time: Time, depot: str, count: int):
            nonlocal grabbed
            # update depot state
            self._resource_sources[name][depot] -= count
            # add to the result
            delivery.add_delivery(name, time, count)
            grabbed += count

        cur_start_time = deadline
        going_right = False

        def step_left():
            nonlocal cur_start_time, going_right
            cur_start_time -= 1
            if cur_start_time < 0:
                # we can't supply resources to deadline, let's go deeper
                going_right = True

        def step_right():
            nonlocal cur_start_time
            cur_start_time += 1

        for material in materials:
            depot = self._find_best_supply(material.name, material.count)
            material_timeline = self._timeline[depot]
            capacity = self._capacity[depot]

            count_left = material.count
            cur_start_time = deadline

            going_right = False

            while count_left > 0:
                # find current area
                idx_left = material_timeline.bisect_key_left(cur_start_time)
                grabbed = 0

                # find first area with count > 0
                if not going_right:
                    while idx_left >= 0 and material_timeline[idx_left][1] == 0:
                        idx_left -= 1
                        if idx_left < 0:
                            going_right = True

                if going_right:
                    # we can't supply given materials to 'start_time' moment
                    # so, let's go deeper
                    cur_start_time = deadline
                    idx_left = material_timeline.bisect_key_left(cur_start_time)
                    idx_right = material_timeline.bisect_key_right(cur_start_time)

                    while idx_right < len(material_timeline) and material_timeline[idx_left][1] == 0:
                        idx_left = idx_right
                        idx_right += 1

                cur_time, grabbed = self._grab_from_current_area(material_timeline, cur_start_time, idx_left,
                                                                 count_left, capacity, going_right, simulate)
                count_left -= grabbed

            min_start_time = min(min_start_time, cur_start_time)

        return delivery, min_start_time
