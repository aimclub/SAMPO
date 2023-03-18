import math
from operator import itemgetter

from sortedcontainers import SortedKeyList

from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.resources import Material
from sampo.schemas.time import Time


class SupplyTimeline:
    def __init__(self, landscape_config: LandscapeConfiguration):
        self._timeline = {}
        for landscape in landscape_config.get_all_resources():
            self._timeline[landscape.id] = SortedKeyList([(Time(0), landscape.count), (Time.inf(), landscape.count)],
                                                         itemgetter(0))

    def find_min_material_time(self, start_time: Time, materials: list[Material], batch_size: int) -> Time:
        sum_materials = sum([material.count for material in materials])
        ratio = sum_materials / batch_size
        batches = math.ceil(ratio)

        first_batch = [material.copy().with_count(material.count // batches) for material in materials]
        return self.supply_resources(start_time, first_batch, True)

    def deliver_materials(self, start_time: Time, finish_time: Time,
                          materials: list[Material], batch_size: int) -> tuple[Time, Time]:
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

        start_time = self.supply_resources(start_time, first_batch, False)
        finish_time = max([self.supply_resources(finish_time, batch, False) for batch in other_batches])

        return start_time, finish_time



    def supply_resources(self, deadline: Time, materials: list[Material], simulate: bool) -> Time:
        """
        Finds minimal time that given materials can be supplied, greater that given start time

        :param deadline: the time work starts
        :param materials: material resources that are required to start
        :param simulate: should timeline only find minimum supply time and not change timeline
        # :param workground_size: material capacity of the workground
        :return: the time when resources are ready
        """
        min_start_time = deadline

        for material in materials:
            material_timeline = self._timeline[material.id]

            count_left = material.count
            cur_start_time = deadline

            while count_left > 0:
                # find current area
                idx_left = material_timeline.bisect_key_left(cur_start_time)
                idx_right = material_timeline.bisect_key_right(cur_start_time)

                # find first area with count > 0
                while material_timeline[idx_left][1] == 0:
                    idx_right = idx_left
                    idx_left -= 1
                # TODO Handle case that we completely can't supply given materials to 'start_time' moment

                time_start = material_timeline[idx_left][0]
                time_end = material_timeline[idx_right][0]
                capacity = material_timeline[0][1]

                # grab resources
                # grab from start point, if fallen onto it
                if time_end == cur_start_time:
                    count_left -= min(count_left, material_timeline[idx_right][1])
                    if not simulate:
                        if count_left < 0:
                            material_timeline[idx_right] = (time_start, -count_left)
                        else:
                            material_timeline[idx_right] = (time_start, 0)

                # grab from current area
                cur_start_time = min(cur_start_time, time_end)
                while count_left > 0 and cur_start_time >= time_start:
                    count_left -= capacity
                    cur_start_time -= 1

                if not simulate:
                    # update the idx_left milestone
                    if cur_start_time < time_start:
                        if count_left >= 0:
                            material_timeline[idx_left] = (time_start, 0)
                        else:
                            material_timeline[idx_left] = (time_start, -count_left)
                    else:
                        # optimization: we don't need to add milestone with count = capacity, e.g. count_left = 0
                        if count_left < 0:
                            material_timeline.add((cur_start_time + 1, capacity + count_left))

            min_start_time = min(min_start_time, cur_start_time)

        return min_start_time





