import uuid

from sortedcontainers import SortedList

from sampo.schemas.graph import GraphNode
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.landscape_graph import LandGraphNode
from sampo.schemas.resources import Material
from sampo.schemas.time import Time
from sampo.schemas.types import ScheduleEvent, EventType


class PlatformTimeline:
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
        self._landscape = landscape_config
        for mat_id, mat_dict in landscape_config.get_platforms_resources().items():
            self._timeline[mat_id] = {
                mat[0]: SortedList(iterable=(ScheduleEvent(-1, EventType.INITIAL, Time(0), None, mat[1]),),
                                   key=event_cmp)
                for mat in mat_dict.items()
            }

    def can_schedule_at_the_moment(self, node: GraphNode, start_time: Time,
                                   materials: list[Material]) -> bool:
        if not node.work_unit.need_materials():
            return True

        platform = self._landscape.works2platform[node]
        if not self._check_material_availability_on_platform(platform, materials, start_time):
            return False

        return True

    def _check_material_availability_on_platform(self, platform: LandGraphNode, materials: list[Material],
                                                 start_time: Time) -> bool:
        """
        Check the materials' availability on the `platform` at the `start_time`
        """
        # TODO Add delivery opportunity checking
        platform_state = self._timeline[platform.id]

        for mat in materials:
            start = platform_state[mat.name].bisect_right(start_time)
            finish = len(platform_state[mat.name])

            if finish - start > 1 or platform_state[mat.name][finish - 1].available_workers_count < mat.count:
                return False

        return True

    def get_material_for_delivery(self, node: GraphNode, materials: list[Material], work_start_time: Time) \
            -> list[Material]:
        """
        Returns `Material`s that should be delivered to the `node`'s platform to start the corresponding work
        """
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

        return request

    def find_min_material_time_with_additional(self, node: GraphNode, start_time: Time,
                                               materials: list[Material]) -> tuple[Time, list[Material]]:
        platform = self._landscape.works2platform[node]
        platform_state = self._timeline[platform.id]
        if not self._check_material_availability_on_platform(platform, materials, start_time):
            for mat in materials:
                ind = len(platform_state[mat.name]) - 1
                mat_last_time = platform_state[mat.name][ind].time
                start_time = max(mat_last_time, start_time)

        mat_request = self.get_material_for_delivery(node, materials, start_time)
        return start_time, mat_request

    def can_provide_resources(self,
                              node: GraphNode,
                              deadline: Time,
                              materials: list[Material],
                              update: bool = False) -> bool:

        start_time = deadline

        materials_for_delivery = self.get_material_for_delivery(node, materials, start_time)
        platform = self._landscape.works2platform[node]
        # TODO Simplify OR because it checks emptiness of materials for delivery
        if (not materials_for_delivery or sum([mat.count for mat in materials_for_delivery]) == 0) \
                and self._check_material_availability_on_platform(platform, materials_for_delivery, start_time):
            # work doesn't need materials
            if update:
                update_timeline_info: list[tuple[str, int, Time]] = [(mat.name, mat.count, start_time) for mat in materials]
                self.update_timeline(platform.id, update_timeline_info)

            return True
        return False

    def update_timeline(self, platform_id: str, update_timeline_info: list[tuple[str, int, Time]]) -> None:
        res_holder_state = self._timeline[platform_id]

        for res_info in update_timeline_info:
            task_index = self._task_index
            self._task_index += 1

            res_name, res_count, start_time = res_info
            res_state = res_holder_state[res_name]
            start_idx = res_state.bisect_right(start_time)

            available_res_count = res_state[start_idx - 1].available_workers_count

            res_state.add(
                ScheduleEvent(task_index, EventType.START, start_time, None, available_res_count - res_count)
            )
