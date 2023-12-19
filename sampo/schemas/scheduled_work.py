from dataclasses import dataclass
from typing import Any, Callable

from sampo.schemas.contractor import Contractor
from sampo.schemas.resources import Equipment, ConstructionObject, Worker
from sampo.schemas.serializable import AutoJSONSerializable
from sampo.schemas.time import Time
from sampo.schemas.works import WorkUnit
from sampo.schemas.zones import ZoneTransition
from sampo.utilities.serializers import custom_serializer


@dataclass
class ScheduledWork(AutoJSONSerializable['ScheduledWork']):
    """
    Contains all necessary info to represent WorkUnit in schedule:

    * WorkUnit
    * list of workers, that are required to complete task
    * start and end time
    * contractor, that complete task
    * list of equipment, that is needed to complete the task
    * list of materials - set of non-renewable resources
    * object - variable, that is used in landscape
    """

    ignored_fields = ['equipments', 'materials', 'object', 'work_unit']

    def __init__(self,
                 work_unit: WorkUnit,
                 start_end_time: tuple[Time, Time],
                 workers: list[Worker],
                 contractor: Contractor | str,
                 equipments: list[Equipment] | None = None,
                 zones_pre: list[ZoneTransition] | None = None,
                 zones_post: list[ZoneTransition] | None = None,
                 c_object: ConstructionObject | None = None):
        self.id = work_unit.id
        self.name = work_unit.name
        self.display_name = work_unit.display_name
        self.is_service_unit = work_unit.is_service_unit
        self.volume = work_unit.volume
        self.volume_type = work_unit.volume_type
        self.start_end_time = start_end_time
        self.workers = workers if workers is not None else []
        self.equipments = equipments if equipments is not None else []
        self.zones_pre = zones_pre if zones_pre is not None else []
        self.zones_post = zones_post if zones_post is not None else []
        self.object = c_object if c_object is not None else []

        if contractor is not None:
            if isinstance(contractor, str):
                self.contractor = contractor
            else:
                self.contractor = contractor.name if contractor.name else contractor.id
        else:
            self.contractor = ""

        self.cost = sum([worker.get_cost() * self.duration.value for worker in self.workers])

    def __str__(self) -> str:
        return f'ScheduledWork[work_unit={self.id}, start_end_time={self.start_end_time}, ' \
               f'workers={self.workers}, contractor={self.contractor}]'

    def __repr__(self) -> str:
        return self.__str__()

    @custom_serializer('workers')
    @custom_serializer('zones_pre')
    @custom_serializer('zones_post')
    @custom_serializer('start_end_time')
    def serialize_serializable_list(self, value) -> list:
        return [t._serialize() for t in value]

    @classmethod
    @custom_serializer('start_end_time', deserializer=True)
    def deserialize_time(cls, value) -> tuple[Time, Time]:
        return Time._deserialize(value[0]), Time._deserialize(value[1])

    @classmethod
    @custom_serializer('workers', deserializer=True)
    def deserialize_workers(cls, value) -> list[Worker]:
        return [Worker._deserialize(t) for t in value]

    @classmethod
    @custom_serializer('zones_pre', deserializer=True)
    @custom_serializer('zones_post', deserializer=True)
    def deserialize_zone_transitions(cls, value) -> list[ZoneTransition]:
        return [ZoneTransition._deserialize(t) for t in value]

    @property
    def start_time(self) -> Time:
        return self.start_end_time[0]

    @start_time.setter
    def start_time(self, val: Time):
        self.start_end_time = (val, self.start_end_time[1])

    @property
    def finish_time(self) -> Time:
        return self.start_end_time[1]

    @finish_time.setter
    def finish_time(self, val: Time):
        self.start_end_time = (self.start_end_time[0], val)

    @staticmethod
    def start_time_getter() -> Callable[[], Time]:
        return lambda x: x.start_end_time[0]

    @staticmethod
    def finish_time_getter() -> Callable[[], Time]:
        return lambda x: x.start_end_time[1]

    @property
    def duration(self) -> Time:
        start, end = self.start_end_time
        return end - start

    def to_dict(self) -> dict[str, Any]:
        return {
            'task_id': self.id,
            'task_name': self.name,
            'start': self.start_time.value,
            'finish': self.finish_time.value,
            'contractor_id': self.contractor,
            'workers': {worker.name: worker.count for worker in self.workers},
        }
