from dataclasses import dataclass

from sampo.schemas.identifiable import Identifiable
from sampo.schemas.requirements import WorkerReq, EquipmentReq, MaterialReq, ConstructionObjectReq, ZoneReq
from sampo.schemas.resources import Material
from sampo.schemas.serializable import AutoJSONSerializable
from sampo.utilities.serializers import custom_serializer


@dataclass
class WorkUnit(AutoJSONSerializable['WorkUnit'], Identifiable):
    """
    Class that describe vertex in graph (one work/task)
    """
    def __init__(self,
                 id: str,
                 name: str,
                 worker_reqs: list[WorkerReq] = None,
                 equipment_reqs: list[EquipmentReq] = None,
                 material_reqs: list[MaterialReq] = None,
                 object_reqs: list[ConstructionObjectReq] = None,
                 zone_reqs: list[ZoneReq] = None,
                 group: str = 'default',
                 is_service_unit: bool = False,
                 volume: float = 0,
                 volume_type: str = 'unit',
                 display_name: str = "",
                 workground_size: int = 100):
        """
        :param worker_reqs: list of required professions (i.e. workers)
        :param equipment_reqs: list of required equipment
        :param material_reqs: list of required materials (e.g. logs, stones, gravel etc.)
        :param object_reqs: list of required objects (e.g. electricity, pipelines, roads)
        :param zone_reqs: list of required zone statuses (e.g. opened/closed doors, attached equipment, etc.)
        :param group: union block of works
        :param is_service_unit: service units are additional vertexes
        :param volume: scope of work
        :param volume_type: unit of scope of work
        :param display_name: name of work
        """
        super(WorkUnit, self).__init__(id, name)
        if material_reqs is None:
            material_reqs = []
        if object_reqs is None:
            object_reqs = []
        if equipment_reqs is None:
            equipment_reqs = []
        if worker_reqs is None:
            worker_reqs = []
        if zone_reqs is None:
            zone_reqs = []
        self.worker_reqs = worker_reqs
        self.equipment_reqs = equipment_reqs
        self.object_reqs = object_reqs
        self.material_reqs = material_reqs
        self.zone_reqs = zone_reqs
        self.group = group
        self.is_service_unit = is_service_unit
        self.volume = volume
        self.volume_type = volume_type
        self.display_name = display_name if display_name else name
        self.workground_size = workground_size

    def __del__(self):
        for name, attr in self.__dict__.items():
            del attr

    def need_materials(self) -> list[Material]:
        return [req.material() for req in self.material_reqs]

    @custom_serializer('worker_reqs')
    def worker_reqs_serializer(self, value: list[WorkerReq]):
        """
        Return serialized list of worker requirements

        :param value: list of worker requirements
        :return: list of worker requirements
        """
        return [wr._serialize() for wr in value]

    @classmethod
    @custom_serializer('worker_reqs', deserializer=True)
    def worker_reqs_deserializer(cls, value):
        """
        Get list of worker requirements

        :param value: serialized list of work requirements
        :return: list of worker requirements
        """
        return [WorkerReq._deserialize(wr) for wr in value]

    @custom_serializer('zone_reqs')
    def zone_reqs_serializer(self, value: list[WorkerReq]):
        """
        Return serialized list of worker requirements

        :param value: list of worker requirements
        :return: list of worker requirements
        """
        return [wr._serialize() for wr in value]

    @classmethod
    @custom_serializer('zone_reqs', deserializer=True)
    def zone_reqs_deserializer(cls, value):
        """
        Get list of worker requirements

        :param value: serialized list of work requirements
        :return: list of worker requirements
        """
        return [WorkerReq._deserialize(wr) for wr in value]

    def __getstate__(self):
        # custom method to avoid calling __hash__() on GraphNode objects
        return self._serialize()

    def __setstate__(self, state):
        new_work_unit = self._deserialize(state)
        self.worker_reqs = new_work_unit.worker_reqs
        self.equipment_reqs = new_work_unit.equipment_reqs
        self.object_reqs = new_work_unit.object_reqs
        self.material_reqs = new_work_unit.material_reqs
        self.zone_reqs = new_work_unit.zone_reqs
        self.id = new_work_unit.id
        self.name = new_work_unit.name
        self.is_service_unit = new_work_unit.is_service_unit
        self.volume = new_work_unit.volume
        self.volume_type = new_work_unit.volume_type
        self.group = new_work_unit.group
        self.display_name = new_work_unit.display_name
        self.workground_size = new_work_unit.workground_size
