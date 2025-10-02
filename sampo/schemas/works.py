from dataclasses import dataclass
from typing import Any

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
                 model_name: dict[str, Any] | str,
                 worker_reqs: list[WorkerReq] = None,
                 equipment_reqs: list[EquipmentReq] = None,
                 material_reqs: list[MaterialReq] = None,
                 object_reqs: list[ConstructionObjectReq] = None,
                 zone_reqs: list[ZoneReq] = None,
                 description: str = '',
                 group: str = 'main project',
                 priority: int = 1,
                 is_service_unit: bool = False,
                 volume: float = 0,
                 display_name: str = "",
                 workground_size: int = 100):
        """
        :param model_name: dict with information that describes type of work for resource model.
                           In minimal it should contain 'granular_name' and 'measurement' entries.
                           `str` model_type is equal to {'granular_name': your_str_value, 'measurement': 'unit'}
        :param worker_reqs: list of required professions (i.e. workers)
        :param equipment_reqs: list of required equipment
        :param material_reqs: list of required materials (e.g. logs, stones, gravel etc.)
        :param object_reqs: list of required objects (e.g. electricity, pipelines, roads)
        :param zone_reqs: list of required zone statuses (e.g. opened/closed doors, attached equipment, etc.)
        :param description: the description. It is useful, for example, to show it on visualization
        :param group: union block of works
        :param is_service_unit: service units are additional vertexes
        :param volume: scope of work
        :param display_name: name of work
        """
        if isinstance(model_name, str):
            model_name = {'granular_name': model_name}
        if 'measurement' not in model_name:
            model_name['measurement'] = 'unit'

        self.model_name = model_name

        super(WorkUnit, self).__init__(id, 'dummy')

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
        self.description = description
        self.group = group
        self.is_service_unit = is_service_unit
        self.volume = float(volume)
        self.display_name = display_name if display_name else model_name['granular_name']
        self.priority = priority

    def __del__(self):
        for attr in self.__dict__.values():
            del attr

    def need_materials(self) -> list[Material]:
        return [req.material() for req in self.material_reqs]

    @custom_serializer('worker_reqs')
    @custom_serializer('zone_reqs')
    @custom_serializer('material_reqs')
    def serialize_serializable_list(self, value) -> list:
        """
        Return serialized list of values.
        Values should be serializable.

        :param value: list of values
        :return: list of serialized values
        """
        return [t._serialize() for t in value]

    @classmethod
    @custom_serializer('material_reqs', deserializer=True)
    def material_reqs_deserializer(cls, value) -> list[MaterialReq]:
        """
        Get list of material requirements

        :param value: serialized list of material requirements
        :return: list of material requirements
        """
        return [MaterialReq._deserialize(wr) for wr in value]

    @classmethod
    @custom_serializer('worker_reqs', deserializer=True)
    def worker_reqs_deserializer(cls, value) -> list[WorkerReq]:
        """
        Get list of worker requirements

        :param value: serialized list of work requirements
        :return: list of worker requirements
        """
        return [WorkerReq._deserialize(wr) for wr in value]

    @classmethod
    @custom_serializer('zone_reqs', deserializer=True)
    def zone_reqs_deserializer(cls, value) -> list[ZoneReq]:
        """
        Get list of worker requirements

        :param value: serialized list of work requirements
        :return: list of worker requirements
        """
        return [ZoneReq._deserialize(wr) for wr in value]

    @classmethod
    @custom_serializer('material_reqs', deserializer=True)
    def material_reqs_deserializer(cls, value) -> list[MaterialReq]:
        """
        Get list of material requirements

        :param value: serialized list of material requirements
        :return: list of material requirements
        """
        return [MaterialReq._deserialize(wr) for wr in value]

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
        self.model_name = new_work_unit.model_name
        self.is_service_unit = new_work_unit.is_service_unit
        self.volume = new_work_unit.volume
        self.group = new_work_unit.group
        self.display_name = new_work_unit.display_name
        self.priority = new_work_unit.priority
