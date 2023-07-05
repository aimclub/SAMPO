from dataclasses import dataclass

from sampo.schemas.identifiable import Identifiable
from sampo.schemas.requirements import WorkerReq, EquipmentReq, MaterialReq, ConstructionObjectReq
from sampo.schemas.resources import Worker, Material
from sampo.schemas.serializable import AutoJSONSerializable
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.utilities.serializers import custom_serializer


@dataclass
class WorkUnit(AutoJSONSerializable['WorkUnit'], Identifiable):
    """
    Class that describe vertex in graph (one work/task)
    """
    def __init__(self, id: str, name: str, worker_reqs: list[WorkerReq] = [], equipment_reqs: list[EquipmentReq] = [],
                 material_reqs: list[MaterialReq] = [], object_reqs: list[ConstructionObjectReq] = [],
                 group: str = 'default', is_service_unit=False, volume: float = 0,
                 volume_type: str = "unit", display_name: str = "", workground_size: int = 100):
        """
        :param worker_reqs: list of required professions (i.e. workers)
        :param equipment_reqs: list of required equipment
        :param material_reqs: list of required materials (e.g. logs, stones, gravel etc.)
        :param object_reqs: list of required objects (e.g. electricity, pipelines, roads)
        :param group: union block of works
        :param is_service_unit: service units are additional vertexes
        :param volume: scope of work
        :param volume_type: unit of scope of work
        :param display_name: name of work
        """
        super(WorkUnit, self).__init__(id, name)
        self.worker_reqs = worker_reqs
        self.equipment_reqs = equipment_reqs
        self.object_reqs = object_reqs
        self.material_reqs = material_reqs
        self.group = group
        self.is_service_unit = is_service_unit
        self.volume = volume
        self.volume_type = volume_type
        self.display_name = display_name
        self.workground_size = workground_size

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

    # TODO: move this logit to WorkTimeEstimator
    def estimate_static(self, worker_list: list[Worker], work_estimator: WorkTimeEstimator = None) -> Time:
        """
        Calculate summary time of task execution (without stochastic part)

        :param worker_list:
        :param work_estimator: available
        :return: time of task execution
        """
        if work_estimator:
            # TODO Is it should be here, not in preprocessing???
            workers = {w.name.replace("_res_fact", ""): w.count for w in worker_list}
            work_time = work_estimator.estimate_time(self.name.split("_stage_")[0], self.volume, workers)
            if work_time > 0:
                return work_time

        return work_estimator.estimate_time(self, worker_list)

    def estimate_stochastic(self, worker_list: list[Worker], work_estimator: WorkTimeEstimator = None) -> Time:
        """
        Calculate summary time of task execution (considering stochastic part)

        :param work_estimator:
        :param worker_list:
        :param rand: random number
        :return: time of task execution
        """
        return work_estimator.estimate_time(self, worker_list)

    def __getstate__(self):
        # custom method to avoid calling __hash__() on GraphNode objects
        return self._serialize()

    def __setstate__(self, state):
        new_work_unit = self._deserialize(state)
        self.worker_reqs = new_work_unit.worker_reqs
        self.equipment_reqs = new_work_unit.equipment_reqs
        self.object_reqs = new_work_unit.object_reqs
        self.material_reqs = new_work_unit.material_reqs
        self.id = new_work_unit.id
        self.name = new_work_unit.name
        self.is_service_unit = new_work_unit.is_service_unit
        self.volume = new_work_unit.volume
        self.volume_type = new_work_unit.volume_type
        self.group = new_work_unit.group
        self.display_name = new_work_unit.display_name
        self.workground_size = new_work_unit.workground_size


# Function is chosen because it has a quadratic decrease in efficiency as the number of commands on the object
# increases, after the maximum number of commands begins to decrease in efficiency, and its growth rate depends on
# the maximum number of commands.
# sum(1 - ((x-1)^2 / max_groups^2), where x from 1 to groups_count
