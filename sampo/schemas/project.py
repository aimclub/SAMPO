from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.schedule import Schedule
from sampo.schemas.serializable import AutoJSONSerializable

from sampo.utilities.serializers import custom_serializer


class ScheduledProject(AutoJSONSerializable['ScheduledProject']):

    ignored_fields = ['raw_schedule', 'raw_wg']

    def __init__(self, wg: WorkGraph, raw_wg: WorkGraph, contractors: list[Contractor], schedule: Schedule):
        """
        Contains schedule and all information about its creation
        :param wg: the original work graph
        :param raw_wg: restructured work graph, which was given directly to scheduler to produce this schedule
        :param contractors: list of contractors
        :param schedule: the raw schedule received directly from scheduler
        """
        # the final variant of schedule, without any technical issues
        self.schedule = schedule.unite_stages()
        # the raw schedule, with inseparables
        self.raw_schedule = schedule
        # the original work graph
        self.wg = wg
        # internally processed work graph, with inseparables
        self.raw_wg = raw_wg
        self.contractors = contractors

    @custom_serializer('contractors')
    def serialize_contractors(self, value) -> list:
        return [v._serialize() for v in value]

    @classmethod
    @custom_serializer('contractors', deserializer=True)
    def deserialize_equipment(cls, value) -> list[Contractor]:
        return [Contractor._deserialize(v) for v in value]

    def critical_path(self) -> list[GraphNode]:
        from sampo.scheduler.utils.critical_path import critical_path_schedule_lag_optimized
        return critical_path_schedule_lag_optimized(self.raw_wg.nodes,
                                                    self.raw_schedule.to_schedule_work_dict,
                                                    self.wg.nodes)
