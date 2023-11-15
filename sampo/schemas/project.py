from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph
from sampo.schemas.schedule import Schedule
from sampo.schemas.serializable import AutoJSONSerializable
from sampo.utilities.schedule import fix_split_tasks, offset_schedule
from sampo.utilities.serializers import custom_serializer


class ScheduledProject(AutoJSONSerializable['ScheduledProject']):
    def __init__(self, wg: WorkGraph, contractors: list[Contractor], schedule: Schedule):
        self.schedule = schedule
        self.schedule_df = fix_split_tasks(schedule.full_schedule_df)
        self.wg = wg
        self.contractors = contractors

    @custom_serializer('contractors')
    def serialize_contractors(self, value):
        return [v._serialize() for v in value]

    @classmethod
    @custom_serializer('contractors', deserializer=True)
    def deserialize_equipment(cls, value):
        return [Contractor._deserialize(v) for v in value]

    def set_start_date(self, start_date: str) -> 'ScheduledProject':
        self.schedule_df = offset_schedule(self.schedule_df, start_date)
        return self
