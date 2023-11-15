from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph
from sampo.schemas.schedule import Schedule
from sampo.schemas.serializable import AutoJSONSerializable
from sampo.utilities.serializers import custom_serializer


class ScheduledProject(AutoJSONSerializable['ScheduledProject']):

    ignored_fields = ['raw_schedule', 'raw_wg']

    def __init__(self, wg: WorkGraph, raw_wg: WorkGraph, contractors: list[Contractor], schedule: Schedule):
        self.schedule = schedule.unite_stages()
        self.raw_schedule = schedule
        self.wg = wg
        self.raw_wg = raw_wg
        self.contractors = contractors

    @custom_serializer('contractors')
    def serialize_contractors(self, value):
        return [v._serialize() for v in value]

    @classmethod
    @custom_serializer('contractors', deserializer=True)
    def deserialize_equipment(cls, value):
        return [Contractor._deserialize(v) for v in value]
