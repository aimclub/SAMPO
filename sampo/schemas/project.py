from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph
from sampo.schemas.schedule import Schedule
from sampo.schemas.serializable import AutoJSONSerializable


class ScheduledProject(AutoJSONSerializable['ScheduledProject']):
    def __init__(self, wg: WorkGraph, contractors: list[Contractor], schedule: Schedule):
        self.schedule = schedule
        self.wg = wg
        self.contractors = contractors
