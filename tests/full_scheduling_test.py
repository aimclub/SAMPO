from sampo.generator import SimpleSynthetic
from sampo.generator.types import SyntheticGraphType
from sampo.scheduler.base import SchedulerType
from sampo.scheduler.generate import generate_schedule
from sampo.schemas.time import Time


def test_schedule_synthetic(setup_contractors):
    ss = SimpleSynthetic(rand=231)
    wg = ss.work_graph(SyntheticGraphType.General, 100, 200)

    for scheduler_type in list(SchedulerType):
        schedule = generate_schedule(scheduler_type, None, wg, setup_contractors, validate_schedule=True)

        assert schedule.execution_time != Time.inf(), f'Scheduling failed on {scheduler_type.name}'
