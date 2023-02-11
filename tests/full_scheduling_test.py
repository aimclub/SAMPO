from sampo.generator import SimpleSynthetic
from sampo.generator.pipeline.types import SyntheticGraphType
from sampo.scheduler.base import SchedulerType
from sampo.scheduler.generate import generate_schedule
from sampo.structurator import graph_restructuring


def test_schedule_synthetic(setup_contractors):
    ss = SimpleSynthetic(rand=231)
    wg = ss.work_graph(SyntheticGraphType.General, 100, 200)
    wg = graph_restructuring(wg, use_lag_edge_optimization=True)

    for scheduler_type in list(SchedulerType):
        try:
            schedule = generate_schedule(scheduler_type, None, wg, setup_contractors, validate_schedule=True)
        except AssertionError as e:
            raise AssertionError(f'Scheduler {scheduler_type} failed validation', e)

        assert not schedule.execution_time.is_inf(), f'Scheduling failed on {scheduler_type.name}'
