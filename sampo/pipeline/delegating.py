from sampo.scheduler.generic import GenericScheduler, PRIORITIZATION_F, RESOURCE_OPTIMIZE_F


class DelegatingScheduler(GenericScheduler):
    """
    The layer between Generic Scheduler and end Scheduler.
    It's needed to change parametrization functions received from end Scheduler in a simple way.
    """

    def __init__(self, delegate: GenericScheduler):
        super().__init__(delegate.scheduler_type,
                         delegate.resource_optimizer,
                         delegate._timeline_type,
                         self.delegate_prioritization(delegate.prioritization),
                         self.delegate_stochastic_prioritization(delegate.stochastic_prioritization),
                         self.delegate_optimize_resources(delegate.optimize_resources),
                         delegate.work_estimator)

    # noinspection PyMethodMayBeStatic
    def delegate_prioritization(self, prioritization) -> PRIORITIZATION_F:
        return prioritization

    # noinspection PyMethodMayBeStatic
    def delegate_stochastic_prioritization(self, prioritization) -> PRIORITIZATION_F:
        return prioritization

    # noinspection PyMethodMayBeStatic
    def delegate_optimize_resources(self, optimize_resources) -> RESOURCE_OPTIMIZE_F:
        return optimize_resources
