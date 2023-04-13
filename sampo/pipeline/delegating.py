from sampo.scheduler.generic import GenericScheduler


class DelegatingScheduler(GenericScheduler):

    def __init__(self, delegate: GenericScheduler):
        super().__init__(delegate.scheduler_type,
                         delegate.resource_optimizer,
                         delegate._timeline_type,
                         self.delegate_prioritization(delegate.prioritization),
                         self.delegate_optimize_resources(delegate.optimize_resources),
                         delegate.work_estimator)

    # noinspection PyMethodMayBeStatic
    def delegate_prioritization(self, prioritization):
        return prioritization

    # noinspection PyMethodMayBeStatic
    def delegate_optimize_resources(self, optimize_resources):
        return optimize_resources
