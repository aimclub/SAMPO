from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Tuple, Optional, List, Union, Dict, Callable

import numpy as np
import scipy.optimize
from numpy import ndarray

from sampo.metrics.resources_in_time.base import ResourceOptimizer, init_borders, prepare_answer, is_resources_good
from sampo.scheduler.base import Scheduler
from sampo.schemas.contractor import Contractor, WorkerContractorPool
from sampo.schemas.graph import WorkGraph
from sampo.schemas.time import Time

MAX_NEWTON_CG_ITERATIONS: int = 48
NEWTON_CG_UNDEFINED_INF: float = 1e+5


@dataclass
class NewtonCGOptimizer(ResourceOptimizer):
    """
    Unsuccessful resource optimizer. Should be reimagined
    """
    scheduler: Scheduler

    gradient_scale: Optional[Union[float, int]] = 10e+5
    worker_weights: Optional[ndarray] = field(default_factory=lambda: defaultdict(lambda: 1))
    worker_factor: Optional[int] = 100
    max_workers: Optional[int] = 500

    _opt_result: ndarray = field(init=False)
    _nfev: int = field(init=False, default=0)  # Number of Function EValuations
    _worker_weights: ndarray = field(init=False)
    _opt_bounds: Tuple[ndarray, ndarray] = field(init=False)
    _margin: int = field(init=False, default=None)

    _can_satisfy_deadline: Callable[[ndarray], bool] = field(init=False, default=None)

    __last: ndarray = field(init=False, default=None)

    def optimize(self, wg: WorkGraph,
                 deadline: Time,
                 worker_weights: Dict[str, Union[int, float]] = None,
                 agents: Optional[WorkerContractorPool] = None,
                 dry_resources: Optional[bool] = False) -> Union[Tuple[Contractor, Time], Tuple[None, None]]:
        self._nfev = 0
        left_counts, right_counts, agent_names = init_borders(wg, self.scheduler, deadline,
                                                              self.worker_factor, self.max_workers, agents)
        self._init_worker_weights(agent_names, worker_weights or self.worker_weights)

        assert left_counts is not None, 'Deadline cannot be satisfied by any set of resources'

        if right_counts is None:
            # TODO: uncomment
            # return prepare_answer(left_counts, agent_names, wg, self.scheduler, False)
            # TODO: remove
            right_counts = left_counts * self.worker_factor

        self._opt_bounds = left_counts, right_counts

        agents = agents or agent_names
        self._can_satisfy_deadline = partial(is_resources_good,
                                             wg=wg, agent_names=agents, scheduler=self.scheduler, deadline=deadline)

        x0, x1 = self._init_x0_x1(left_counts, right_counts)

        opt = scipy.optimize.newton(func=self._target_function,
                                    x0=x0,
                                    x1=x1,
                                    tol=1 / self.gradient_scale,
                                    maxiter=MAX_NEWTON_CG_ITERATIONS)
        self._opt_result = (opt * self.gradient_scale).astype(int)
        print(f'NewtonCG resource optimizer: {self._nfev} fev; {self._opt_result.mean()} mean res')

        return prepare_answer(self._opt_result, agents, wg, self.scheduler, dry_resources)

    def _is_resource_good(self, x: ndarray):
        # TODO: simplify code, when tested
        ib = self._in_bounds(x)
        if ib:
            sd = self._can_satisfy_deadline(agent_counts=x)
            return sd
        return False

    def _init_x0_x1(self, lower_bounds: ndarray, upper_bounds: ndarray):
        # TODO: refactor initializing logics
        result = np.amax((lower_bounds, (lower_bounds + upper_bounds) / 1.75), axis=0)
        ib = self._in_bounds(result)
        rg = self._can_satisfy_deadline(agent_counts=result)
        while not rg and ib:
            result = np.amin((result + 1, upper_bounds), axis=0).astype(int)
            ib = self._in_bounds(result)
            rg = self._can_satisfy_deadline(agent_counts=result)
        return (result + 1) / self.gradient_scale, result / self.gradient_scale

    def _init_worker_weights(self, agent_names: List[str], agent_weights: Dict[str, Union[float, int]]):
        self._worker_weights = np.array([agent_weights[name] for name in agent_names])

    def _in_bounds(self, x: ndarray):
        lower, upper = self._opt_bounds
        return all(x >= lower) and all(x <= upper)

    def _loss(self, x: ndarray):
        lower, _ = self._opt_bounds
        power = 2 if any(x < 0) else 3
        loss = np.abs(x - lower) * self._worker_weights
        if not self._is_resource_good(x):
            return (((loss + 1) * self._worker_weights) ** power).sum()
        return (loss * self._worker_weights).sum()

    def _target_function(self, x: ndarray):
        # TODO: This function doesn't produce proper gradient values and causes fluctuation and non convergence
        scaled_x = (x * self.gradient_scale).astype(int)
        self._nfev += 1

        # result = (scaled_x * self._worker_weights).sum()
        r = self._loss(scaled_x)
        return r

