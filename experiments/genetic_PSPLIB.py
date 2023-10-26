import uuid

import numpy as np

from sampo.generator.pipeline.project import get_start_stage
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.requirements import WorkerReq
from sampo.schemas.time import Time
from sampo.schemas.works import WorkUnit

instances = [30, 60, 90, 120]
workers = ['R1', 'R2', 'R3', 'R4']

for wg_size in instances:
    dataset = np.load(f'psplib_datasets/problems_{str(wg_size)}.npy', allow_pickle=True)
    for wg_info in dataset:
        adj_matrix, res_matrix, contractor_info = wg_info

        nodes = []
        for i in range(wg_size):
            for j in range(i, wg_size):
                if adj_matrix[i][j] == 1:

                    worker_reqs = [WorkerReq(workers[idx_req-1], Time(0), res_matrix[i][idx_req], res_matrix[i][idx_req], workers[idx_req-1]) for idx_req in range(1, 5)]

                    work_unit = WorkUnit(str(uuid.uuid4()), str(i), worker_reqs)
                    node = GraphNode(work_unit, [])

        start_node = get_start_stage()
        wg = WorkGraph()