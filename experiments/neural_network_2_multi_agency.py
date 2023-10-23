import os
import time
from multiprocessing import Pool
from random import Random

import numpy as np
import torch
import torchmetrics
from sklearn.preprocessing import StandardScaler

from sampo.generator.base import SimpleSynthetic
from sampo.scheduler.heft.base import HEFTScheduler, HEFTBetweenScheduler
from sampo.scheduler.multi_agency.block_generator import generate_block_graph, SyntheticBlockGraphType
from sampo.scheduler.multi_agency.multi_agency import Agent, Manager, NeuralManager
from sampo.scheduler.selection.metrics import encode_graph
from sampo.scheduler.selection.neural_net import NeuralNetTrainer, NeuralNet, NeuralNetType

p_rand = SimpleSynthetic()
rand = Random()

ma_time = 0.0
net_time = 0.0


def obstruction_getter(i: int):
    return None


def run_interation(iter: int, blocks_num: int = 10, graph_size: int = 200) -> None:
    global ma_time, net_time
    print(f'Iteration: {iter}')

    scaler = StandardScaler()

    schedulers = [HEFTScheduler(),
                  HEFTBetweenScheduler()]

    contractors = [p_rand.contractor(10) for _ in range(len(schedulers))]

    agents = [Agent(f'Agent {i}', schedulers[i % len(schedulers)], [contractor])
              for i, contractor in enumerate(contractors)]
    agents2 = [Agent(f'Agent {i}', schedulers[i % len(schedulers)], [contractor])
               for i, contractor in enumerate(contractors)]
    manager = Manager(agents)

    # Build block graph
    # --------------------------------------------------------------------------------------------------------------

    bg = generate_block_graph(SyntheticBlockGraphType.RANDOM, blocks_num, [0, 1, 1], lambda x: (None, graph_size), 0.5,
                              rand, obstruction_getter, 2, [3, 4], [3, 4])
    bg1 = bg.__copy__()
    blocks = []
    encoding_blocks = []
    for i, block in enumerate(bg1.toposort()):
        blocks.append(block)
        encode = encode_graph(block.wg)
        encode = [np.double(x) for x in encode]
        encoding_blocks.append(encode)

    encoding_blocks = scaler.fit_transform(encoding_blocks)
    encoding_blocks_tensor = []
    for block in encoding_blocks:
        encoding_blocks_tensor.append(torch.Tensor(block))

    # Multi-agency scheduling process
    # --------------------------------------------------------------------------------------------------------------

    start = time.time()
    scheduled_blocks = manager.manage_blocks(bg)
    finish = time.time()
    print(f'Multi-agency res: {max(sblock.end_time for sblock in scheduled_blocks.values())}')
    ma_time += (finish - start)

    # Neural Manager scheduling process
    # ---------------------------------------------------------------------------------------------------------------

    net = NeuralNet(13, 15, 7, 2, NeuralNetType.CLASSIFICATION)
    scorer = torchmetrics.classification.BinaryAccuracy()
    best_trainer = NeuralNetTrainer(net, torch.nn.CrossEntropyLoss(),
                                    torch.optim.Adam(net.parameters(), lr=0.0001687954784838078),
                                    scorer, 1)
    best_trainer.load_checkpoint(os.path.join(
        os.getcwd(), 'neural_network/checkpoints/best_model_10k_algo.pth'))

    net_contractor = NeuralNet(13, 13, 24, 6, NeuralNetType.REGRESSION)
    scorer = torchmetrics.regression.MeanSquaredError()
    best_trainer_contractor = NeuralNetTrainer(net_contractor, torch.nn.MSELoss(),
                                               torch.optim.Adam(net_contractor.parameters(), lr=0.0003584150994379109),
                                               scorer, 1)
    best_trainer_contractor.load_checkpoint(os.path.join(
        os.getcwd(), 'neural_network/checkpoints/best_model_wg_and_contractor.pth'))

    neural_manager = NeuralManager(agents2,
                                   best_trainer,
                                   best_trainer_contractor,
                                   [HEFTScheduler(), HEFTBetweenScheduler()],
                                   blocks=blocks,
                                   encoding_blocks=encoding_blocks_tensor)

    start = time.time()
    scheduled_blocks = neural_manager.manage_blocks()
    finish = time.time()

    net_time += (finish - start)
    print(f'Neural Multi-agency res: {max(sblock.end_time for sblock in scheduled_blocks.values())}')
    print(f'Times of systems:')
    print(f'Multi-agency time is {ma_time} and neural network is {net_time}')
    del bg1
    del bg


if __name__ == '__main__':

    iterations = 10
    iters = []
    for i in range(0, iterations, iterations // 4):
        task = [i] * (iterations // 4)
        iters.append(task)

    result = []
    with Pool() as pool:
        for task in iters:
            result.extend(pool.map(run_interation, task))

    avg_ma_time = ma_time / iterations
    avg_net_time = net_time / iterations

    print(f'The average execution time of multi-agency (MA) is: {avg_ma_time}')
    print(f'The average execution time of neural network (NN) is: {avg_net_time}')
    print(f'The ratio of MA time to NN time is: {avg_ma_time / avg_net_time}')
