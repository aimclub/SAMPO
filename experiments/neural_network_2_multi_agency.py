import os
import pickle
import time
from random import Random

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from sampo.generator.base import SimpleSynthetic
from sampo.scheduler.heft.base import HEFTScheduler, HEFTBetweenScheduler
from sampo.scheduler.multi_agency.block_generator import generate_block_graph, SyntheticBlockGraphType
from sampo.scheduler.multi_agency.multi_agency import Agent, Manager
from sampo.scheduler.selection import metrics
from sampo.scheduler.selection.neural_net import NeuralNetTrainer, NeuralNet

r_seed = 231
p_rand = SimpleSynthetic(rand=r_seed)
rand = Random(r_seed)

ma_time = 0.0
net_time = 0.0

def obstruction_getter(i: int):
    return None


def run_interation(blocks_num, graph_size) -> None:
    global ma_time, net_time

    schedulers = [HEFTScheduler(),
                  HEFTBetweenScheduler()]

    contractors = [p_rand.contractor(10) for _ in range(len(schedulers))]

    agents = [Agent(f'Agent {i}', schedulers[i % len(schedulers)], [contractor])
              for i, contractor in enumerate(contractors)]
    manager = Manager(agents)

    # Build block graph

    bg = generate_block_graph(SyntheticBlockGraphType.RANDOM, blocks_num, [0, 1, 1], lambda x: (None, graph_size), 0.5,
                              rand, obstruction_getter, 2, [3, 4], [3, 4])
    conjuncted = bg.to_work_graph()

    # Multi-agency

    start = time.time()
    scheduled_blocks = manager.manage_blocks(bg)
    finish = time.time()
    print(f'Multi-agency res: {max(sblock.end_time for sblock in scheduled_blocks.values())}')
    ma_time += (finish - start)

    # Neural Network

    scaler = StandardScaler()
    with open(os.path.join(os.getcwd(), 'neural_network/checkpoints/scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    net = NeuralNet(13, 14, 7, 2)

    best_trainer = NeuralNetTrainer(net, torch.nn.CrossEntropyLoss(),
                                    torch.optim.Adam(net.parameters(), lr=5.338346838224648e-05))
    best_trainer.load_checkpoint(os.path.join(
        os.getcwd(), 'neural_network/checkpoints/best_model_1k_objs_standart_scaler.pth'))

    best_trainer.model = best_trainer.model.double()

    start = time.time()
    encoded_wg = np.asarray(metrics.encode_graph(conjuncted)).reshape(1, -1)
    scaled_metrics = scaler.transform(encoded_wg)

    scaled_metrics = torch.Tensor(torch.from_numpy(scaled_metrics.reshape(1, -1)))

    result = best_trainer.predict(scaled_metrics)
    best_schedule = schedulers[int(result[0])]
    scheduler = best_schedule.schedule(conjuncted, contractors[int(result[0])])
    finish = time.time()

    print(f'Neural network results: {scheduler.execution_time}')
    net_time += (finish - start)


if __name__ == '__main__':

    graph_size = 100
    block_num = 5
    iterations = 100

    for i in range(iterations):
        print(f'\nIteration {i}')
        run_interation(block_num, graph_size)

    avg_ma_time = ma_time / iterations
    avg_net_time = net_time / iterations

    print(f'The average execution time of multi-agency (MA) is: {avg_ma_time}')
    print(f'The average execution time of neural network (NN) is: {avg_net_time}')
    print(f'The ratio of MA time to NN time is: {avg_ma_time / avg_net_time}')
