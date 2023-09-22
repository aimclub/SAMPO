from functools import partial
from random import Random
from typing import IO

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sampo.generator.base import SimpleSynthetic
from sampo.scheduler.heft.base import HEFTScheduler, HEFTBetweenScheduler
from sampo.scheduler.multi_agency.block_generator import generate_block_graph, SyntheticBlockGraphType
from sampo.scheduler.multi_agency.multi_agency import Agent, Manager
from sampo.scheduler.selection import metrics
from sampo.scheduler.selection.neural_net import NeuralNetTrainer, NeuralNet
from sampo.scheduler.selection.validation import cross_val_score
from sampo.scheduler.topological.base import TopologicalScheduler

r_seed = 231
p_rand = SimpleSynthetic(rand=r_seed)
rand = Random(r_seed)


def obstruction_getter(i: int):
    return None


def log(message: str, logfile: IO):
    # print(message)
    logfile.write(message + '\n')


if __name__ == '__main__':
    schedulers = [HEFTScheduler(),
                  HEFTBetweenScheduler(),
                  TopologicalScheduler()]

    contractors = [p_rand.contractor(10) for _ in range(len(schedulers))]

    agents = [Agent(f'Agent {i}', schedulers[i % len(schedulers)], [contractor])
              for i, contractor in enumerate(contractors)]
    manager = Manager(agents)

    with open(f'algorithms_2_multi_agency_comparison.txt', 'w') as logfile:
        logger = partial(log, logfile=logfile)

        # Multi agency
        bg = generate_block_graph(SyntheticBlockGraphType.RANDOM, 10, [0, 1, 1], lambda x: (None, 50), 0.5,
                                  rand, obstruction_getter, 2, [3, 4], [3, 4], logger=logger)
        conjuncted = bg.to_work_graph()
        scheduled_blocks = manager.manage_blocks(bg, logger=logger)
        print(f'Multi-agency res: {max(sblock.end_time for sblock in scheduled_blocks.values())}')

        # Neural network
        model = NeuralNet(13, 15, 6, 2)
        trainer = NeuralNetTrainer(model,
                                   torch.nn.CrossEntropyLoss(),
                                   torch.optim.Adam(model.parameters(), lr=0.0065))

        dataset = pd.read_csv('C:/SAMPO/experiments/neural_network/dataset_mod.csv', index_col='index')
        for col in dataset.columns[:-1]:
            dataset[col] = dataset[col].apply(lambda x: float(x))
        scaler = StandardScaler()
        scaler.fit(dataset.drop(columns=['label']))
        scaled_dataset = scaler.transform(dataset.drop(columns=['label']))
        scaled_dataset = pd.DataFrame(scaled_dataset, columns=dataset.drop(columns=['label']).columns)
        train_dataset = pd.concat([scaled_dataset, dataset['label']], axis=1)

        x_tr, x_ts, y_tr, y_ts = train_test_split(train_dataset.drop(columns=['label']), train_dataset['label'],
                                                  random_state=42)

        score, final_model = cross_val_score(X=x_tr,
                                             y=y_tr,
                                             model=trainer,
                                             epochs=1,
                                             folds=2,
                                             shuffle=True,
                                             scorer=accuracy_score)

        final_model.model = final_model.model.double()

        wg_metrics = np.asarray(metrics.encode_graph(conjuncted)).reshape(-1, 1)
        scaler = StandardScaler()
        scaler.fit(wg_metrics)
        scaled_metrics = scaler.transform(wg_metrics)

        scaled_metrics = scaled_metrics[:13]
        scaled_metrics = torch.Tensor(torch.from_numpy(scaled_metrics.reshape(1, -1)))

        print(scaled_metrics)
        result = final_model.predict(scaled_metrics)
        print(result)