import os

import numpy as np
import pandas as pd
import torch
from ray import tune
from ray.air import session, Checkpoint
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

from sampo.scheduler.selection.neural_net import NeuralNetTrainer, NeuralNet
from sampo.scheduler.selection.validation import cross_val_score

# path = 'C:/SAMPO/experiments/neural_network/dataset_mod.csv'
path = os.path.join(os.getcwd(), '../../../experiments/neural_network/datasets/dataset_mod2.csv')
dataset = pd.read_csv(path, index_col='index')
for col in dataset.columns[:-1]:
    dataset[col] = dataset[col].apply(lambda x: float(x))

train_dataset = pd.DataFrame()

for col in dataset.columns[:-1]:
    scaler = Normalizer()
    scaler.fit([dataset[col].values])
    scaled_dataset = scaler.transform([dataset[col].values])
    tmp = np.array(scaled_dataset).reshape(1, -1)[0]
    tmp = pd.Series(tmp)
    frame = {
        col: tmp
    }
    scaled_dataset = pd.DataFrame(frame)
    train_dataset = pd.concat([train_dataset, scaled_dataset], axis=1)

train_dataset = pd.concat([train_dataset, dataset['label']], axis=1)
x_tr, x_ts, y_tr, y_ts = train_test_split(train_dataset.drop(columns=['label']), train_dataset['label'],
                                          random_state=42)


def train(config):
    checkpoint = session.get_checkpoint()
    model = NeuralNet(input_size=8,
                      layer_size=config['layer_size'],
                      layer_count=config['layer_count'])
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    net = NeuralNetTrainer(model, criterion, optimizer)
    device = 'cpu'

    x_train, x_test, y_train, y_test = x_tr, x_ts, y_tr, y_ts
    best_trainer: NeuralNetTrainer | None = None
    score, best_loss, best_trainer = cross_val_score(X=x_train,
                                                     y=y_train,
                                                     model=net,
                                                     epochs=config['epochs'],
                                                     folds=config['cv'],
                                                     shuffle=True,
                                                     scorer=config['scorer'])
    checkpoint_data = {
        'model_state_dict': best_trainer.model.state_dict(),
        'optimizer_state_dict': best_trainer.optimizer.state_dict()
    }
    checkpoint = Checkpoint.from_dict(checkpoint_data)
    session.report({'loss': best_loss, 'accuracy': score}, checkpoint=checkpoint)
    print('accuracy:', score)
    print('Finished Training')
    print('------------------------------------------------------------------------')


def best_model(best_trained_model):
    x_train, x_test, y_train, y_test = x_tr, x_ts, y_tr, y_ts

    predicted = best_trained_model.predict_proba([torch.Tensor(v) for v in x_test.values])
    array = []
    label_test = y_test.to_numpy()
    for i in range(len(predicted)):
        flag = 0 if predicted[i][0] > predicted[i][1] else 1
        array.append(int(flag == label_test[i]))
    print('Best trial test set accuracy:', sum(array) / len(array))


def main():
    config = {
        'layer_size': tune.grid_search([i for i in range(7, 20)]),
        'layer_count': tune.grid_search([i for i in range(7, 20)]),
        'lr': tune.loguniform(1e-7, 1e-2),
        # 'lr': tune.grid_search([0.0001, 0.000055, 0.000075, 0.000425]),
        'epochs': tune.grid_search([100]),
        'cv': tune.grid_search([10]),
        'scorer': tune.grid_search([accuracy_score])
    }

    scheduler = ASHAScheduler(
        metric='loss',
        mode='min',
        max_t=10,
        grace_period=1,
        reduction_factor=2
    )

    reporter = CLIReporter(
        metric_columns=['loss', 'accuracy']
    )

    result = tune.run(
        train,
        resources_per_trial={'cpu': 6},
        config=config,
        num_samples=1,
        scheduler=scheduler,
        progress_reporter=reporter,
    )

    best_trial = result.get_best_trial('loss', 'min', 'last')
    best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    best_checkpoint_data = None
    try:
        best_checkpoint_data = best_checkpoint.to_dict()
    except Exception as e:
        Exception(f'{best_checkpoint} with {e}')

    best_trained_model = NeuralNet(8, layer_size=best_trial.config['layer_size'],
                                   layer_count=best_trial.config['layer_count'],
                                   out_size=2)
    best_trained_model.load_state_dict(best_checkpoint_data['model_state_dict'])
    best_trained_optimizer = torch.optim.Adam(best_trained_model.model.parameters(), lr=best_trial.config['lr'])
    best_trained_optimizer.load_state_dict(best_checkpoint_data['optimizer_state_dict'])
    best_trainer = NeuralNetTrainer(best_trained_model, torch.nn.CrossEntropyLoss(), best_trained_optimizer)

    best_model(best_trainer)

    # f = open(f'C:/SAMPO/experiments/neural_network/checkpoints/best_model_10000_objs.pth', "w")
    f = open(os.path.join(os.getcwd(), '../../../experiments/neural_network/checkpoints/best_model_10000_objs.pth'), 'w')
    f.close()

    # best_trainer.save_checkpoint(f'C:/SAMPO/experiments/neural_network/checkpoints/')
    best_trainer.save_checkpoint(os.path.join(os.getcwd(), '../../../experiments/neural_network/checkpoints/'), 'best_model_10000_objs.pth')

    print(f'Best trial config: {best_trial.config}')
    print(f'Best trial validation loss: {best_trial.last_result["loss"]}')
    print(f'Best trial final validation accuracy: {best_trial.last_result["accuracy"]}')


if __name__ == '__main__':
    main()
