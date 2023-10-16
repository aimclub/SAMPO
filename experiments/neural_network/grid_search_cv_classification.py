import os

import pandas as pd
import torch
import torchmetrics.classification
from ray import tune
from ray.air import session, Checkpoint
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sampo.scheduler.selection.neural_net import NeuralNetTrainer, NeuralNet, NeuralNetType
from sampo.scheduler.selection.validation import cross_val_score

path = os.path.join(os.getcwd(), 'datasets/dataset_1000_objs.csv')
dataset = pd.read_csv(path, index_col='index')
for col in dataset.columns[:-1]:
    dataset[col] = dataset[col].apply(lambda x: float(x))

x_tr, x_ts, y_tr, y_ts = train_test_split(dataset.drop(columns=['label']), dataset['label'],
                                          stratify=dataset['label'])

scaler = StandardScaler()
scaler.fit(x_tr)
scaled_dataset = scaler.transform(x_tr)
scaled_dataset = pd.DataFrame(scaled_dataset, columns=x_tr.columns)
x_tr = scaled_dataset

scaler = StandardScaler()
scaler.fit(x_ts)
scaled_dataset = scaler.transform(x_ts)
scaled_dataset = pd.DataFrame(scaled_dataset, columns=x_ts.columns)
x_ts = scaled_dataset

def train(config):
    checkpoint = session.get_checkpoint()
    model = NeuralNet(input_size=13,
                      layer_size=config['layer_size'],
                      layer_count=config['layer_count'],
                      task_type=NeuralNetType.CLASSIFICATION)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scorer = torchmetrics.classification.BinaryAccuracy()
    net = NeuralNetTrainer(model, criterion, optimizer, scorer, 32)
    device = 'cpu'

    x_train, x_test, y_train, y_test = x_tr, x_ts, y_tr, y_ts
    best_trainer: NeuralNetTrainer | None = None
    score, best_loss, best_trainer = cross_val_score(X=x_train,
                                                     y=y_train,
                                                     model=net,
                                                     epochs=config['epochs'],
                                                     folds=config['cv'],
                                                     shuffle=True,
                                                     type_task=NeuralNetType.CLASSIFICATION)
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
        'iters': tune.grid_search([i for i in range(15)]),
        # 'layer_size': tune.grid_search([i for i in range(10, 15)]),
        'layer_size': tune.qrandint(5, 30),
        'layer_count': tune.qrandint(5, 35),
        # 'layer_count': tune.grid_search([i for i in range(5, 9)]),
        'lr': tune.loguniform(1e-7, 1e-1),
        # 'lr': tune.grid_search([0.0001, 0.000055, 0.000075, 0.000425]),
        'epochs': tune.grid_search([100]),
        'cv': tune.grid_search([7])
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

    best_trained_model = NeuralNet(13, layer_size=best_trial.config['layer_size'],
                                   layer_count=best_trial.config['layer_count'],
                                   out_size=2,
                                   task_type=NeuralNetType.CLASSIFICATION)
    best_trained_model.load_state_dict(best_checkpoint_data['model_state_dict'])
    best_trained_optimizer = torch.optim.Adam(best_trained_model.model.parameters(), lr=best_trial.config['lr'])
    best_trained_optimizer.load_state_dict(best_checkpoint_data['optimizer_state_dict'])
    best_trainer = NeuralNetTrainer(best_trained_model, torch.nn.CrossEntropyLoss(), best_trained_optimizer,
                                    scorer=torchmetrics.classification.BinaryAccuracy(), batch_size=32)

    best_model(best_trainer)

    f = open(os.path.join(os.getcwd(), 'checkpoints/best_model_1k_objs_standart_scaler.pth'), 'w')
    f.close()

    best_trainer.save_checkpoint(os.path.join(os.getcwd(), 'checkpoints/'), 'best_model_1k_objs_standart_scaler.pth')

    print(f'Best trial config: {best_trial.config}')
    print(f'Best trial validation loss: {best_trial.last_result["loss"]}')
    print(f'Best trial final validation accuracy: {best_trial.last_result["accuracy"]}')


if __name__ == '__main__':
    main()
