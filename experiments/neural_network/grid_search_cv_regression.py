import os

import pandas as pd
import torch
import torchmetrics
from ray import tune
from ray.air import session, Checkpoint
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import train_test_split

from sampo.scheduler.selection.neural_net import NeuralNetTrainer, NeuralNet, NeuralNetType
from sampo.scheduler.selection.validation import cross_val_score

path = os.path.join(os.getcwd(), 'datasets/wg_contractor_dataset_100000_objs.csv')
dataset = pd.read_csv(path, index_col='index')
for col in dataset.columns[:-1]:
    dataset[col] = dataset[col].apply(lambda x: float(x))

dataset['label'] = dataset['label'].apply(lambda x: [int(i) for i in x.split()])

x_tr, x_ts, y_tr, y_ts = train_test_split(dataset.drop(columns=['label']), dataset['label'])


def train(config: dict) -> None:
    """
    Training function for ray tune process

    :param config: search space of the model's hyperparameters
    """
    model = NeuralNet(input_size=13,
                      layer_size=config['layer_size'],
                      layer_count=config['layer_count'],
                      out_size=6,
                      task_type=NeuralNetType.REGRESSION)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scorer = torchmetrics.regression.MeanSquaredError()
    net = NeuralNetTrainer(model, criterion, optimizer, scorer, 32)

    x_train, x_test, y_train, y_test = x_tr, x_ts, y_tr, y_ts
    score, best_loss, best_trainer = cross_val_score(X=x_train,
                                                     y=y_train,
                                                     model=net,
                                                     epochs=config['epochs'],
                                                     folds=config['cv'],
                                                     shuffle=True,
                                                     type_task=NeuralNetType.REGRESSION)
    # Checkpoint - structure of the saved model
    checkpoint_data = {
        'model_state_dict': best_trainer.model.state_dict(),
        'optimizer_state_dict': best_trainer.optimizer.state_dict()
    }
    # Report loss and score immediate metrics
    checkpoint = Checkpoint.from_dict(checkpoint_data)
    session.report({'loss': best_loss, 'score': score}, checkpoint=checkpoint)
    print('MSE:', score)
    print('Finished Training')
    print('------------------------------------------------------------------------')


def best_test_score(best_trained_model: NeuralNetTrainer) -> None:
    x_train, x_test, y_train, y_test = x_tr, x_ts, y_tr, y_ts

    predicted = best_trained_model.predict([torch.Tensor(v) for v in x_test.values])
    array = []
    label_test = y_test.to_numpy()
    for i in range(len(predicted)):
        array.append(sum((predicted[i] - label_test[i]) ** 2))
    print('Best trial test set RMSE:', (sum(array) / len(array)) ** (1 / 2))


def main():
    # Dict represents the search space by model's hyperparameters
    config = {
        'iters': tune.grid_search([i for i in range(1)]),
        'layer_size': tune.qrandint(5, 30),
        'layer_count': tune.qrandint(5, 35),
        'lr': tune.loguniform(1e-4, 1e-1),
        'epochs': tune.grid_search([2]),
        'cv': tune.grid_search([2]),
    }

    scheduler = ASHAScheduler(
        metric='loss',
        mode='min',
        max_t=10,
        grace_period=1,
        reduction_factor=2
    )

    reporter = CLIReporter(
        metric_columns=['loss', 'score']
    )

    # Here you can change the number of CPU's you want to use for tuning
    result = tune.run(
        train,
        resources_per_trial={'cpu': 6},
        config=config,
        num_samples=1,
        scheduler=scheduler,
        progress_reporter=reporter,
    )

    # Receive the trial with the best results
    best_trial = result.get_best_trial('loss', 'min', 'last')
    best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    best_checkpoint_data = None
    try:
        best_checkpoint_data = best_checkpoint.to_dict()
    except Exception as e:
        Exception(f'{best_checkpoint} with {e}')

    # Construct the best trainer based on the best checkpoint data
    best_trained_model = NeuralNet(13, layer_size=best_trial.config['layer_size'],
                                   layer_count=best_trial.config['layer_count'],
                                   out_size=6)
    best_trained_model.load_state_dict(best_checkpoint_data['model_state_dict'])
    best_trained_optimizer = torch.optim.Adam(best_trained_model.model.parameters(), lr=best_trial.config['lr'])
    best_trained_optimizer.load_state_dict(best_checkpoint_data['optimizer_state_dict'])
    scorer = torchmetrics.regression.MeanSquaredError()
    best_trainer = NeuralNetTrainer(best_trained_model, torch.nn.CrossEntropyLoss(), best_trained_optimizer, scorer, 32)

    # Print score of the best trainer on test sample
    best_test_score(best_trainer)

    best_trainer.save_checkpoint(os.path.join(os.getcwd(), 'checkpoints/'), 'best_model_wg_and_contractor.pth')

    print(f'Best trial config: {best_trial.config}')
    print(f'Best trial validation loss: {best_trial.last_result["loss"]}')
    print(f'Best trial final validation accuracy: {best_trial.last_result["score"]}')


if __name__ == '__main__':
    main()
