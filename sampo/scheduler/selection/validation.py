import pandas as pd
import torch
from sklearn.model_selection import KFold

from sampo.scheduler.selection.metrics import one_hot_encode
from sampo.scheduler.selection.neural_net import NeuralNetTrainer, NeuralNetType


def cross_val_score(X: pd.DataFrame,
                    y: pd.DataFrame,
                    model: NeuralNetTrainer,
                    epochs: int = 100,
                    folds: int = 5,
                    shuffle: bool = False,
                    random_state: int | None = None,
                    type_task: NeuralNetType = None) \
        -> tuple[list[float] | dict[str, float], int, NeuralNetTrainer]:
    """
    Evaluate metric by cross-validation and also record score times.

    :param X: The data to fit (DataFrame).
    :param y: The column that contains target variable to try to predict.
    :param model: The object (inherited from nn.Module).
    :param epochs: Number of epochs during which the model is trained.
    :param folds: Training dataset is split on 'folds' folds for cross-validation.
    :param shuffle: 'True' if the splitting dataset on folds should be random, 'False' - otherwise.
    :param random_state:
    :return: List of scores that correspond to each validation fold.
    """
    kf = KFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
    scores = 0
    best_loss = 0
    best_trainer: NeuralNetTrainer | None = None
    transform_data = lambda x: one_hot_encode(x, 2) if type_task == NeuralNetType.CLASSIFICATION else x

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        train_tensor = torch.stack([torch.Tensor(v) for v in X.iloc[train_idx, :].values])
        train_target_tensor = torch.stack([torch.Tensor(transform_data(v)) for v in y.iloc[train_idx].values])
        test_tensor = torch.stack([torch.Tensor(v) for v in X.iloc[test_idx, :].values])
        test_target_tensor = torch.stack([torch.Tensor(transform_data(v)) for v in y.iloc[test_idx].values])

        model.fit(train_tensor, train_target_tensor, epochs)
        tmp_score, loss = model.validate(test_tensor, test_target_tensor)
        if tmp_score > scores:
            best_trainer = model
            scores = max(scores, tmp_score)
            best_loss = loss

    return scores, best_loss, best_trainer
