from typing import Callable

import pandas as pd
import torch
from sklearn.model_selection import KFold

from sampo.scheduler.selection.metrics import one_hot_encode, one_hot_decode
from sampo.scheduler.selection.neural_net import NeuralNet


def cross_val_score(train_dataset: pd.DataFrame,
                    target_column: str,
                    model: NeuralNet,
                    epochs: int = 100,
                    folds: int = 5,
                    shuffle: bool = False,
                    random_state: int | None = None,
                    scorer: dict[str, Callable[[list, list], dict[str, float]]] | Callable[[list, list], float] = None) \
        -> list[float] | dict[str, float]:
    """
    Evaluate metric by cross-validation and also record score times.

    :param train_dataset: The data to fit (DataFrame).
    :param target_column: The column, that contains target variable to try to predict.
    :param model: The object (inherited from nn.Module).
    :param epochs: Number of epochs during which the model is trained.
    :param folds: Training dataset is splited on 'folds' folds for cross-validation.
    :param shuffle: 'True' if the splitting dataset on folds should be random, 'False' - otherwise.
    :param random_state:
    :param scorer: Dictionary of callable scorers or just function.
    :return: List of scores that correspond to each validation fold.
    """
    kf = KFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(train_dataset)):
        train_tensor = [torch.Tensor(v) for v in train_dataset.iloc[train_idx, :].drop(columns=[target_column]).values]
        train_target_tensor = [torch.Tensor(one_hot_encode(v, 2)) for v in train_dataset.iloc[train_idx, list(train_dataset.columns).index('label')].values]
        test_tensor = [torch.Tensor(v) for v in train_dataset.iloc[test_idx, :].drop(columns=[target_column]).values]
        test_target_tensor = [torch.Tensor(one_hot_encode(v, 2)) for v in train_dataset.iloc[test_idx, list(train_dataset.columns).index('label')].values]

        model.fit(train_tensor, train_target_tensor, epochs)
        predicted_target_tensor = model.predict(test_tensor)
        predicted_target_tensor = [one_hot_decode(elem) for elem in predicted_target_tensor]
        test_target_tensor = [one_hot_decode(elem) for elem in test_target_tensor]

        if isinstance(scorer, dict):
            for name, scorer in scorer.items():
                scores.append(scorer(test_target_tensor, predicted_target_tensor))
        else:
            scores.append(scorer(test_target_tensor, predicted_target_tensor))

    return scores
