import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, KFold
from torch import nn

from sampo.schemas.graph import WorkGraph


class NeuralNet(nn.Module):
    def __init__(self, input_size: int, layer_size: int, layer_count: int, out_size: int, learning_rate=0.001):
        super(NeuralNet, self).__init__()
        self._layers_count = layer_count
        self._linear0 = torch.nn.Linear(input_size, layer_size)
        for i in range(layer_count - 2):
            self.__dict__[f'_linear{i + 1}'] = torch.nn.Linear(layer_size, layer_size)
        self.__dict__[f'_linear{layer_count - 1}'] = torch.nn.Linear(layer_size, out_size)
        self._learning_rate = learning_rate
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        x = self._linear0(x)
        for i in range(1, self._layers_count):
            linear = self.__dict__[f'_linear{i}']
            x = F.relu(x)
            x = linear(x)
        x = F.softmax(x, dim=0)
        return x

    def fit(self, x, y, epochs=10):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)

        # Train the model
        total_step = len(x)
        for epoch in range(epochs):
            for i, (image, label) in enumerate(zip(x, y)):
                # Move tensors to the configured device
                image = image.to(self._device)
                label = label.to(self._device)

                # Forward pass
                outputs = self(image)
                loss = criterion(outputs, label.float())

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, epochs, i + 1, total_step, loss.item()))

    def predict(self, x: list) -> torch.tensor:
        result = []
        with torch.no_grad():
            for image in x:
                image = image.to(self._device)
                outputs = self(image)
                _, predicted = torch.max(outputs.data, 0)
                result.append(torch.tensor(one_hot_encode(predicted, 3)))
        return result

    def test_accuracy(self, x, y) -> float:
        # Test the model
        # In the test phase, don't need to compute gradients (for memory efficiency)
        correct = 0
        total = 0
        loss_test = 0
        predicted = self.predict(x)
        criterion = torch.nn.CrossEntropyLoss()

        for image, label, predicted_label in zip(x, y, predicted):
            total += 1
            correct += predicted_label.tolist() == label.tolist()
            loss_test = criterion(predicted_label, label)

        accuracy = 100.0 * correct / total
        loss = loss_test / len(y)

        print(f'Test set: Average loss: {loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
        return accuracy


def one_hot_encode(v, max_v):
    res = [float(0) for _ in range(max_v)]
    res[v] = float(1)
    return res


def load_dataset(filename: str) -> tuple[list, list, list, list]:
    df = pd.read_csv(filename)
    df.reset_index()
    x_train, x_test, y_train, y_test = train_test_split(df.drop('label', axis=1).to_numpy(), df['label'].to_numpy(),
                                                        stratify=df['label'].to_numpy())
    return [torch.Tensor(v) for v in x_train[:, 1:]],\
           [torch.Tensor(v) for v in x_test[:, 1:]], \
           [torch.Tensor(one_hot_encode(v, 3)) for v in y_train], \
           [torch.Tensor(one_hot_encode(v, 3)) for v in y_test]


def cross_val_score(train_dataset: pd.DataFrame, target_column: str, model: NeuralNet, epochs: int = 100, folds: int = 5,
                    shuffle: bool = False, random_state: int | None = None) -> list[float]:
    """
    Evaluate metric by cross-validation and also record score times.

    :param train_dataset: The data to fit (DataFrame)
    :param target_column: The column, that contains target variable to try to predict
    :param model: The object (inherited from nn.Module)
    :param epochs: Number of epochs during which the model is trained
    :param folds: Training dataset is splited on 'folds' folds for cross-validation
    :param shuffle: 'True' if the splitting dataset on folds should be random, 'False' - otherwise
    :param random_state:
    :return: List of scores that correspond to each validation fold
    """
    kf = KFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(train_dataset)):
        train_tensor = [torch.Tensor(v) for v in train_dataset.iloc[train_idx, :].drop(columns=[target_column]).values]
        train_target_tensor = [torch.Tensor(one_hot_encode(v, 3)) for v in train_dataset.loc[train_idx, target_column].values]
        test_tensor = [torch.Tensor(v) for v in train_dataset.iloc[test_idx, :].drop(columns=[target_column]).values]
        test_target_tensor = [torch.Tensor(one_hot_encode(v, 3)) for v in train_dataset.loc[test_idx, target_column].values]

        model.fit(train_tensor, train_target_tensor, epochs)

        scores.append(model.test_accuracy(test_tensor, test_target_tensor))

    return scores


def metric_resource_constrainedness(wg: WorkGraph) -> list[float]:
    """
    The resource constrainedness of a resource type k is defined as  the average number of units requested by all
    activities divided by the capacity of the resource type

    :param wg: Work graph
    :return: List of RC coefficients for each resource type
    """
    rc_coefs = []
    resource_dict = {}

    for node in wg.nodes:
        for req in node.work_unit.worker_reqs:
            resource_dict[req.kind] = {'activity_amount': 1, 'volume': 0}

    for node in wg.nodes:
        for req in node.work_unit.worker_reqs:
            resource_dict[req.kind]['activity_amount'] += 1
            resource_dict[req.kind]['volume'] += req.volume

    for name, value in resource_dict.items():
        rc_coefs.append(value['activity_amount'] / value['volume'])

    return rc_coefs


def metric_graph_parallelism_degree(wg: WorkGraph) -> list[float]:
    parallelism_degree = []
    current_node = wg.start

    stack = [current_node]
    while stack:
        tmp_stack = []
        parallelism_coef = 0
        for node in stack:
            parallelism_coef += 1
            for child in node.children:
                tmp_stack.append(child)
        parallelism_degree.append(parallelism_coef)
        stack = tmp_stack.copy()

    return parallelism_degree


def metric_vertex_count(wg: WorkGraph) -> float:
    return wg.vertex_count


def metric_average_work_per_activity(wg: WorkGraph) -> float:
    return sum(node.work_unit.volume for node in wg.nodes) / wg.vertex_count


def metric_max_children(wg: WorkGraph) -> float:
    return max((len(node.children) for node in wg.nodes if node.children))


def metric_average_resource_usage(wg: WorkGraph) -> float:
    return sum(sum((req.min_count + req.max_count) / 2 for req in node.work_unit.worker_reqs)
               for node in wg.nodes) / wg.vertex_count


def metric_max_parents(wg: WorkGraph) -> float:
    return max((len(node.parents) for node in wg.nodes if node.parents))


def encode_graph(wg: WorkGraph) -> list[float]:
    return [
        metric_vertex_count(wg),
        metric_average_work_per_activity(wg),
        metric_max_children(wg),
        metric_average_resource_usage(wg),
        metric_max_children(wg),
        *metric_graph_parallelism_degree(wg)
    ]
