import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
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
                loss = criterion(outputs, label)

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, epochs, i + 1, total_step, loss.item()))

    def predict(self, x: list) -> list:
        result = []
        with torch.no_grad():
            for image in x:
                image = image.to(self._device)
                outputs = self(image)
                _, predicted = torch.max(outputs.data, 0)
                result.append(one_hot_encode(predicted, 3))
        return result

    def test_accuracy(self, x, y):
        # Test the model
        # In the test phase, don't need to compute gradients (for memory efficiency)
        correct = 0
        total = 0
        predicted = self.predict(x)

        for image, label, predicted_label in zip(x, y, predicted):
            total += 1
            correct += predicted_label == label.tolist()

        print(f'Accuracy of the network on the {len(x)} test images: {100 * correct / total} %')


def one_hot_encode(v, max_v):
    res = [0 for _ in range(max_v)]
    res[v] = 1
    return res


def load_dataset(filename: str) -> tuple[list, list, list, list]:
    df = pd.read_csv(filename)
    df.reset_index()
    x_train, x_test, y_train, y_test = train_test_split(df.drop('label', axis=1).to_numpy(), df['label'].to_numpy(),
                                                        stratify=df['label'].to_numpy())
    return [torch.Tensor(v) for v in x_train[:, 1:]], \
           [torch.Tensor(v) for v in x_test[:, 1:]], \
           [torch.Tensor(one_hot_encode(v, 3)) for v in y_train], \
           [torch.Tensor(one_hot_encode(v, 3)) for v in y_test]


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
        metric_max_children(wg)
    ]
