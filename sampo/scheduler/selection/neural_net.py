import os

import numpy as np
import torch
import torch.nn.functional as F
from ray.air import session
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sampo.scheduler.selection.metrics import one_hot_encode


class NeuralNetType:
    REGRESSION = 'REGRESSION'
    CLASSIFICATION = 'CLASSIFICATION'


class NeuralNet(nn.Module):
    def __init__(self, input_size: int = 13, layer_size: int = 15, layer_count: int = 6, out_size: int = 2,
                 task_type: NeuralNetType = NeuralNetType.REGRESSION):
        super().__init__()

        self.task_type = task_type
        self._layers_count = layer_count
        self._linear0 = torch.nn.Linear(input_size, layer_size)
        self.model = nn.Sequential(self._linear0)
        for i in range(layer_count):
            self.model.add_module(name=f'_relu',
                                  module=torch.nn.ReLU())
            self.model.add_module(name=f'_linear{i + 1}',
                                  module=torch.nn.Linear(layer_size, layer_size))
        self.model.add_module(name=f'_linear{layer_count + 1}',
                              module=torch.nn.Linear(layer_size, out_size))

    def forward(self, X):
        X = self.model(X)
        if self.task_type == NeuralNetType.CLASSIFICATION:
            X = F.softmax(X, dim=0)
        else:
            X = X
        return X


class NeuralNetTrainer:
    def __init__(self, model: NeuralNet, criterion, optimizer, scorer, batch_size):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scorer = scorer
        self.batch_size = batch_size

    def fit(self, x, y, epochs=10):
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Train the model
        for epoch in range(epochs):

            total_loss = 0.0
            total_step = len(loader)

            for i, (image, label) in enumerate(loader):
                image = image.to(self._device)
                label = label.to(self._device)

                outputs = self.model(image)
                loss = self.criterion(outputs, label.float())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, epochs, i + 1, total_step, loss.item()))

            print(f'Epoch [{epoch + 1}/{epochs}], Average Loss: {total_loss / total_step:.4f}')

    def validate(self, x, y):
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=self.batch_size)

        val_loss = 0.0
        val_score = 0.0
        total = len(loader)

        for i, (image, label) in enumerate(loader):
            with torch.no_grad():
                image = image.to(self._device)
                label = label.to(self._device)

                outputs = self.model(image)
                loss = self.criterion(outputs, label.float())

                val_score += self.scorer(outputs, label.float()).item()

                val_loss += loss.item()

        session.report(
            {'loss': val_loss / total, 'score': val_score / total}
        )
        return val_score / total, val_loss / total

    def predict(self, x: list):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        result = []

        with torch.no_grad():
            for image in x:
                image = image.to(self._device)
                outputs = self.model(image)

                if self.model.task_type is NeuralNetType.CLASSIFICATION:
                    _, predicted = torch.max(outputs.data, 0)
                    result.append(torch.tensor(one_hot_encode(predicted, 2)))
                elif self.model.task_type is NeuralNetType.REGRESSION:
                    result.append(outputs)

        if self.model.task_type is NeuralNetType.CLASSIFICATION:
            return np.asarray([torch.max(v) for v in result])
        elif self.model.task_type is NeuralNetType.REGRESSION:
            return np.asarray([v.numpy() for v in result])

    def predict_proba(self, x: list) -> np.array:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        result = []
        with torch.no_grad():
            for image in x:
                # image = image.to(self._device)
                outputs = self.model(image)
                result.append(outputs)
        return np.array([v.numpy() for v in result])

    def save_checkpoint(self, tmp_checkpoint_dir, file_name):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, file_name)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint = torch.load(tmp_checkpoint_dir)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
