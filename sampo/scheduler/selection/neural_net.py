import os

import numpy as np
import torch
import torch.nn.functional as F
from ray.air import session
from torch import nn

from sampo.scheduler.selection.metrics import one_hot_encode


class NeuralNet(nn.Module):
    def __init__(self, input_size: int = 13, layer_size: int = 15, layer_count: int = 6, out_size: int = 2):
        super().__init__()

        self._layers_count = layer_count
        self._linear0 = torch.nn.Linear(input_size, layer_size)
        self.model = nn.Sequential(self._linear0)
        # self.model.add_module()
        for i in range(layer_count):
            self.__dict__[f'_linear{i + 1}'] = torch.nn.Linear(layer_size, layer_size)
            self.model.add_module(name=f'_linear{i + 1}',
                                  module=self.__dict__[f'_linear{i + 1}'])
        self.__dict__[f'_linear{layer_count + 1}'] = torch.nn.Linear(layer_size, out_size)
        self.model.add_module(name=f'_linear{layer_count + 1}',
                              module=self.__dict__[f'_linear{layer_count + 1}'])

    def forward(self, X):
        X = self._linear0(X)
        for i in range(1, self._layers_count + 2):
            linear = self.__dict__[f'_linear{i}']
            X = F.relu(X)
            X = linear(X)
        X = F.softmax(X, dim=0)
        return X


class NeuralNetTrainer:
    def __init__(self, model: NeuralNet, criterion, optimizer: torch.optim):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, x, y, epochs=10):
        # criterion = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)
        total_loss = 0.0

        # Train the model
        total_step = len(x)
        for epoch in range(epochs):
            for i, (image, label) in enumerate(zip(x, y)):
                # Move tensors to the configured device
                image = image.to(self._device)
                label = label.to(self._device)

                # Forward pass
                outputs = self.model(image)
                loss = self.criterion(outputs, label.float())

                # Backpropagation and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, epochs, i + 1, total_step, loss.item()))
        return total_loss / len(y)

    def validate(self, x, y):
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, (image, label) in enumerate(zip(x, y)):
            with torch.no_grad():
                image = image.to(self._device)
                label = label.to(self._device)

                outputs = self.model(image)
                loss = self.criterion(outputs, label.float())

                _, predicted = torch.max(outputs.data, 0)
                _, label = torch.max(label.data, 0)
                total += 1

                # accuracy = self.scorer(y, predicted)
                correct += int(int(predicted) == int(label))

                val_loss += loss.item()
                val_steps += 1

        session.report(
            {'loss': val_loss / val_steps, 'accuracy': correct / total}
        )
        return correct / total, val_loss / val_steps

    def predict(self, x: list) -> torch.tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        result = []
        with torch.no_grad():
            for image in x:
                image = image.to(self._device)
                outputs = self.model(image)
                _, predicted = torch.max(outputs.data, 0)
                result.append(torch.tensor(one_hot_encode(predicted, 2)))
        return np.array([torch.max(v) for v in result])
        # return result

    def predict_proba(self, x: list) -> torch.tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        result = []
        with torch.no_grad():
            for image in x:
                # image = image.to(self._device)
                outputs = self.model(image)
                result.append(outputs)
        return np.array([v.numpy() for v in result])

    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "best_model_2.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint = torch.load(tmp_checkpoint_dir)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

