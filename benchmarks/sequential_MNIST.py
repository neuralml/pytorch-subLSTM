import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as trans
import torchvision.datasets as datasets
import pandas as pd

sys.path.insert(0, '.')

from src.subLSTM import SubLSTM
from src.wrappers import RNNClassifier


# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Using the {0}'.format(device))

# Set dataset path
data_path = 'benchmarks/data/MNIST'

# Transform data to the sequential MNIST task
transform = trans.Compose([
    trans.ToTensor(),
    trans.Lambda(lambda x: x.view(-1, 1))
])

# Load data
train = datasets.MNIST(root=data_path, train=True, transform=transform, download=True)
test = datasets.MNIST(root=data_path, train=False, transform=transform, download=True)

# Set up the models: LSTM, GRu and SubLSTM (w & w/o fixed forget gates)
input_size, hidden_size, n_classes = 1, 50, 10

models = [
    ('fix-subLSTM', RNNClassifier(
        SubLSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True),
        rnn_output_size=hidden_size, n_classes=n_classes
    )),
    ('subLSTM', RNNClassifier(
        SubLSTM(input_size=input_size, hidden_size=hidden_size, fixed_forget=False, batch_first=True),
        rnn_output_size=hidden_size, n_classes=n_classes
    )),
    ('LSTM', RNNClassifier(
        nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True),
        rnn_output_size=hidden_size, n_classes=n_classes
    )),
    ('GRU', RNNClassifier(
        nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True),
        rnn_output_size=hidden_size, n_classes=n_classes
    ))
]

# Training Parameters
epochs, batch_size, learning_rate, momentum = 100, 200, 1e-4, 0.9

# Create a folder to store the results
results_path = os.getcwd() + '/benchmarks/results/all_models-E=10-BS=200-lr=1e-4'
if not os.path.exists(results_path):
    os.makedirs(results_path)

results = {}
save_period = 200  # save every 200 mini-batches

train_loader = DataLoader(train, batch_size=batch_size)
criterion = torch.nn.NLLLoss()

for name, rnn in models:

    print('Training model: {}'.format(name))

    optimizer = optim.RMSprop(rnn.parameters(), lr=learning_rate, momentum=momentum)
    epoch_losses = []

    for e in range(epochs):
        running_loss = 0.0
        epoch_losses.append(0.0)

        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()

            # Forward, backwards and optimization step
            loss = criterion(rnn(inputs), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % save_period == save_period - 1:  # print every 200 mini-batches
                print('\t[%d, %5d] loss: %.3f' %
                      (e + 1, i + 1, running_loss / save_period))
                epoch_losses.append(running_loss / save_period)
                running_loss = 0.0

        # record the losses for each model
        epoch_losses[-1] /= (i + 1)

    results[name] = np.asarray(epoch_losses)
    print('Finished training model: {0}\n\t Final loss {1}'.format(name, results[name][-1]))

results = pd.DataFrame.from_dict(results)
results.to_csv(path_or_buf=results_path + '/results.csv')
