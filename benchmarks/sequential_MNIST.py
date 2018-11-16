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
from benchmarks.training import train

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
train_data = datasets.MNIST(root=data_path, train=True, transform=transform, download=True)
test_data = datasets.MNIST(root=data_path, train=False, transform=transform, download=True)

# Set up the models: LSTM, GRu and SubLSTM (w & w/o fixed forget gates)
input_size, hidden_size, n_classes = 1, 50, 10

models = [
    ('subLSTM', RNNClassifier(
        SubLSTM(input_size=input_size, hidden_size=hidden_size, fixed_forget=False, batch_first=True),
        rnn_output_size=hidden_size, n_classes=n_classes
    )),
    ('fix-subLSTM', RNNClassifier(
        SubLSTM(input_size=input_size, hidden_size=hidden_size, fixed_forget=True, batch_first=True),
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
log_interval = 200  # save every 200 mini-batches

# Set up training
train_data_loader = DataLoader(train_data, batch_size=batch_size)
criterion = torch.nn.CrossEntropyLoss()

for name, rnn in models:
    print('Training model: {}'.format(name))
    # Train the model using the specified parameters
    optimizer = optim.RMSprop(rnn.parameters(), lr=learning_rate, momentum=momentum)
    results[name] = train(rnn, train_data_loader, optimizer, criterion, epochs, log_interval)

    print('Finished training model: {0}\n\t Final loss {1}'.format(name, results[name][-1]))

results = pd.DataFrame.from_dict(results)
results.to_csv(path_or_buf=results_path + '/training_results.csv')
