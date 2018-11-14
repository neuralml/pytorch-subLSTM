import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as trans
import torchvision.datasets as datasets

sys.path.insert(0, '.')

from src.subLSTM import SubLSTM
from src.wrappers import RNNClassifier


# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Set up the models: LSTM, GRu and SubLSTM
input_size, hidden_size, n_classes = 1, 100, 10

models = [
    ('LSTM', RNNClassifier(
        nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True),
        rnn_output_size=hidden_size, n_classes=n_classes
    )),
    ('GRU', RNNClassifier(
        nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True),
        rnn_output_size=hidden_size, n_classes=n_classes
    )),
    ('fix-subLSTM', RNNClassifier(
        SubLSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True),
        rnn_output_size=hidden_size, n_classes=n_classes
    )),
    ('subLSTM', RNNClassifier(
        SubLSTM(input_size=input_size, hidden_size=hidden_size, fixed_forget=False, batch_first=True),
        rnn_output_size=hidden_size, n_classes=n_classes
    )),
]


# Training Parameters
epochs, batch_size, learning_rate, momentum = 10, 200, 1e-4, 0.9

train_loader = DataLoader(train, batch_size=batch_size)
criterion = torch.nn.NLLLoss()

for name, rnn in models:

    print('Training model: {}'.format(name))

    optimizer = optim.RMSprop(rnn.parameters(), lr=learning_rate, momentum=momentum)

    for e in range(epochs):
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()

            loss = criterion(rnn(inputs), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 200 == 199:  # print every 200 mini-batches
                print('\t[%d, %5d] loss: %.3f' %
                      (e + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    print('Finished training model: {}'.format(name))
