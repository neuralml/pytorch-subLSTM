import numpy as np
import torch.nn as nn

from src.subLSTM import SubLSTM
from src.wrappers import RNNClassifier

def init_model(model_type, input_size, n_layers, hidden_size, output_size,
                device, class_task=True):
    if model_type == 'subLSTM':
        rnn = SubLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            fixed_forget=False, 
            batch_first=True
        )

    elif model_type == 'fix-subLSTM':
        rnn = SubLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            fixed_forget=True, 
            batch_first=True
        )
    
    elif model_type == 'LSTM':
        rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
        )

    elif model_type == 'GRU':
        rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
        )
        
    else:
        raise ValueError('Unrecognized RNN type')

    if class_task:
        model = RNNClassifier(
            rnn=rnn, rnn_output_size=hidden_size, n_classes=output_size
        ).to(device=device)
    else:
        raise NotImplementedError()

    return model


def train(model, data_loader, optimizer, criterion, log_interval, device):
    model.set_train()
    loss_trace, running_loss = [], 0.0
    dataset_size, batch_size = len(data_loader.dataset), data_loader.batch_size, 

    for i, data in enumerate(data_loader):
        # Load one batch into the device being used.
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Set all gradients to zero
        optimizer.zero_grad()

        # Forward, backward and optimization step.
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print the loss every log-interval mini-batches and save it to the trace
        if i % log_interval == log_interval - 1:
            print('\t[batches %5d / %5d] loss: %.3f' %
                ((i + 1) * batch_size, dataset_size, running_loss / log_interval))
            loss_trace.append(running_loss / log_interval)
            running_loss = 0.0

    return loss_trace

def test(model, data_loader, criterion, device):
    total_loss = 0.0
    model.set_eval()

    for i, data in enumerate(data_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        total_loss += criterion(model(inputs), labels)

    return total_loss / (i + 1)
