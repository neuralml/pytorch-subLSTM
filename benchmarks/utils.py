import numpy as np
import torch
import torch.nn as nn
import time


def detach_hidden_state(hidden_state):
    """
    Use this method to detach the hidden state from the previous batch's history.
    This way we can carry hidden states values across training which improves
    convergence  while avoiding multiple initializations and autograd computations
    all the way back to the start of start of training.
    """

    if isinstance(hidden_state, torch.Tensor):
        return hidden_state.detach()
    elif isinstance(hidden_state, list):
        return [detach_hidden_state(h) for h in hidden_state]
    elif isinstance(hidden_state, tuple):
        return tuple(detach_hidden_state(h) for h in hidden_state)
    return None


def train(model, data_loader, criterion, optimizer, grad_clip,
        track_hidden, log_interval, device, verbose):
    """
    Train the model for one epoch over the whole dataset.
    """
    model.train(True)
    loss_trace, running_loss = [], 0.0
    n_batches = len(data_loader)

    # Keep track or the hidden state over the whole epoch. This allows faster training?
    hidden = None

    for i, data in enumerate(data_loader):
        # Load one batch into the device being used.
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Set all gradients to zero.
        optimizer.zero_grad()

        # If reusing hidden states, detach them from the computation graph 
        # of the previous batch. Using the previous value may speed up training 
        # but detaching is needed to avoid backprogating to the start of training.
        hidden = detach_hidden_state(hidden) if track_hidden else None

        # Forward and backward steps
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs, labels)

        loss.backward()

        # Clipping (helps with exploding gradients) and then gradient descent
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item()

        # Print the loss every log-interval mini-batches and save it to the trace
        if i % log_interval == log_interval - 1:
            if verbose:
                print('\t[batches %5d / %5d] loss: %.5f' %
                    (i + 1, n_batches, running_loss / log_interval))

            loss_trace.append(running_loss / log_interval)
            running_loss = 0.0

    return loss_trace


def test(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            total_loss += criterion(model(inputs)[0], labels)

    return total_loss / (i + 1)


def compute_accuracy(model, data_loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs = inputs.to(device)

            outputs = model(inputs)[0]
            predictions = torch.argmax(outputs, dim=1)

            total += predictions.size(0)
            correct += (labels == predictions).sum().item()

    return correct / total
