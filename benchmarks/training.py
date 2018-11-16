import numpy as np

def train(model, data_loader, optimizer, criterion, epochs, log_interval):
    model.set_train()
    loss_trace = []

    for e in range(epochs):
        running_loss = 0.0
        loss_trace.append(0.0)

        old_params = [np.copy(param.data) for param in model.parameters()]

        for i, data in enumerate(data_loader):
            inputs, labels = data
            optimizer.zero_grad()

            # Forward, backwards and optimization step
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % log_interval == log_interval - 1:  # print every 200 mini-batches
                print('\t[%d, %5d] loss: %.3f' %
                    (e + 1, i + 1, running_loss / log_interval))
                loss_trace.append(running_loss / log_interval)
                running_loss = 0.0

    return loss_trace

def test(model, data_loader, criterion):
    total_loss = 0.0
    model.set_eval()

    for i, data in enumerate(data_loader):
        inputs, labels = data
        total_loss += criterion(model(inputs), labels)

    return total_loss / (i + 1)
