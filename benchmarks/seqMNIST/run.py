# coding: utf-8

import sys
import os
import argparse
import time
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as trans
import torchvision.datasets as dataset

# To use wrapper.py and utils.py
sys.path.insert(0, '../')

from subLSTM.nn import SubLSTM
from wrappers import init_model
from utils import train, test, compute_accuracy

########################################################################################
# PARSE THE INPUT
########################################################################################

parser = argparse.ArgumentParser(description='PyTorch Sequential MNIST LSTM model test')

# Model parameters
parser.add_argument('--model', type=str, default='subLSTM',
    help='RNN model tu use. One of subLSTM|fix-subLSTM|LSTM|GRU')
parser.add_argument('--nlayers', type=int, default=1,
    help='number of layers')
parser.add_argument('--nhid', type=int, default=50,
    help='number of hidden units per layer')

# Data parameters
parser.add_argument('--data', type=str, default='MNIST',
    help='location of the data set')
parser.add_argument('--train-test-split', type=float, default=0.2,
    help='proportion of trainig data used for validation')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
    help='batch size')
parser.add_argument('--shuffle', action='store_true',
    help='shuffle the data at the start of each epoch.')
parser.add_argument('--input-size', type=int, default=1,
    help='the default dimensionality of each input timestep.'
    'defaults to 1, meaning instances are treated like one large 1D sequence')

# Training parameters
parser.add_argument('--epochs', type=int, default=40,
    help='max number of training epochs')
parser.add_argument('--optim', type=str, default='rmsprop',
    help='learning rule, supports adam|sparseadam|adamax|rmsprop|sgd|adagrad|adadelta')
parser.add_argument('--lr', type=float, default=1e-4,
    help='initial learning rate')
parser.add_argument('--l2-norm', type=float, default=0,
    help='weight of L2 norm')
parser.add_argument('--clip', type=float, default=1,
    help='gradient clipping')
parser.add_argument('--track-hidden', action='store_true',
    help='keep the hidden state values across a whole epoch of training')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
    help='report interval')

# Replicability and storage
parser.add_argument('--save', type=str,  default='results',
    help='path to save the final model')
parser.add_argument('--seed', type=int, default=18092,
    help='random seed')

# CUDA
parser.add_argument('--cuda', action='store_true',
    help='use CUDA')

# Print options
parser.add_argument('--verbose', action='store_true',
    help='print the progress of training to std output.')

args = parser.parse_args()

########################################################################################
# SETTING UP THE DIVICE AND SEED
########################################################################################

torch.manual_seed(args.seed)
if args.cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


########################################################################################
# LOAD DATA
########################################################################################

transform = trans.Compose([
    trans.ToTensor(),
    trans.Lambda(lambda x: x.view(-1, args.input_size))
])

# Load data
data_path, batch_size = args.data, args.batch_size

train_data = dataset.MNIST(
    root=data_path, train=True, transform=transform, download=True)

# Split train data into training and validation sets
N = len(train_data)
val_size = int(N * 0.2)
train_data, validation_data = torch.utils.data.random_split(
    train_data, [N - val_size, val_size])

train_data = DataLoader(train_data, batch_size=batch_size, shuffle=args.shuffle)
validation_data = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

test_data = dataset.MNIST(
    root=data_path, train=False, transform=transform, download=True)
test_data = DataLoader(test_data, batch_size=batch_size, shuffle=False)

########################################################################################
# CREATE THE MODEL
########################################################################################

input_size, hidden_size, n_classes = args.input_size, args.nhid, 10

model = init_model(
    model_type=args.model,
    n_layers=args.nlayers, hidden_size=args.nhid,
    input_size=input_size, output_size=10, class_task=True,
    dropout=0.0, device=device
)

########################################################################################
# SET UP OPTIMIZER & OBJECTIVE FUNCTION
########################################################################################

if args.optim == 'adam':
    optimizer = optim.Adam(model.parameters(),
        lr=args.lr, eps=1e-9, weight_decay=args.l2_norm, betas=[0.9, 0.98])
elif args.optim == 'sparseadam':
    optimizer = optim.SparseAdam(model.parameters(),
        lr=args.lr, eps=1e-9, weight_decay=args.l2_norm, betas=[0.9, 0.98])
elif args.optim == 'adamax':
    optimizer = optim.Adamax(model.parameters(),
        lr=args.lr, eps=1e-9, weight_decay=args.l2_norm, betas=[0.9, 0.98])
elif args.optim == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(),
        lr=args.lr, eps=1e-10, weight_decay=args.l2_norm, momentum=0.9)
elif args.optim == 'sgd':
    optimizer = optim.SGD(model.parameters(),
        lr=args.lr, weight_decay=args.l2_norm, momentum=0.9) # 0.01
elif args.optim == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(),
        lr=args.lr, weight_decay=args.l2_norm, lr_decay=0.9)
elif args.optim == 'adadelta':
    optimizer = optim.Adadelta(model.parameters(),
        lr=args.lr, weight_decay=args.l2_norm, rho=0.9)
else:
    raise ValueError(r'Optimizer {0} not recognized'.format(args.optim))

criterion = nn.CrossEntropyLoss()

########################################################################################
# TRAIN MODEL
########################################################################################

epochs, log_interval = args.epochs, args.log_interval
loss_trace, best_loss = [], np.inf
save_path = args.save + '/{0}_{1}_{2}'.format(args.model, args.nlayers, args.nhid)

if not os.path.exists(save_path):
    os.makedirs(save_path)

if args.verbose:
    print('Training {} model with parameters:'
            '\n\tnumber of layers: {}'
            '\n\thidden units: {}'
            '\n\tmax epochs: {}'
            '\n\tbatch size: {}'
            '\n\toptimizer: {}, lr={}, l2={}'.format(
                args.model, args.nlayers, hidden_size, epochs,
                batch_size, args.optim, args.lr, args.l2_norm
            ))

    if args.cuda and torch.cuda.is_available():
        print('\tusing CUDA')
    else:
        print('\tusing CPU')
        if torch.cuda.is_available():
            print('\tWARNING: CUDA device available but not being used. \
                run with --cuda option to enable it.\n\n')

try:
    for e in range(epochs):
        if args.verbose:
            print('training epoch {0}'.format(e + 1))

        start_time = time.time()

        # Train model for 1 epoch over whole dataset
        epoch_trace = train(
            model=model, data_loader=train_data,
            criterion=criterion, optimizer=optimizer, grad_clip=args.clip,
            log_interval=log_interval,
            device=device,
            track_hidden=args.track_hidden,
            verbose=args.verbose
        )

        print((time.time() - start_time) * 1e6/1e5)

        loss_trace.extend(epoch_trace)

        # Check validation loss
        val_loss = test(model, validation_data, criterion, device)

        if args.verbose:
            print('epoch {} finished \
                \n\ttotal time {} \
                \n\ttraining_loss = {:5.4f} \
                \n\tvalidation_loss = {:5.4f}'.format(
                    e + 1,
                    time.time() - start_time,
                    np.sum(epoch_trace) / len(epoch_trace),
                    val_loss))

        if val_loss < best_loss:
            with open(save_path + '/model.pt', 'wb') as f:
                torch.save({
                    'model_params':{
                        'model_type': args.model,
                        'hidden_size': hidden_size,
                        'n_layers': args.nlayers,
                        'input_size': input_size,
                        'output_size': 10,
                        'class_task': True
                    },
                    'epoch': e + 1,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'loss': val_loss
                }, f)
            best_loss = val_loss

except KeyboardInterrupt:
    if args.verbose:
        print('Keyboard interruption. Terminating training.')

# Save the trace of the loss during training
with open(save_path + '/trace.csv', 'w') as f:
    wr = csv.writer(f, quoting=csv.QUOTE_ALL)
    wr.writerow(loss_trace)

########################################################################################
# VALIDATE
########################################################################################

with open(save_path + '/model.pt', 'rb') as f:
    model.load_state_dict(torch.load(f)['model_state'])

test_loss = test(model, test_data, criterion, device)
accuracy = compute_accuracy(model, test_data, device)
if args.verbose:
    print('Training ended:\n\ttest loss {:5.4f}\n\taccuracy {:3.2%}'.format(
        test_loss, accuracy))
