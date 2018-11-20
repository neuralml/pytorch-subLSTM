# coding: utf-8

import argparse
import time
import math

import sys
import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
import torchvision.transforms as trans
import torchvision.datasets as dataset

sys.path.insert(0, '.')

from src.subLSTM import SubLSTM
from src.wrappers import RNNClassifier
from benchmarks.utils import train, test, init_model

parser = argparse.ArgumentParser(description='PyTorch Sequential MNIST LSTM model test')

parser.add_argument(
    '--data', type=str, default='./benchmarks/seqMNIST/MNIST', help='location of the data set')
parser.add_argument(
    '--model', type=str, default='subLSTM', 
    help='RNN model tu use. One of subLSTM|fix-subLSTM|LSTM|GRU')
parser.add_argument('--nlayers', type=int, default=1,
    help='number of layers')
parser.add_argument('--nhid', type=int, default=50,
    help='number of hidden units per layer')
parser.add_argument('--lr', type=float, default=1e-4,
    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
    help='gradient clipping')
parser.add_argument('--optim', type=str, default='rmsprop',
    help='learning rule, supports adam|sparseadam|adamax|rmsprop|sgd|adagrad|adadelta')
parser.add_argument('--epochs', type=int, default=40,
    help='max number of training epochs')
parser.add_argument('--batch-size', type=int, default=200, metavar='N',
    help='batch size')
parser.parse_args('--train-test-split', type=float, default=0.2,
    help='proportion of trainig data used for validation')
parser.add_argument('--seed', type=int, default=1111,
    help='random seed')
parser.add_argument('--cuda', action='store_true',
    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
    help='report interval')
parser.add_argument('--save', type=str,  default='./benchmarks/seqMNIST/results',
    help='path to save the final model')

args = parser.parse_args()

print('Training {} model with parameters:' \
    '\n\tnumber of layers: {}'\
    '\n\thidden units: {}'\
    '\n\tmax epochs: {}'\
    '\n\tbatch size: {}' \
    '\n\toptimizer: {}, lr={}'.format(
        args.model, args.nlayers, args.nhid, args.epochs,
        args.batch_size, args.optim, args.lr
))

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        device = torch.device('cpu')
        print('\tusing CPU\n\tWARNING: CUDA device available but not being used. \
            run with --cuda option to enable it.')
    else:
        torch.cuda.manual_seed(args.seed)
        device = torch.device('cuda')
        print('\tusing CUDA device')
else:
    print('\tusing CPU')
    device = torch.device('cpu')

print()

###################################################################################################
# LOAD DATA
###################################################################################################

transform = trans.Compose([
    trans.ToTensor(),
    trans.Lambda(lambda x: x.view(-1, 1))
])

# Load data
data_path, batch_size = args.data, args.batch_size

train_data = dataset.MNIST(root=data_path, train=True, transform=transform, download=True)

# Split train data into training and validation sets
N = len(train_data)
val_size = int(N * 0.2)
train_data, validation_data = torch.utils.data.random_split(train_data, [N - val_size, val_size])

train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
validation_data = DataLoader(validation_data, batch_size=batch_size, shuffle=True)

test_data = dataset.MNIST(root=data_path, train=False, transform=transform, download=True)
test_data = DataLoader(test_data, batch_size=batch_size, shuffle=True)

###################################################################################################
# CREATE THE MODEL
###################################################################################################

input_size, hidden_size, n_classes = 1, args.nhid, 10

model = init_model(
    model_type=args.model,
    n_layers=args.nlayers, hidden_size=args.nhid,
    input_size=1, output_size=10, class_task=True,
    device=device,
)

###################################################################################################
# SET UP TRAINING PARAMETERS
###################################################################################################

if args.optim == 'adam':
  optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-9, betas=[0.9, 0.98]) # 0.0001
if args.optim == 'sparseadam':
  optimizer = optim.SparseAdam(model.parameters(), lr=args.lr, eps=1e-9, betas=[0.9, 0.98]) # 0.0001
if args.optim == 'adamax':
  optimizer = optim.Adamax(model.parameters(), lr=args.lr, eps=1e-9, betas=[0.9, 0.98]) # 0.0001
elif args.optim == 'rmsprop':
  optimizer = optim.RMSprop(model.parameters(), lr=args.lr, eps=1e-10, momentum=0.9) # 0.0001
elif args.optim == 'sgd':
  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9) # 0.01
elif args.optim == 'adagrad':
  optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
elif args.optim == 'adadelta':
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
else:
    raise ValueError(r'Optimizer {0} not recognized'.format(args.optim))

criterion = nn.CrossEntropyLoss()

###################################################################################################
# TRAIN MODEL
###################################################################################################

epochs, log_interval = args.epochs, args.log_interval
loss_trace, best_loss = [], np.inf
save_path = args.save + '/{0}_{1}_{2}'.format(args.model, args.nlayers, args.nhid)

if not os.path.exists(save_path):
    os.makedirs(save_path)

try:
    for e in range(epochs):
        print('training epoch {0}'.format(e + 1))
        start_time = time.time()

        # Train model for 1 epoch over whole dataset
        epoch_trace = train(
            model=model, data_loader=train_data, 
            criterion=criterion, optimizer=optimizer, grad_clip=args.clip,
            log_interval=log_interval,
            device=device
        )

        loss_trace.extend(epoch_trace)

        # Check validation loss
        val_loss = test(model, validation_data, criterion, device)

        print('epoch {} finished \
            \n\ttotal time {} \
            \n\ttraining_loss = {:5.4f} \
            \n\tvalidation_loss = {:5.4f}'.format(
                e + 1, time.time() - start_time, np.sum(epoch_trace) / len(epoch_trace), val_loss))

        if val_loss < best_loss:
            with open(save_path + '/model.txt', 'wb') as f:
                torch.save(model, f)
            best_loss = val_loss
    
except KeyboardInterrupt:
    print('Keyboard interruption. Exiting training.')

# Save the trace of the loss during training
pd.DataFrame.from_dict({args.model: loss_trace}).to_csv(path_or_buf=save_path + '/trace.csv')

###################################################################################################
# VALIDATE
###################################################################################################

with open(save_path + '/model.txt', 'rb') as f:
    model = torch.load(f)

test_loss = test(model, test_data, criterion, device)
print('Training ended:\n\t test loss {:5.2f}'.format(test_loss))
