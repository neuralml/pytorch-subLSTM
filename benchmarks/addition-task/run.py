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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, '../')

from subLSTM.nn import SubLSTM
from wrappers import init_model
from utils import train, test


########################################################################################
# PARSE THE INPUT
########################################################################################

parser = argparse.ArgumentParser(description='Addition task')

# Model parameters
parser.add_argument('--model', type=str, default='subLSTM', 
    help='RNN model tu use. One of subLSTM|fix-subLSTM|LSTM|GRU')
parser.add_argument('--nlayers', type=int, default=1,
    help='number of layers')
parser.add_argument('--nhid', type=int, default=50,
    help='number of hidden units per layer')
parser.add_argument('--gact', type=str, default='relu',
    help='gate activation function relu|sig')
parser.add_argument('--gbias', type=float, default=0,
    help='gating bias')

# Data parameters
parser.add_argument('--seq-length', type=int, default=20,
    help='sequence length')
parser.add_argument('--num-addends', type=int, default=2,
    help='the number of addends to be unmasked in each sequence'
    'must be less than the sequence length')
parser.add_argument('--min-arg', type=float, default=-10,
    help='minimum value of the addends')
parser.add_argument('--max-arg', type=float, default=10,
    help='maximum value of the addends')
parser.add_argument('--training-size', type=int, default=1000,
    help='size of the randomly created training set')
parser.add_argument('--testing-size', type=int, default=200,
    help='size of the randomly created test set')
parser.add_argument('--train-val-split', type=float, default=0.2,
    help='proportion of trainig data used for validation')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
    help='batch size')

# Training parameters
parser.add_argument('--epochs', type=int, default=40,
    help='max number of training epochs')
parser.add_argument('--optim', type=str, default='rmsprop',
    help='gradient descent method,'
    'supports adam|sparseadam|adamax|rmsprop|sgd|adagrad|adadelta')
parser.add_argument('--lr', type=float, default=1e-4,
    help='initial learning rate')
parser.add_argument('--l2-norm', type=float, default=0,
    help='weight of L2 norm')
parser.add_argument('--clip', type=float, default=1,
    help='gradient clipping')
parser.add_argument('--track-hidden', action='store_true',
    help='keep the hidden state values across a whole epoch of training')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
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

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

########################################################################################
# CREATE THE DATA GENERATOR
########################################################################################

seq_len, num_addends = args.seq_length, args.num_addends
min_arg, max_arg = args.min_arg, args.max_arg
N, batch_size, test_size = args.training_size, args.batch_size, args.testing_size

train_size = int(N * (1 - args.train_val_split))
val_size = N - train_size

class BatchGenerator:
    def __init__(self,  training_size, batch_size, min_arg, max_arg):
        self.min_arg = min_arg
        self.max_arg = max_arg
        self.batch_size = batch_size
        self.training_size = training_size

    def __iter__(self):
        for _ in range(len(self)):
            yield self.next_batch()

    def __len__(self):
        return self.training_size // self.batch_size

    def next_batch(self):
        batch_size, min_arg, max_arg = self.batch_size, self.min_arg, self.max_arg
        inputs = np.random.uniform(
            low=min_arg, high=max_arg, size=(batch_size, seq_len, 2))
        inputs[:, :, 1] = -1

        # Neat trick to sample the positions to unmask
        mask = np.random.rand(batch_size, seq_len).argsort(axis=1)[:,:num_addends]
        mask.sort(axis=1)

        # Mask is in the wrong shape (batch_size, num_addends) for slicing
        inputs[range(batch_size), mask.T, 1] = 1
        targets = np.sum(inputs[:, :, 0] * (inputs[:, :, 1] == 1), axis=1).reshape(-1, 1)

        inputs = torch.as_tensor(inputs, dtype=torch.float)
        targets = torch.as_tensor(targets, dtype=torch.float)

        return inputs, targets

training_data = BatchGenerator(train_size, batch_size, min_arg, max_arg)
validation_data = BatchGenerator(val_size, val_size, min_arg, max_arg)
test_data = BatchGenerator(test_size, test_size, min_arg, max_arg)

########################################################################################
# CREATE THE MODEL
########################################################################################

input_size, hidden_size, responses = 2, args.nhid, 1

model = init_model(
    model_type=args.model,
    n_layers=args.nlayers, hidden_size=args.nhid,
    input_size=input_size, output_size=responses, class_task=False,
    device=device
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

criterion = nn.MSELoss()

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
            model=model, data_loader=training_data, 
            criterion=criterion, optimizer=optimizer, grad_clip=args.clip,
            log_interval=log_interval,
            device=device,
            track_hidden=args.track_hidden,
            verbose=args.verbose
        )

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

if args.verbose:
    print('Training ended:\n\ttest loss {:5.4f}'.format(
        test_loss))
