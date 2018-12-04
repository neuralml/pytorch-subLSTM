# coding: utf-8

import sys
import os
import argparse
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as trans
import torchvision.datasets as dataset

# To use wrapper.py and utils.py
sys.path.insert(0, '../')

from subLSTM.nn import SubLSTM
from wrappers import init_model
from utils import set_activation_recorder, test, compute_accuracy

########################################################################################
# PARSE INPUT
########################################################################################

parser = argparse.ArgumentParser(
    description="Recording activation of LSTM variants on sequential MNIST test set")
# Model
parser.add_argument('--model', type=str, help='path to saved model')

# DATA
parser.add_argument('--data', type=str, default='MNIST', help='path to MNIST dataset')

# ANALYSIS
parser.add_argument('--record-act', action='store_true',
    help='record the activations of units and gates')
parser.add_argument('--save', type=str, default='',
    help='path folder used to save recorded data')

# CUDA
parser.add_argument('--cuda', action='store_true', help='use CUDA')

# Print options
parser.add_argument('--verbose', action='store_true',
    help='print the progress of training to std output.')

args = parser.parse_args()

########################################################################################
# SETTING UP THE DIVICE AND SEED
########################################################################################

if args.cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

########################################################################################
# LOAD MODEL
########################################################################################

args.model = 'results/bc4trained/subLSTM_1_50/model.pt'

with open(args.model, mode='rb') as f:
    saved_model = torch.load(f)

model = init_model(device=device, **saved_model['model_params'])
model.load_state_dict(saved_model['model_state'])

print(saved_model['model_params'])

########################################################################################
# LOAD DATA
########################################################################################

input_size = saved_model['model_params']['input_size']

transform = trans.Compose([
    trans.ToTensor(),
    trans.Lambda(lambda x: x.view(-1, input_size))
])

# Load data
test_data = dataset.MNIST(
    root=args.data, train=False, transform=transform, download=True)
test_data = DataLoader(test_data, batch_size=20, shuffle=False)

criterion = nn.CrossEntropyLoss()

########################################################################################
# TEST AND SAVE
########################################################################################

args.record_act = True

if args.record_act:
    hook, recordings = set_activation_recorder(model)

test_loss = test(model, test_data, criterion, device)
accuracy = compute_accuracy(model, test_data, device)
if args.verbose:
    print('Training ended:\n\ttest loss {:5.4f}\n\taccuracy {:3.2%}'.format(
        test_loss, accuracy))

if args.record_act and args.save:
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    with open(args.save + 'recordings.csv', mode='w') as f:
        wr = csv.DictWriter(f, fieldnames=recordings.keys(), quoting=csv.QUOTE_ALL)
        wr.writeheader()
        wr.writerow(recordings)
