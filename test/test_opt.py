import sys
import torch
import time
from torch.nn import LSTM

sys.path.insert(0, '../src/')

from subLSTM.nn import SubLSTM

n_layers = 1
state_size = 128
batch_size = 16
timesteps = 784
input_features = 1
trials = 1000
device = 'cpu'

X = torch.randn(timesteps, batch_size, input_features).to(device)
h = torch.randn(n_layers, batch_size, state_size).to(device)
C = torch.randn(n_layers, batch_size, state_size).to(device)

def run(model):
    forward = 0
    backward = 0
    for _ in range(trials):
        start = time.time()
        out, (new_h, new_C) = model(X, (h, C))

        forward += time.time() - start

        start = time.time()
        out.sum().backward()
        backward += time.time() - start

    return forward, backward

sublstm = SubLSTM(input_features, state_size).to(device=device)
lstm = LSTM(input_features, state_size).to(device=device)

sublstm_forward, sublstm_backward = run(sublstm)

print('subLSTM -> Forward: {:.3f} us | Backward {:.3f} us'.format(
    sublstm_forward * 1e6/1e5, sublstm_backward * 1e6/1e5))

lstm_forward, lstm_backward = run(lstm)

print('LSTM -> Forward: {:.3f} us | Backward {:.3f} us'.format(
    lstm_forward * 1e6/1e5, lstm_backward * 1e6/1e5))