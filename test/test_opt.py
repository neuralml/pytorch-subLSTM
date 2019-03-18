import torch
import time
from subLSTM.nn import SubLSTM
from torch.nn import LSTM

batch_size = 16
input_features = 1
state_size = 128
timesteps = 784
n_layers = 1
trials = 100

X = torch.randn(timesteps, batch_size, input_features)
h = torch.randn(n_layers, batch_size, state_size)
C = torch.randn(n_layers, batch_size, state_size)

sublstm = SubLSTM(input_features, state_size, fixed_forget=False)

forward = 0
backward = 0
for _ in range(trials):
    start = time.time()
    out, (new_h, new_C) = sublstm(X, (h, C))
    forward += time.time() - start

    start = time.time()
    out.sum().backward()
    backward += time.time() - start

print('subLSTM -> Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e5, backward * 1e6/1e5))

lstm = LSTM(input_features, state_size)

forward = 0
backward = 0
for _ in range(trials):
    start = time.time()
    out, (new_h, new_C) = lstm(X, (h, C))
    forward += time.time() - start

    start = time.time()
    out.sum().backward()
    backward += time.time() - start

print('LSTM -> Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e5, backward * 1e6/1e5))