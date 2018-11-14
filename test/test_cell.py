import sys
import torch

sys.path.insert(0, '.')

from src.subLSTM import SubLSTMCell


in_size, hidden_size, batch_size = 30, 100, 2
input = torch.randn(batch_size, in_size, requires_grad=False)

cell = SubLSTMCell(in_size, hidden_size)

hx, cx = cell(input)

hx.sum().backward()

assert hx.size() == (batch_size, hidden_size)
assert cx.size() == (batch_size, hidden_size)

fix_cell = SubLSTMCell(in_size, hidden_size, fix_subLSTM=True)

hx, cx = fix_cell(input)

hx.sum().backward()

assert hx.size() == (batch_size, hidden_size)
assert cx.size() == (batch_size, hidden_size)

