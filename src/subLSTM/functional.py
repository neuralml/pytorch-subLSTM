import torch
import torch.nn.functional as F


# noinspection PyPep8Naming
def SubLSTMCellF(input, hidden, W_i, W_r, b_i=None, b_h=None):
    h_tm1, c_tm1 = hidden

    if b_i is None:
        b_i = torch.zeros_like(input)
    if b_h is None:
        b_h = torch.zeros_like(h_tm1)

    proj_input = torch.sigmoid(F.linear(input, W_i, b_i) + F.linear(h_tm1, W_r, b_h))

    in_gate, out_gate, z_t, f_gate = proj_input.chunk(4, 1)

    c_t = c_tm1 * f_gate + z_t - in_gate
    h_t = torch.sigmoid(c_t) - out_gate

    return h_t, c_t


# noinspection PyPep8Naming
def fixSubLSTMCellF(input, hidden, W_i, W_r, f_gate, b_i=None, b_h=None):
    h_tm1, c_tm1 = hidden

    if b_i is None:
        b_i = torch.zeros_like(input)
    if b_h is None:
        b_h = torch.zeros_like(h_tm1)

    proj_input = torch.sigmoid(F.linear(input, W_i, b_i) + F.linear(h_tm1, W_r, b_h))

    in_gate, out_gate, z_t = proj_input.chunk(3, 1)
    f_gate = f_gate.clamp(0, 1)

    c_t = c_tm1 * f_gate + z_t - in_gate
    h_t = torch.sigmoid(c_t) - out_gate

    return h_t, c_t
