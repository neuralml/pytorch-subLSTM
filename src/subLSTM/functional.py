import torch
import torch.nn.functional as F


def SubLSTMCellF(input, hidden, W_i, W_r, b_i, b_h):
    h_tm1, c_tm1 = hidden

    proj_input = torch.sigmoid(F.linear(input, W_i, b_i) + F.linear(h_tm1, W_r, b_h))
    in_gate, out_gate, z_t, f_gate = proj_input.chunk(4, 1)

    c_t = c_tm1 * f_gate + z_t - in_gate
    h_t = torch.sigmoid(c_t) - out_gate

    return h_t, c_t


def fixSubLSTMCellF(input, hidden, W_i, W_r, f_gate, b_i, b_h):
    h_tm1, c_tm1 = hidden

    proj_input = torch.simoid(F.linear(input, W_i, b_i) + F.linear(h_tm1, W_r, b_h))
    in_gate, out_gate, z_t = proj_input.chunk(3, 1)

    f_gate = f_gate.clamp(0, 1)

    c_t = c_tm1 * f_gate + z_t - in_gate
    h_t = torch.sigmoid(c_t) - out_gate

    return h_t, c_t
