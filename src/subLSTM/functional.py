import torch
import torch.nn.functional as F


def sublstm(input, hidden, input_layer, recurrent_layer):
    h_tm1, c_tm1 = hidden
    proj_input = torch.sigmoid(input_layer(input) + recurrent_layer(h_tm1))

    in_gate, out_gate, z_t, f_gate = proj_input.chunk(4, 1)

    c_t = c_tm1 * f_gate + z_t - in_gate
    h_t = torch.sigmoid(c_t) - out_gate

    return h_t, c_t

def fsublstm(input, hidden, input_layer, recurrent_layer, f_gate=None):
    h_tm1, c_tm1 = hidden
    proj_input = torch.sigmoid(input_layer(input) + recurrent_layer(h_tm1))

    in_gate, out_gate, z_t = proj_input.chunk(3, 1)
    f_gate = f_gate.clamp(0, 1)

    c_t = c_tm1 * f_gate + z_t - in_gate
    h_t = torch.sigmoid(c_t) - out_gate

    return h_t, c_t
