import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.jit.script
def sublstm(input, hidden, W, R, bi, bh):
    # type: (Tensor, Tuple[Tensor, Tensor], Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    h_tm1, c_tm1 = hidden

    proj_input = torch.sigmoid(F.linear(input, W , bi) + F.linear(h_tm1, R, bh))
    in_gate, out_gate, z_t, f_gate = proj_input.chunk(4, 1)

    c_t = c_tm1 * f_gate + z_t - in_gate
    h_t = torch.sigmoid(c_t) - out_gate

    return h_t, c_t

@torch.jit.script
def fsublstm(input, hidden, W, R, bi, bh, f_gate):
    # type: (Tensor, Tuple[Tensor, Tensor], Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    h_tm1, c_tm1 = hidden

    proj_input = torch.sigmoid(F.linear(input, W , bi) + F.linear(h_tm1, R, bh))

    in_gate, out_gate, z_t = proj_input.chunk(3, 1)
    f_gate = f_gate.clamp(0, 1)

    c_t = c_tm1 * f_gate + z_t - in_gate
    h_t = torch.sigmoid(c_t) - out_gate

    return h_t, c_t
