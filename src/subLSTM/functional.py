import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.jit.script
def slstm_cell(input, h_tm1, c_tm1, W, R, bi, bh):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    proj_input = torch.sigmoid(torch.addmm(bi, input, W.t()) + torch.addmm(bh, h_tm1, R.t()))
    in_gate, out_gate, z_t, f_gate = proj_input.chunk(4, 1)

    c_t = c_tm1 * f_gate + z_t * in_gate
    h_t = torch.sigmoid(c_t) - out_gate

    return h_t, c_t


@torch.jit.script
def fslstm_cell(input, h_tm1, c_tm1, W, R, bi, bh, f_gate):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]

    proj_input = torch.sigmoid(torch.addmm(bi, input, W.t()) + torch.addmm(bh, h_tm1, R.t()))

    in_gate, out_gate, z_t = proj_input.chunk(3, 1)
    f_gate = f_gate.clamp(0, 1)

    c_t = c_tm1 * f_gate + z_t - in_gate
    h_t = torch.sigmoid(c_t) - out_gate

    return h_t, c_t


@torch.jit.script
def sublstm_forward(input, hidden, weights, num_layers, dropout, training):
    # type: (List[Tensor], Tuple[Tensor, Tensor], List[Tensor], int, float, bool) -> Tuple[List[Tensor], Tuple[Tensor, Tensor]]
    timesteps = len(input)
    hx, cx = hidden

    for time in range(timesteps):
        new_h, new_c = torch.zeros_like(hx), torch.zeros_like(cx)
        for l in range(num_layers):
            W, R, bi, bh = weights[4*l: 4*(l + 1)]
            h, c = slstm_cell(input[time], hx[l], cx[l], W, R, bi, bh)

            if l < num_layers - 1 and dropout > 0:
                h = F.dropout(h, dropout, training)

            new_h[l], new_c[l] = h, c
            input[time] = h

        hx, cx = new_h, new_c

    return input, (hx, cx)


@torch.jit.script
def fsublstm_forward(input, hidden, weights, num_layers, dropout, training):
    # type: (List[Tensor], Tuple[Tensor, Tensor], List[Tensor], int, float, bool) -> Tuple[List[Tensor], Tuple[Tensor, Tensor]]
    timesteps = len(input)
    hx, cx = hidden

    for l in range(len(weights)):
        weights[l*5] = weights[l*5].t()
        weights[l*5+1] = weights[l*5+1].t()

    for time in range(timesteps):
        new_h, new_c = torch.zeros_like(hx), torch.zeros_like(cx)
        for l in range(num_layers):
            W, R, bi, bh, f_gate = weights[5*l: 5*(l + 1)]
            h, c = fslstm_cell(input[time], hx[l], cx[l], W, R, bi, bh, f_gate)

            if l < num_layers - 1 and dropout > 0:
                h = F.dropout(h, dropout, training)

            new_h[l], new_c[l] = h, c
            input[time] = h

        hx, cx = new_h, new_c

    return input, (hx, cx)
