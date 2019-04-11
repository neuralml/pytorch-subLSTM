import sublsmt

class sLSTMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, hx, cx, W, R, bi, bh):
        outputs = sublstm.forward(input, W, R, bi, bh, hx, cx)
        new_h, new_cell = outputs[:2]
        variables = outputs[1:] + [W, R, input, hx, cx]
        ctx.save_for_backward(*variables)

        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):

        # print(ctx.saved_variables[4].shape)
        # print(grad_h.shape)

        outputs = sublstm.backward(
            grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_variables)
        d_input, d_old_cell, d_old_h, dW, dR, d_bias = outputs
        return d_input, d_old_h, d_old_cell, dW, dR, d_bias, d_bias


def sublstm_forward(input, hidden, weights, num_layers):
    # type: (List[Tensor], Tuple[Tensor, Tensor], List[Tensor], int) -> Tuple[List[Tensor], Tuple[Tensor, Tensor]]
    timesteps = len(input)
    hx, cx = hidden

    for l in range(num_layers):
        weights[l*4] = weights[l*4].t()
        weights[l*4+1] = weights[l*4+1].t()

    for time in range(timesteps):
        new_h, new_c = torch.zeros_like(hx), torch.zeros_like(cx)
        for l in range(num_layers):
            W, R, bi, bh = weights[4*l: 4*(l + 1)]
            h, c = slstm_cell(input[time], hx[l], cx[l], W, R, bi, bh)

            # if l < num_layers - 1 and dropout > 0:
            #     out = dropout(out)

            new_h[l], new_c[l] = h, c
            input[time] = h

        hx, cx = new_h, new_c

    return input, (hx, cx)