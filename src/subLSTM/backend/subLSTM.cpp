#include <iostream>
#include <vector>

#include <torch/torch.h>


at::Tensor d_sigmoid(at::Tensor z){
    auto s = at::sigmoid(z);
    return (1 - s) * s;
}

std::vector<at::Tensor> _sLSTM_cell_forward(
        at::Tensor input,
        at::Tensor W,
        at::Tensor R,
        at::Tensor bi,
        at::Tensor bh
        at::Tensor h_tm1,
        at::Tensor c_tm1) {

    auto pre_act_gates = at::addmm(bi, input, W) + at::addmm(bh, h_tm1, R);
    auto gates = at::sigmoid(pre_act_gates).chunk(4, dim=1)

    auto in_gate = gates[0];
    auto f_gate = gates[1];
    auto z = gates[2];
    auto out_gate = gates[3];

    auto c_t = f_gate * c_tm1 + z - in_gate
    auto h_t = at::sigmoid(c_t) - out_gate

    return {h_t, c_t, h_tm1, c_tm1, pre_act_gates, f_gate};
}

std::vector<at::Tensor> _sLSTM_cell_backward(
        at::Tensor grad_h,
        at::Tensor grad_cell,
        at::Tensor cell_t,
        at::Tensor cell_tm1,
        at::Tensor in_t,
        at::Tensor h_tm1,
        at::Tensor pre_act_gates,
        at::Tensor f_gate,
        at::Tensor W,
        at::Tensor R) {

    auto d_cell = grad_h * d_sigmoid(cell_t) + grad_cell;
    // grads w.r.t. in_gate, f_gate, z and out_gate
    auto grads = torch.cat({-d_cell, d_cell * cell_tm1, d_cell, -grad_h}, /*dim*/1);

    grads *= d_sigmoid(pre_act_gates);
    auto t_grads = grads.t();

    // Compute the gradients
    auto grad_W = t_grads.mm(in_t)
    auto grad_R = t_grads.mm(h_tm1);
    auto grad_bias = grads.sum(/*dim=*/0, /*keepdim=*/true);

    // Compute errors
    auto grad_h_tm1 = grads.mm(R);
    auto grad_input = grads.mm(W);
    auto grad_cell_tm1 = grad_cell * f_gate;

    return {grad_h_tm1, grad_cell_tm1, grad_input, grad_w, grad_R, grad_bias};
}

// std::vector<at::Tensor> SubLSTM_forward(
//     at::Tensor input,
//     vector<at::Tensor> hidden,
//     vector<at::Tensor> weights,
//     vector<at::Tensor> bias,
//     float dropout,
//     bool training,
// ){
//     auto timesteps = input.size();
//     auto num_layers = weights.size();

//     auto hx = hidden[0];
//     auto cx = hidden[1];

//     vector<at::Tensor> cell_output = NULL;
//     at::Tensor new_hx = NULL;
//     at::Tensor new_cx = NULL;

//     for(size_t t = 0; t < timesptes; t++) {
//         for(size_t l = 0; l < num_layers; l++) {

//             cell_output = _sLSTM_cell_forward(input[t], weights[l], bias[l], hx[l], cx[l]);
//             input[t] = hx[l] = cell_output[0];
//             cx[l] = cell_output[1];
//         }
//     }

//     auto ouput = torch.stack(input);

//     return {output, hidden};
// }