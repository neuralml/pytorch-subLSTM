#include <iostream>
#include <vector>

#include <torch/torch.h>

#include <subLSTM.h>


at::Tensor d_sigmoid(at::Tensor z){
    auto s = at::sigmoid(z);
    return (1 - s) * s;
}

subLSTM::subLSTM(int numLayers, bool batchFirst)
{
    this->batchFirst = batchFirst;
    this->numLayers = numLayers;
}

subLSTM::~subLSTM()
{
}


 std::tuple<at::Tensor, std::vector<std::tuple<at::Tensor, at::Tensor>>> subLSTM::forward(
    at::Tensor inputs,
    std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>> weights,
    std::vector<std::tuple<at::Tensor, at::Tensor>> hidden){
    
    if (this->batchFirst)
        inputs.transpose(0, 1);

    auto timesteps = inputs.size(0);

    std::vector<at::Tensor> outputs;
    for(size_t i = 0; i < inputs.size(0); i++)
        outputs.push_back(inputs[i]);

    at::Tensor w_i, w_h, b_i, b_h, h_tm1, c_tm1;
    at::Tensor z, f_gate, in_gate, out_gate, c_t, h_t;
    
    for(size_t t = 0; t < timesteps; t++){
        for(size_t l = 0; l < this->numLayers; l++){
            std::tie(w_i, w_h, b_i, b_h) = weights[l];
            std::tie(h_tm1, c_tm1) = hidden[l];
            
            z = torch::nn::Linear(outputs[t], w_i, b_i) + torch::nn::Linear(h_tm1, w_h, b_h);
            z = at::sigmoid(z);

            std::tie(in_gate, out_gate, z, f_gate) = chunk4(z);

            c_t = c_tm1 * f_gate + z - in_gate;
            h_t = at::sigmoid(c_t) - out_gate;

            outputs[t] = h_t;
            hidden[l] = std::make_tuple(h_t, c_t);
        }
    }
    const std::vector<at::Tensor> &ref = outputs;
    return std::make_tuple(at::stack(at::TensorList(&outputs.begin(), &outputs.end()), 0), hidden);
}

 std::tuple<at::Tensor, std::vector<std::tuple<at::Tensor, at::Tensor>>> subLSTM::forward(
    at::Tensor inputs,
    std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor>> weights,
    at::Tensor f_gate,
    std::vector<std::tuple<at::Tensor, at::Tensor>> hidden){
    
    auto timesteps = inputs.size(0);

    std::vector<at::Tensor> outputs;
    for(size_t i = 0; i < inputs.size(0); i++)
        outputs.push_back(inputs[i]);

    at::Tensor w_i, w_h, b_i, b_h, h_tm1, c_tm1;
    at::Tensor z, in_gate, out_gate, c_t, h_t;
    
    for(size_t t = 0; t < timesteps; t++){
        for(size_t l = 0; l < this->numLayers; l++){
            std::tie(w_i, w_h, b_i, b_h) = weights[l];
            std::tie(h_tm1, c_tm1) = hidden[l];
            
            z = torch::nn::Linear(outputs[t], w_i, b_i) + torch::nn::Linear(h_tm1, w_h, b_h);
            z = at::sigmoid(z);

            std::tie(in_gate, out_gate, z) = chunk3(z);

            c_t = c_tm1 * f_gate + z - in_gate;
            h_t = at::sigmoid(c_t) - out_gate;

            outputs[t] = h_t;
            hidden[l] = std::make_tuple(h_t, c_t);
        }
    }
    const std::vector<at::Tensor> &ref = outputs;
    return std::make_tuple(at::stack(at::TensorList(&outputs.begin(), &outputs.end()), 0), hidden);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> chunk4(at::Tensor z){
    auto chunks = z.chunk(4, 1);
    return std::make_tuple(chunks[0], chunks[1], chunks[2], chunks[3]);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> chunk3(at::Tensor z){
    auto chunks = z.chunk(3, 1);
    return std::make_tuple(chunks[0], chunks[1], chunks[2]);
}
