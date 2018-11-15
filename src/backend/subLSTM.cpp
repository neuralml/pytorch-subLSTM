#include <iostream>
#include <torch/torch.h>
#include <vector>


at::Tensor d_sigmoid(at::Tensor z){
    auto s = at::sigmoid(z);
    return (1 - s) * s;
}
