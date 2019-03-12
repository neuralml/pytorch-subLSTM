#include <vector>
#include <torch/torch.h>

#if !defined(_SUBLSTM_H)
#define _SUBLSTM_H

class subLSTM
{
private:
    int numLayers;
    bool batchFirst;
    bool fixFGate;

public:
    subLSTM(int numLayers, bool batchFirst);
    ~subLSTM(int numLayers, bool batchFirst);
    std::tuple<at::Tensor, std::vector<std::tuple<at::Tensor, at::Tensor>>> forward(at::Tensor input,
        std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>> weights,
        std::vector<std::tuple<at::Tensor, at::Tensor>> hidden);
     std::tuple<at::Tensor, std::vector<std::tuple<at::Tensor, at::Tensor>>> subLSTM::forward(
        at::Tensor inputs,
        std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor>> weights,
        at::Tensor f_gate,
        std::vector<std::tuple<at::Tensor, at::Tensor>> hidden)
};

#endif // MACRO
