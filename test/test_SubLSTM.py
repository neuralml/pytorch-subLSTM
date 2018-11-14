import sys
import torch

sys.path.insert(0, '.')

from src.subLSTM import SubLSTM

timestep, batch_size, in_size = 40, 20, 50
input = torch.randn(timestep, batch_size, in_size, requires_grad=False)


def one_layer_test():
    hidden_size = 100
    slstm = SubLSTM(in_size, hidden_size, num_layers=1, fixed_forget=False)

    out, hx = slstm.forward(input)

    assert out.size() == (timestep, batch_size, hidden_size)
    assert hx[0][0].size() == (batch_size, hidden_size)
    assert hx[0][1].size() == (batch_size, hidden_size)

    out.sum().backward()

    fix_slstm = SubLSTM(in_size, hidden_size, num_layers=1, fixed_forget=True)

    out, hx = fix_slstm.forward(input)

    assert out.size() == (timestep, batch_size, hidden_size)
    assert hx[0][0].size() == (batch_size, hidden_size)
    assert hx[0][1].size() == (batch_size, hidden_size)

    out.sum().backward()


def multiple_layer_test():
    hidden_size = [100, 50]
    num_layers = 2

    slstm = SubLSTM(in_size, hidden_size, num_layers=num_layers, fixed_forget=False)

    out, hx = slstm.forward(input)

    assert out.size() == (timestep, batch_size, hidden_size[-1])

    for l in range(num_layers):
        assert hx[l][0].size() == (batch_size, hidden_size[l])
        assert hx[l][1].size() == (batch_size, hidden_size[l])

    out.sum().backward()

    fix_slstm = SubLSTM(in_size, hidden_size, num_layers=num_layers, fixed_forget=True)

    out, hx = fix_slstm.forward(input)

    assert out.size() == (timestep, batch_size, hidden_size[-1])
    for l in range(num_layers):
        assert hx[l][0].size() == (batch_size, hidden_size[l])
        assert hx[l][1].size() == (batch_size, hidden_size[l])

    out.sum().backward()


if __name__ == '__main__':
    multiple_layer_test()
