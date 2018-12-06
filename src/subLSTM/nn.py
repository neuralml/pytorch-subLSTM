import math
from itertools import product

import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.rnn import RNNCellBase
# from torch.nn.modules.rnn import PackedSequence
# from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as pad


class SubLSTMCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True):
        super(SubLSTMCell, self).__init__()

        # Set the parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        gate_size = 4 * hidden_size

        self.input_layer = nn.Linear(input_size, gate_size, bias=bias)
        self.recurrent_layer = nn.Linear(hidden_size, gate_size, bias=bias)

        self.gates = Sigmoid()
        self.output = Sigmoid()

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.children():
            try:
                module.reset_parameters()
            except AttributeError:
                pass

    def forward(self, input: torch.Tensor, hx):
        h_tm1, c_tm1 = hx, hx

        proj = self.gates(self.input_layer(input) + self.recurrent_layer(h_tm1))
        in_gate, out_gate, z_t, f_gate = proj_input.chunk(4, 1)

        c_t = c_tm1 * f_gate + z_t - in_gate
        h_t = self.output(c_t) - out_gate

        return h_t, c_t


class fixSubLSTMCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True):
        super(SubLSTMCell, self).__init__()

        # Set the parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        gate_size = 3 * hidden_size

        self.input_layer = nn.Linear(input_size, gate_size, bias=bias)
        self.recurrent_layer = nn.Linear(hidden_size, gate_size, bias=bias)
        self.f_gate = nn.Parameter(torch.Tensor(hidden_size))

        self.gates = Sigmoid()
        self.output = Sigmoid()

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.children():
            try:
                module.reset_parameters()
            except AttributeError:
                pass

    def forward(self, input, hx):
        h_tm1, c_tm1 = hx, hx

        proj = self.gates(self.input_layer(input) + self.recurrent_layer(h_tm1))
        in_gate, out_gate, z_t = proj_input.chunk(3, 1)
        f_gate = self.f_gate.clamp(0, 1)

        c_t = c_tm1 * f_gate + z_t - in_gate
        h_t = self.output(c_t) - out_gate

        return h_t, c_t


# noinspection PyShadowingBuiltins,PyPep8Naming
class SubLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                    fixed_forget=True, batch_first=False):

        super(SubLSTM, self).__init__()

        # Uncomment to get layers of different size. Disable for consistency with LSTM
        # if isinstance(hidden_size, list) and len(hidden_size) != num_layers:
        #     raise ValueError(
        #         'Length of hidden_size list is not the same as num_layers.'
        #         'Expected {0} got {1}'.format(
        #             num_layers, len(hidden_size))
        #     )

        # if isinstance(hidden_size, int):
        #     hidden_size = [hidden_size] * num_layers

        # Some python "magic" to assign all parameters as class attributes
        self.__dict__.update(locals())

        # Use for bidirectional later
        suffix = ''

        for layer_num in range(num_layers):

            layer_in_size = input_size if layer_num == 0 else hidden_size
            layer_out_size = hidden_size

            if fixed_forget:
<<<<<<< Updated upstream
                f = nn.Parameter(torch.Tensor(hidden_size))

                layer_param.append(f)
                name_template.append('f_{}{}')

            param_names = [x.format(layer_num, suffix) for x in name_template]
            for name, value in zip(param_names, layer_param):
                setattr(self, name, value)
=======
                layer = fixSubLSTMCell(layer_in_size, layer_out_size, bias)
            else:
                layer = SubLSTMCell(layer_in_size, layer_out_size, bias)
>>>>>>> Stashed changes

            self.add_module('layer_{}'.format(layer_num + 1), layer)

        self.flatten_parameters()
        self.reset_parameters()

    @property
    def all_weights(self):
        return [[getattr(self, name) for name in param_names] 
            for param_names in self._all_params]

    @property
    def all_layers(self):
        return [getattr(self, 'layer_{}'.format(layer + 1)) for layer in range(self.num_layers)]

    def reset_parameters(self):
<<<<<<< Updated upstream
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for l in range(self.num_layers):
            for weight in self.all_weights[l]:
                weight.data.uniform_(-stdv, stdv)
=======
        for module in self.children():
            module.reset_parameters()
>>>>>>> Stashed changes

    def flatten_parameters(self):
        pass

    def forward(self, input, hx=None):
        # TODO: Check docs later and add the packed sequence and seq2seq models
        # is_packed = isinstance(input, PackedSequence)
        #
        # if is_packed:
        #     input, batch_size = pad(input)
        #     max_batch_size = batch_size[0]
        # else:
        #     batch_size = None
        #     max_batch_size = input.size(0) if self.batch_first else input.size(1)

        max_batch_size = input.size(0) if self.batch_first else input.size(1)

        if hx is None:
            hx = []
            for l in range(self.num_layers):
                # use input.new_zeros so dtype and device are the same as the input's
                hidden = input.new_zeros(
                    (max_batch_size, self.hidden_size), requires_grad=False)
                hx.append((hidden, hidden))

        if self.batch_first:
            input = input.transpose(0, 1)

        timesteps = input.size(0)
        outputs = [input[i] for i in range(timesteps)]
        all_layers = self.all_layers

        for time, l in product(range(timesteps), range(self.num_layers)):
            layer = all_layers[l]

            out, c = layer(outputs[time], hx[l])
            hx[l] = (out, c)
            outputs[time] = out

        out = torch.stack(outputs)
        if self.batch_first:
            out = out.transpose(0, 1)

        # TODO: Check docs later and add the packed sequence option
        # if is_packed:
        #     out = pack(out, batch_size)

        return out, hx

    def _apply(self, fn):
        ret = super(SubLSTM, self)._apply(fn)
        self.flatten_parameters()
        return ret

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        # if self.dropout != 0:
        #     s += ', dropout={dropout}'
        # if self.bidirectional is not False:
        #     s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

