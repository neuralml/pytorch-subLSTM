import math

import torch
import torch.nn as nn
from torch.nn.modules.rnn import RNNCellBase
# from torch.nn.modules.rnn import PackedSequence
# from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as pad

from src.functional import fixSubLSTMCellF, SubLSTMCellF


# noinspection PyPep8Naming,PyShadowingBuiltins
class SubLSTMCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True, fix_subLSTM=False):
        super(SubLSTMCell, self).__init__()

        # Set the parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self._fix_f_gate = fix_subLSTM

        gate_size = (3 if fix_subLSTM else 4) * hidden_size

        self.W_i = nn.Parameter(torch.Tensor(gate_size, input_size))
        self.W_h = nn.Parameter(torch.Tensor(gate_size, hidden_size))

        if fix_subLSTM:
            self.f_gate = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('f_gate', None)

        if bias:
            self.b_i = nn.Parameter(torch.Tensor(gate_size))
            self.b_h = nn.Parameter(torch.Tensor(gate_size))
        else:
            self.register_parameter('b_i', None)
            self.register_parameter('b_h', None)

        self.reset_parameters()

    @property
    def is_fix_subLSTM(self):
        return self._fix_f_gate

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

    def forward(self, input: torch.Tensor, hx=None):
        self.check_forward_input(input)

        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            hx = (hx, hx)

        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')

        if self._fix_f_gate:
            return fixSubLSTMCellF(input, hx, self.W_i, self.W_h, self.f_gate, self.b_i, self.b_h)
        return SubLSTMCellF(input, hx, self.W_i, self.W_h, self.b_i, self.b_h)


# noinspection PyShadowingBuiltins,PyPep8Naming
class SubLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, fixed_forget=True, batch_first=False):

        super(SubLSTM, self).__init__()

        if isinstance(hidden_size, list) and len(hidden_size) != num_layers:
            raise ValueError(
                'Length of hidden_size list is not the same as num_layers. Expected {0} got {1}'.format(
                    num_layers, len(hidden_size))
            )

        if isinstance(hidden_size, int):
            hidden_size = [hidden_size] * num_layers

        # Some python "magic" to assign all parameters as class attributes
        self.__dict__.update(locals())

        num_gates = 3 if fixed_forget else 4

        self._all_weights = []
        # Use for bidirectional later
        suffix = ''

        for layer_num in range(num_layers):

            layer_in_size = input_size if layer_num == 0 else hidden_size[layer_num - 1]
            layer_out_size = hidden_size[layer_num]

            gate_size = num_gates * layer_out_size

            w_i = nn.Parameter(torch.Tensor(gate_size, layer_in_size))
            w_h = nn.Parameter(torch.Tensor(gate_size, layer_out_size))

            layer_param = [w_i, w_h]

            name_template = ['W_{}{}', 'R_{}{}']

            if bias:
                b_i = nn.Parameter(torch.Tensor(gate_size))
                b_h = nn.Parameter(torch.Tensor(gate_size))

                layer_param.extend([b_i, b_h])
                name_template.extend(['b_i_{}{}', 'b_r_{}{}'])

            if fixed_forget:
                f = nn.Parameter(torch.Tensor(hidden_size[layer_num]))

                layer_param.append(f)
                name_template.append('f_{}{}')

            param_name = [x.format(layer_num, suffix) for x in name_template]
            for name, value in zip(param_name, layer_param):
                setattr(self, name, value)

            self._all_weights.append(param_name)

        self.flatten_parameters()
        self.reset_parameters()

    @property
    def all_weights(self):
        return [[getattr(self, weights) for weights in weights] for weights in self._all_weights]

    def flatten_parameters(self):
        pass

    def reset_parameters(self):
        for hidden_size in self.hidden_size:
            std = 1.0 / math.sqrt(hidden_size)
            for weight in self.parameters():
                weight.data.uniform_(-std, std)

    def forward(self, input: torch.Tensor, hx: torch.Tensor=None):

        # TODO: Check docs later and add the packed sequence option and the bidirectional version
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
                hidden = torch.zeros((max_batch_size, self.hidden_size[l]), requires_grad=False)
                hx.append((hidden, hidden))

        Ws = self.all_weights
        if self.batch_first:
            input = input.transpose(0, 1)

        timesteps = input.size(1) if self.batch_first else input.size(0)
        outputs = [input[i] for i in range(timesteps)]

        for time in range(timesteps):
            for layer in range(self.num_layers):
                if self.fixed_forget:
                    if self.bias:
                        w_i, w_h, b_i, b_h, f = Ws[layer]
                    else:
                        w_i, w_h, f = Ws[layer]
                        b_i, b_h = None, None

                    out, c = fixSubLSTMCellF(outputs[time], hx[layer], w_i, w_h, f, b_i, b_h)

                else:
                    if self.bias:
                        w_i, w_h, b_i, b_h = Ws[layer]
                    else:
                        w_i, w_h = Ws[layer]
                        b_i, b_h, f = None, None, None

                    out, c = SubLSTMCellF(outputs[time], hx[layer], w_i, w_h, b_i, b_h)

                hx[layer] = (out, c)
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
