import math
from itertools import product
import numbers

import torch
import torch.nn as nn
import torch.jit as jit
from torch.nn.modules.activation import Sigmoid
# from torch.nn.modules.rnn import PackedSequence
# from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as pad

from .functional import sublstm, fsublstm

@jit.script
def _sublstm_forward(input, hx, weights, num_layers):
    # type: (List[Tensor], List[Tuple[Tensor, Tensor]], List[List[Tensor]], int) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
    timesteps = len(input)

    for time in range(timesteps):
        for l in range(num_layers):
            W, R, bi, bh = weights[l]

            out, c = sublstm(input[time], hx[l], W, R, bi, bh)

            # if l < num_layers - 1 and dropout:
            #     out = dropout(out)

            hx[l] = (out, c)
            input[time] = out

    output = torch.stack(input)

    return output, hx

@jit.script
def _fsublstm_forward(input, hx, weights, num_layers):
    # type: (List[Tensor], List[Tuple[Tensor, Tensor]], List[List[Tensor]], int) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
    timesteps = len(input)

    for time in range(timesteps):
        for l in range(num_layers):
            W, R, bi, bh, f_gate = weights[l]

            out, c = fsublstm(input[time], hx[l], W, R, bi, bh, f_gate)

            # if l < num_layers - 1 and dropout:
            #     out = dropout(out)

            hx[l] = (out, c)
            input[time] = out

    output = torch.stack(input)

    return output, hx


# noinspection PyShadowingBuiltins,PyPep8Naming
class SubLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                    fixed_forget=True, batch_first=False, dropout=0.0, bidirectional=False):

        super(SubLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.fixed_forget = fixed_forget
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.mode = 'fix-subLSTM' if fixed_forget else 'subLSTM'

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        # if dropout > 0 and num_layers == 1:
        #     warnings.warn("dropout option adds dropout after all but last "
        #                   "recurrent layer, so non-zero dropout expects "
        #                   "num_layers greater than 1, but got dropout={} and "
        #                   "num_layers={}".format(dropout, num_layers))

        num_directions = 2 if bidirectional else 1
        gate_size = (3 if fixed_forget else 4) * hidden_size
        self._all_weights = []

        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                W = nn.Parameter(torch.Tensor(gate_size, layer_input_size))
                R = nn.Parameter(torch.Tensor(gate_size, hidden_size))

                layer_param = [W, R]
                name_template = ['W_{}{}', 'R_{}{}']

                if bias:
                    b_i = nn.Parameter(torch.Tensor(gate_size))
                    b_h = nn.Parameter(torch.Tensor(gate_size))
                else:
                    b_i, b_h = torch.tensor(0), torch.tensor(0)

                layer_param.extend([b_i, b_h])
                name_template.extend(['bi_{}{}', 'bh_{}{}'])

                if fixed_forget:
                    f = nn.Parameter(torch.Tensor(hidden_size))

                    layer_param.append(f)
                    name_template.append('f_{}{}')

                param_names = [x.format(layer, suffix) for x in name_template]
                for name, value in zip(param_names, layer_param):
                    setattr(self, name, value)

                self._all_weights.append(param_names)

        self.flatten_parameters()
        self.reset_parameters()

    def flatten_parameters(self):
        """Resets parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        any_param = next(self.parameters()).data
        if not any_param.is_cuda or not torch.backends.cudnn.is_acceptable(any_param):
            return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        all_weights = self._flat_weights
        unique_data_ptrs = set(p.data_ptr() for p in all_weights)
        if len(unique_data_ptrs) != len(all_weights):
            return

        with torch.cuda.device_of(any_param):
            import torch.backends.cudnn.rnn as rnn

            # NB: This is a temporary hack while we still don't have Tensor
            # bindings for ATen functions
            with torch.no_grad():
                # NB: this is an INPLACE function on all_weights, that's why the
                # no_grad() is necessary.
                torch._cudnn_rnn_flatten_weight(
                    all_weights, (4 if self.bias else 2) + (1 if self.fixed_forget else 0),
                    self.input_size, rnn.get_cudnn_mode('LSTM'), self.hidden_size, self.num_layers,
                    self.batch_first, bool(self.bidirectional))
        # pass

    def _apply(self, fn):
        ret = super(SubLSTM, self)._apply(fn)
        self.flatten_parameters()
        return ret

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def check_forward_args(self, input, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = 2 if is_input_packed else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)

        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)

        def check_hidden_size(hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
            if tuple(hx.size()) != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

            check_hidden_size(hidden[0], expected_hidden_size,
                                'Expected hidden[0] size {}, got {}')
            check_hidden_size(hidden[1], expected_hidden_size,
                                'Expected hidden[1] size {}, got {}')

    def forward(self, input, hx=None):
        if self.batch_first:
            input = input.transpose(0, 1)

        batch_size = input.size(1)

        if hx is None:
            hx = []
            for l in range(self.num_layers):
                # use input.new_zeros so dtype and device are the same as the input's
                hidden = input.new_zeros(
                    (batch_size, self.hidden_size), requires_grad=False)
            hx.append((hidden, hidden))

        self.check_forward_args(input, hx, None)

        timesteps = input.size(0)
        output = [input[i] for i in range(timesteps)]

        if self.fixed_forget:
            output, hidden = _fsublstm_forward(output, hx, self.all_weights, self.num_layers)
        else:
            output, hidden  = _sublstm_forward(output, hx, self.all_weights, self.num_layers)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    @property
    def _flat_weights(self):
        return [p for layerparams in self.all_weights for p in layerparams]

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]
