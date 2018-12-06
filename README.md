# subLSTM

This is a PyTorch implementation of subLSTMs as described in the article "Cortical Microcircuits as Gated-Recurrent Units" ([Costa et al., 2017](https://arxiv.org/abs/1711.02448)). The two variants presented in the article are included, the standard model and the one with a constant but learned forget gate (fix-subLSTM). This can be selected by switching the boolean parameter controling the forget gate during initialization (see below).

Currently only the basic Python code using PyTorch is implemented. I'm planning on adding support for Dropout and Seq2Seq models later. I will add a C++ backend if I have the time (and the need to do it).

## Install

To install run the following command from the command line:

```bash
pip install git+https://github.com/mllera14/sublstm.git
```

## Usage

**Parameters**:

Following are the constructor parameters:

| Argument | Default | Description |
| --- | --- | --- |
| input_size | `None` | Size of the input vectors |
| hidden_size | `None` | Number of hidden units |
| num_layers | `1` | Number of layers in the network |
| bias | `True` | Learn the bias. If not, it will be set to 0. |
| batch_first | `False` | Whether data is fed batch first |
| fixed_forget | `False` | Wheter to use subLSTM or fix-subLSMT

## Example usage

### nn Interface

```python
import torch
from subLSTM.nn import SubLSTM

hidden_size = 20
input_size = 10
seq_len = 5
batch_size = 7
hidden = None

input = torch.randn(batch_size, seq_len, input_size)

rnn = SubLSTM(input_size, hidden_size, num_layers=2, bias=True, batch_first=True, fixed_forget=False)

# forward pass
output, hidden = rnn(input, hidden)
```

### Cell Interface

```python
import torch
from subLSTM.nn import fixSubLSTMCell

hidden_size = 20
input_size = 10
seq_len = 5
batch_size = 7
hidden = None

hx = torch.randn(batch_size, hidden_size)
cx = torch.randn(batch_size, hidden_size)

input = torch.randn(batch_size, input_size)

cell = fixSubLSTMCell(input_size, hidden_size, bias=True)
(hx, cx) = cell(input, (hx, cx))

```

## Example Tasks

Two tasks are included at the moment: Sequential MNIST and Delayed Addition. They can be found in the bechmark folders along with bash scripts that run the training with standard parameter values for the two variants implemented (subLSTM w/o fixed forget gates).

## Attributions

A lot of code for the subLSTM was recycled from [pytorch](http://pytorch.org) and the experiment script was based on [ixaxaar](https://github.com/ixaxaar/pytorch-sublstm)'s implementation.