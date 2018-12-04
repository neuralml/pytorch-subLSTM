import torch
import torch.nn as nn

from subLSTM.nn import SubLSTM

class RNNClassifier(nn.Module):
    def __init__(self, rnn, rnn_output_size, n_classes):
        super(RNNClassifier, self).__init__()
        self.n_classes = n_classes
        self.rnn_output_size = rnn_output_size
        self.rnn = rnn
        self.linear = nn.Linear(rnn_output_size, n_classes)
        self.output_layer = nn.Softmax(dim=1)

    def reset_parameters(self):
        self.rnn.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, input, hidden=None):
        output, hidden = self.rnn(input, hidden)
        probs = self.output_layer(self.linear(output[:, -1, :]))
        
        return probs, hidden


class RNNRegressor(nn.Module):
    def __init__(self, rnn, rnn_output_size, responses):
        super(RNNRegressor, self).__init__()
        self.responses = responses
        self.rnn_output_size = rnn_output_size
        self.rnn = rnn
        self.linear = nn.Linear(rnn_output_size, responses)

    def reset_parameters(self):
        self.rnn.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, input, hidden=None):
        output, hidden = self.rnn(input, hidden)
        predicted = self.linear(output[:, -1, :])
        
        return predicted, hidden


def init_model(model_type, hidden_size, input_size, n_layers,
                output_size, device, class_task=True):
    if model_type == 'subLSTM':
        rnn = SubLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            fixed_forget=False, 
            batch_first=True
        )

    elif model_type == 'fix-subLSTM':
        rnn = SubLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            fixed_forget=True, 
            batch_first=True
        )
    
    elif model_type == 'LSTM':
        rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
        )

    elif model_type == 'GRU':
        rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
        )
        
    else:
        raise ValueError('Unrecognized RNN type')

    if class_task:
        model = RNNClassifier(
            rnn=rnn, rnn_output_size=hidden_size, n_classes=output_size
        ).to(device=device)
    else:
        model = RNNRegressor(
            rnn=rnn, rnn_output_size=hidden_size, responses=output_size
        ).to(device=device)

    return model