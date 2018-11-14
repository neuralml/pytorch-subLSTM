import torch.nn as nn


class RNNClassifier(nn.Module):
    def __init__(self, rnn, rnn_output_size, n_classes, output_layer='Softmax'):
        super(RNNClassifier, self).__init__()
        self.n_classes = n_classes
        self.rnn_output_size = rnn_output_size
        self.rnn = rnn
        self.linear = nn.Linear(rnn_output_size, n_classes)

        if output_layer == 'Softmax':
            self.output_layer = nn.Softmax(dim=1)
        else:
            raise ValueError()

        # for param in self.parameters():
        #     param.requires_grad = True

    def forward(self, input):
        output = self.rnn(input)[0]
        # output = output[:, -1, :].view(output.size(0), self.rnn_output_size)
        return self.output_layer(self.linear(output[:, -1, :]))
