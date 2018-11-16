import numpy as np
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

        self._mode = True

    def forward(self, input):
        output = self.rnn(input)[0]
        probs = self.output_layer(self.linear(output[:, -1, :]))
        
        if self._mode:
            return probs
        return np.argmax(probs, axis=0)

    def set_train(self):
        self._mode = True

    def set_eval(self):
        self._mode = False
