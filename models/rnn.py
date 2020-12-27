import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNMODEL(nn.Module):
    def __init__(self, in_feature, out_feature, hidden, num_layer):
        super(RNNMODEL, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.hidden = hidden
        self.num_layer = num_layer

        self.embed_fc = torch.nn.Linear(in_feature, hidden)

        self.fc1 = torch.nn.Linear(hidden*2, hidden)
        self.fc2 = torch.nn.Linear(hidden, hidden)
        self.fc3 = torch.nn.Linear(hidden, out_feature)

        self.rnncell = torch.nn.LSTM(
            hidden, hidden, num_layer, bidirectional=True, batch_first=True)
        self.activation = torch.nn.ReLU()

    def forward(self, input, input_len):

        x = self.embed_fc(input)

        packed_x = pack_padded_sequence(x, input_len, batch_first=True)
        packed_x, h = self.rnncell(packed_x)
        x, _ = pad_packed_sequence(packed_x, batch_first=True)

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x
