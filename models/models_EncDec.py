import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderRNN(nn.Module):
    def __init__(self, in_feature, embedding_size, hidden_size, num_layer, dropout):
        super(EncoderRNN, self).__init__()
        self.in_feature = in_feature
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.embed_fc = torch.nn.Linear(in_feature, embedding_size)
        self.rnn = torch.nn.LSTM(
            embedding_size, hidden_size, num_layer)
        self.dropout = nn.Dropout(dropout)
        self.activate_embed = nn.ReLU()

    def forward(self, input, input_len):

        embedded = self.embed_fc(input)
        embedded = self.activate_embed(embedded)
        packed_embedded = pack_padded_sequence(embedded, input_len)

        _, (hidden, cell) = self.rnn(packed_embedded)

        return hidden, cell


class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size,
                 output_size, num_layer, dropout):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(
            output_size, embedding_size, padding_idx=0)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layer)
        self.fc = nn.Linear(hidden_size, output_size)
        self.output_size = output_size
        self.softmax = nn.Softmax(dim=1)
        self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        output = output.squeeze(0)
        output = self.activation(output)
        output = self.fc(output)

        return output, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder: EncoderRNN, decoder: DecoderRNN, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, train, train_len, target):
        # train = [batch,len,2]
        # target = [batch,len]
        batch_size = target.shape[1]
        target_len = target.shape[0]
        target_output_size = self.decoder.output_size

        # tensor to store decoder output
        outputs = torch.zeros(target_len, batch_size,
                              target_output_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder

        hidden, cell = self.encoder(train, train_len)

        # first input to the decoder  is the  <SOS> tokens

        input = target[0, :]

        for t in range(1, target_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            # place outputs in a tensor holding it for each token

            outputs[t] = output

            # if (output.sum(1) == 0).sum() > 0:
            #     print(1)
            # decide if we are going to use teacher forcing or not
            # teacher_force = random.random()<teacher_force_ratio

            # get the highest predictd token from output
            # input = output.argmaxf(1)
            # prob = torch.softmax(output, 1)
            # dist = torch.distributions.Categorical(prob)
            # input = dist.sample()
            input = output.argmax(1)

        return outputs
