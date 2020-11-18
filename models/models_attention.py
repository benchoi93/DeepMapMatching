
import torch
import torch.nn as nn
from torch.nn.modules import dropout
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderRNN(nn.Module):
    def __init__(self, in_feature, embedding_size, hidden_size, num_layer, dropout):
        super(EncoderRNN, self).__init__()

        self.in_feature = in_feature
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.embed_fc = torch.nn.Linear(in_feature, embedding_size)
        self.embedding = torch.nn.Embedding(in_feature, embedding_size)
        self.rnn = torch.nn.LSTM(
            embedding_size, hidden_size, num_layer, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        # linear layer choose the important factor from forward and backward
        self.fc_hidden = nn.Linear(hidden_size*2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size*2, hidden_size)
        self.activate_embed = torch.nn.ReLU()

    def forward(self, input, input_len):

        #input = self.dropout(self.embed_fc(input))
        embedded = self.dropout(self.embed_fc(input))
        embedded = self.activate_embed(embedded)

        packed_embedded = pack_padded_sequence(embedded, input_len)

        packed_outputs, (hidden, cell) = self.rnn(packed_embedded)

        encoder_states, _ = pad_packed_sequence(packed_outputs)

        ### encoder_states = [seq_len, batch, hidden_size*num_direction]
        # hidden, cell = [n layers * num_directions, batch, hidden_size]

        #x, _ = pad_packed_sequence(encoder_states)

        hidden = torch.tanh(self.fc_hidden(
            torch.cat((hidden[0:1], hidden[1:2]), dim=2)))
        cell = torch.tanh(self.fc_hidden(
            torch.cat((cell[0:1], cell[1:2]), dim=2)))

        #x,_ = pad_packed_sequence(packed_x,batch_first=True)
        return encoder_states, hidden, cell


class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size,
                 output_size, num_layers, dropout):
        super(DecoderRNN, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            output_size, embedding_size)

        self.rnn = nn.LSTM(hidden_size*2+embedding_size,
                           hidden_size, num_layers)

        self.attn = nn.Linear(hidden_size*3, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=0)
        self.Sigmoid = nn.Sigmoid()

        self.fc = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size*3+embedding_size, output_size)

        self.dropout = nn.Dropout(dropout)

        #self.fc_hidden = nn.Linear(1, sequence_length)

       # self.dropout = nn.Dropout(p)

    def forward(self, input, encoder_states, hidden, cell, mask):

        # input = [batch size]
        # hidden = [n layers, batch size, hidden_size]
        # cell = [n layers, batch size, hiddden_size]

        # n directions in the decoder will both always be 1
        input = input.unsqueeze(0)

        # input = [1, batch size]

        #embedding = self.dropout(self.embedding(input))
        embedding = self.dropout(self.embedding(input))

        # embedded = [1, batch size, embbeding_size]

        # to meet the same dimension between encoder_states and hidden, repeat function is used

        ### encoder_states [seq_len, batch_size, hidden*2]
        ### hidden = [1, batch_size, hidden_size]

        # hidden_1 = hidden.permute(2, 1, 0)
        # hidden_1 = self.fc_hidden(hidden_1)
        # h_reshaped = hidden_1.permute(2, 1, 0)

        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)

        ### h_reshaped = [ seq_len, batch_size, hidden_size]

        # energy = self.Sigmoid(self.energy(
        #     torch.cat((h_reshaped, encoder_states), dim=2)))

        energy = torch.tanh(self.attn(
            torch.cat((h_reshaped, encoder_states), dim=2)))

        attention = self.v(energy).squeeze(2)

        #attention = [squ_length, batch]

        attention = attention.masked_fill(mask == 0, -1e10)

        attention_weight = self.softmax(attention)
        # attention_weight = [seq_length,N]

        attention_weight = attention_weight.unsqueeze(2)
        # attention_weight = [seq_length,N,1]

        attention_weight = attention_weight.permute(1, 2, 0)
        # attention_weight =[N,1,seq_length]

        encoder_states = encoder_states.permute(1, 0, 2)
        # encoder_states = [N,seq_length,hidden_size*2]

        # (N,1,hidden_size*2)--> (1,N,hidden_size*2)
        context_vector = torch.bmm(
            attention_weight, encoder_states).permute(1, 0, 2)

        # context vector = [1,N,hidden_size*2]

        rnn_input = torch.cat((context_vector, embedding), dim=2)

        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        # output = [seq len, batch size, hiden_size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
        output = output.squeeze(0)
        embedded = embedding.squeeze(0)
        context_vector = context_vector.squeeze(0)

        prediction = self.fc_out(
            torch.cat((output, context_vector, embedded), dim=1))

        #prediction = self.fc_out(output)

        # prediction = [batch size, output dim]s

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder: EncoderRNN, decoder: DecoderRNN, input_pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.input_pad_idx = input_pad_idx

    def create_mask(self, train):
        mask = (train[:, :, 0] != self.input_pad_idx)
        return mask

    def forward(self, train, train_len, target):
        # train = [len,batch_size,2]
        # target = [len,batch]
        batch_size = target.shape[1]
        target_len = target.shape[0]
        target_output_size = self.decoder.output_size

        # tensor to store decoder output
        outputs = torch.zeros(target_len, batch_size,
                              target_output_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder

        encoder_states, hidden, cell = self.encoder(train, train_len)

        # first input to the decoder  is the  <SOS> tokens In this cas

        input = target[0, :]
        mask = self.create_mask(train)

        for t in range(1, target_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(
                input, encoder_states, hidden, cell, mask)

            # place outputs in a tensor holding it for each token

            outputs[t] = output

            # if (output.sum(1) == 0).sum() > 0:
            # decide if we are going to use teacher forcing or not
            # teacher_force = random.random() < teacher_force_ratio

            # get the highes predicted token from previous result
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token

            input = top1

            # input = target[:, t] if teacher_force else top1

        return outputs

# %%

# %%
