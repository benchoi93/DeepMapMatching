import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        device,
        max_len,
        kernel_size,
        conv_stride,
        pool_size,
        pool_stride
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Linear(src_vocab_size, embedding_size)
        self.src_conv_embedding = nn.Conv1d(1, 1, kernel_size, conv_stride)
        self.activation = nn.ReLU()
        self.src_pooling = nn.MaxPool2d(pool_size, pool_stride)
        self.linear = nn.Linear(embedding_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(
            trg_vocab_size, embedding_size, padding_idx=0)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.embedding = nn.Sequential(self.linear,
                                       self.activation)
        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, trg):

        src = src.to(self.device)
        con1 = self.src_conv_embedding(src.unsqueeze(1))
        con1 = self.activation(con1)
        pool1 = self.src_pooling(con1)
        pool1 = pool1.squeeze(1)

        src = pool1.permute(1, 0, 2).to(self.device)
        trg = trg.permute(1, 0).to(self.device)

        src_seq_length, N, _ = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(1)
            .expand(src_seq_length, N)
            .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length)
            .unsqueeze(1)
            .expand(trg_seq_length, N)
            .to(self.device)
        )

        src_word = self.activation(self.src_word_embedding(src))
        src_word = self.embedding(src_word)
        src_word = self.embedding(src_word)

        src_pos = self.src_position_embedding(src_positions)
        # src_pos = self.embedding(src_pos)
        # src_pos = self.embedding(src_pos)

        trg_word = self.trg_word_embedding(trg)
        trg_word = self.embedding(trg_word)
        trg_word = self.embedding(trg_word)

        trg_pos = self.trg_position_embedding(trg_positions)

        embed_src = self.dropout(
            (src_word +
             src_pos)
        )
        embed_trg = self.dropout(
            (trg_word +
             trg_pos)
        )
        src_1 = src[:, :, 0]
        src_padding_mask = self.make_src_mask(src_1)
        trg_mask = self.transformer.generate_square_subsequent_mask(
            trg_seq_length).to(self.device)

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out


# #%%
# import numpy as np
# def get_sinusoid_encoding_table(n_seq, d_hidn):
#     def cal_angle(position, i_hidn):
#         return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)
#     def get_posi_angle_vec(position):
#         return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

#     sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
#     sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index sin
#     sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index cos

#     return sinusoid_table
# # %%
# n_seq = 64

# pos_encoding = get_sinusoid_encoding_table(64,256)

# print(pos_encoding)
# # %%
# pos_encoding = torch.FloatTensor(pos_encoding)
# nn_pos = nn.Embedding.from_pretrained(pos_encoding, freeze=True)

# positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(inputs.size(0), inputs.size(1)).contiguous() + 1
# pos_mask = inputs.eq(0)

# positions.masked_fill_(pos_mask, 0)
# pos_embs = nn_pos(positions) # position embedding
