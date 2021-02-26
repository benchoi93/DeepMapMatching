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
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Linear(src_vocab_size, embedding_size)
        # self.activation_1 = nn.ReLU()
        # self.activation_2 = nn.Sigmoid()
        # self.src_linear = nn.Linear(embedding_size, embedding_size)
        # self.trg_linear = nn.Linear(embedding_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            embedding_size, num_heads, forward_expansion, dropout)
        self.decoder_layer = nn.TransformerDecoderLayer(
            embedding_size, num_heads, forward_expansion, dropout)
        # self.decoder_norm = nn.LayerNorm(embedding_size)
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, trg):

        # src=[src_seq_len,N,2]
        #trg = [trg_seq,N]
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

        src_word = self.src_word_embedding(src)
        # src_word = self.activation_1(self.src_linear(src_word))
        # src_word = self.activation_1(self.src_linear(src_word))

        src_pos = self.src_position_embedding(src_positions)
        # src_pos = self.activation(self.src_linear(src_pos))
        # src_pos = self.activation(self.src_linear(src_pos))

        trg_word = self.trg_word_embedding(trg)
        # trg_word = self.activation(self.trg_linear(trg_word))
        # trg_word = self.activation(self.trg_linear(trg_word))

        trg_pos = self.trg_position_embedding(trg_positions)
        # trg_pos = self.activation(self.trg_linear(trg_pos))
        # trg_pos = self.activation(self.trg_linear(trg_pos))

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

        # out, enc_attn, dec_attn = self.transformer(
        #     embed_src,
        #     embed_trg
        #     ,src_key_padding_mask=src_padding_mask,
        #     tgt_mask=trg_mask)

        memory, enc_attn = self.transformer.encoder(
            embed_src, src_key_padding_mask=src_padding_mask)
        out, dec_attn = self.transformer.decoder(embed_trg, memory, trg_mask)

        # enc_attn_weight = self.encoder_layer.self_attn(
        #     embed_src, embed_src, embed_src, key_padding_mask=src_padding_mask)[1]

        # # enc_atten_weight = [N,L,S] where N is the batch size, L is the target sequene length, S is the source sequence length
        # # dec_attn_weight = self.decoder_layer.self_attn(
        # #     embed_trg, embed_trg, embed_trg, attn_mask=trg_mask)[1]
        # tgt2 = self.decoder_layer.self_attn(
        #     embed_trg, embed_trg, embed_trg, attn_mask=trg_mask)[0]
        # tgt = embed_trg + self.decoder_layer.dropout1(tgt2)
        # tgt = self.decoder_layer.norm1(tgt)
        # tgt2 = self.decoder_layer.multihead_attn(tgt, memory, memory)[0]
        # dec_attn_weight = self.decoder_layer.multihead_attn(tgt, memory, memory)[
        #     1]

        # out = self.transformer.decoder(embed_trg, memory, trg_mask)

        # tgt = tgt + self.decoder_layer.dropout2(tgt2)
        # tgt = self.decoder_layer.norm2(tgt)
        # tgt2 = self.decoder_layer.linear2(self.decoder_layer.dropout(self.decoder_layer.activation(self.decoder_layer.linear1(tgt))))
        # tgt = tgt + self.decoder_layer.dropout3(tgt2)
        # tgt = self.decoder_layer.norm3(tgt)

        out = self.fc_out(out)
        return (out, enc_attn, dec_attn, memory)
