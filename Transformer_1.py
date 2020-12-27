import os
import pandas as pd
from numpy.lib.function_base import average
from tensorboardX import SummaryWriter
import tensorboardX
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time

from models.Transformer import *

from src.util import *
import argparse
import time
import glob
import sys
# %%
sys.argv = ['']
# temporary code for debugging

torch.cuda.empty_cache()

model_summary_writer = SummaryWriter(
    'log/encoder_decoder_{}'.format(time.time()/60))

parser = argparse.ArgumentParser()
#parser.add_argument('--gps-data', default="data/GPS/GPS_0.npy", type=str)
parser.add_argument(
    '--gps_data', default="data/training_data/GPSmax7_new_6.npy", type=str)
parser.add_argument(
    '--label_data', default="data/training_data/Label_smax7_new_6.npy", type=str)
# parser.add_argument('--label_data', default="Label_1.npy", type=str)
parser.add_argument('--train-ratio', default=0.7, type=float)
parser.add_argument('--learning-rate', default=3e-4, type=float)
# parser.add_argument('--training_num', default=100, type=int)
parser.add_argument('--batch_size', default=100, type=int)


args = parser.parse_args()


# initialize dataset
#raw_input = datacombination("data/GPS/*.npy")
#raw_target = datacombination("data/Label/*.npy")
raw_input = np.load(args.gps_data)[0:1000]
raw_target = np.load(args.label_data)[0:1000]
raw_target[raw_target < 0] = 0
raw_target = raw_target[:, 1:]

# %%
# device setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


raw_input = normalization(raw_input)
train_ratio = args.train_ratio
test_ratio = 1-train_ratio
n_data = raw_target.shape[0]
n_train = int(n_data*train_ratio)
n_test = n_data-n_train

randidx = np.sort(np.random.permutation(n_data))
train_idx = randidx[:n_train]
test_idx = randidx[n_train:(n_train+n_test)]


train_input = raw_input[train_idx]
train_target = raw_target[train_idx]
train_len = train_input[:, :, 0] != -1
train_len = train_len.sum(axis=1)

test_input = raw_input[test_idx]
test_target = raw_target[test_idx]
test_len = test_input[:, :, 0] != -1
test_len = test_len.sum(axis=1)


# change to tensor
train_input = torch.Tensor(train_input)
train_target = torch.LongTensor(train_target)
train_target = train_target.squeeze()
train_len = torch.LongTensor(train_len)


test_input = torch.Tensor(test_input)
# test_target is LongTensor for embeeding
test_target = torch.LongTensor(test_target)
# test_target = torch.Tensor(test_target)
test_target = test_target.squeeze()
test_len = torch.LongTensor(test_len)
# %%


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
        max_len,
        device,
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_emebdding = nn.Embedding(max_len, embedding_size)
        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_prd_idx = src_pad_idx

        def forward(self, src, trg):
            N, src_seq_length, _ = src.shape
            N, trg_seq_length = trg.shape

            src_positions = (
                torch.arange(0, src_seq_length).unsqueeze(
                    1).expand(src_seq_length, N)
                .to(self.device)
            )
            trg_positions = (
                torch.arange(0, trg_seq_length).unsqueeze(
                    1).expand(trg_seq_length, N)
                .to(self.device)
            )


# %%
# initialize deep networks
src_pad_idx = -1
trg_pad_idx = 0
src_vocab_size = 2
trg_vocab_size = 231
