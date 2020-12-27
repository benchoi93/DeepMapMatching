

# %%
# from Encoder_Decoder_Attention import Decoder_num_layers, Encoder_num_layers
# from Encoder_Decoder_Attention import Decoder_output_size
import sys
from numpy.lib.function_base import average
from tensorboardX import SummaryWriter
import tensorboardX
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time

from models.models import *

from src.util import *
import argparse
import time
import glob

torch.cuda.empty_cache()

# temporary code for debugging
sys.argv = ['']
# temporary code for debugging

model_summary_writer = SummaryWriter(
    'log/encoder_decoder_{}'.format(time.time()))

parser = argparse.ArgumentParser()
# parser.add_argument('--gps-data', default="data/GPS/GPS_0.npy", type=str)
# parser.add_argument('--gps-data', default="GPS.npy", type=str)
parser.add_argument('--gps_data', default="GPSmax7_new_1.npy", type=str)
parser.add_argument('--label_data', default="Label_smax7_new_1.npy", type=str)
parser.add_argument('--train-ratio', default=0.7, type=float)
parser.add_argument('--learning-rate', default=0.007, type=float)
# parser.add_argument('--training_num', default=100, type=int)
parser.add_argument('--batch_size', default=1000, type=int)


args = parser.parse_args()

Encoder_in_feature = 2
Encoder_emb_size = 256
Encoder_hid_size = 512
Encoder_num_layers = 1

Decoder_emb_size = 256
Decoder_hidden_size = 512
Decoder_output_size = 231
Decoder_num_layers = 1
# initialize dataset

# raw_input = datacombination("data/GPS/*.npy")
raw_input = np.load(args.gps_data)
raw_target = np.load(args.label_data)

raw_input = raw_input[0:2000]
raw_target = raw_target[0:2000]
raw_target[raw_target < 0] = 0

# %%

# add start and end token in the target data

# delete padding data
# raw_target = raw_target[:, 0:raw_target.shape[1]-1]

# device setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# normalization
x_min = 127.01691
x_max = 127.07

y_min = 37.48184
y_max = 37.532416


def apply_normalize_x(x): return (x-x_min)/(x_max-x_min) if x != -1 else -1
def apply_normalize_y(y): return (y-y_min)/(y_max-y_min) if y != -1 else -1


raw_input[:, :, 0] = np.vectorize(apply_normalize_x)(raw_input[:, :, 0])
raw_input[:, :, 1] = np.vectorize(apply_normalize_y)(raw_input[:, :, 1])


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

# train_target is longtensor for embedding
train_target = torch.LongTensor(train_target)
train_target = train_target.squeeze()

# train_target is Tensor for embed_fc

# train_target = torch.Tensor(train_target)
# train_target = train_target.squeeze()
train_len = torch.LongTensor(train_len)


test_input = torch.Tensor(test_input)
# test_target is LongTensor for embeeding
test_target = torch.LongTensor(test_target)
# test_target = torch.Tensor(test_target)
test_target = test_target.squeeze()
test_len = torch.LongTensor(test_len)


# initialize deep networks
encoder = EncoderRNN(Encoder_in_feature, Encoder_emb_size,
                     Encoder_hid_size, Encoder_num_layers)
decoder = DecoderRNN(Decoder_emb_size,
                     Decoder_hidden_size, Decoder_output_size, Decoder_num_layers)
model = Seq2Seq(encoder, decoder, device)

# set opimizer adn criterioin
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-10)
criterion = nn.CrossEntropyLoss(ignore_index=0)


# train_len, idx = torch.sort(train_len, descending=True)
# train_input = train_input[idx]
# train_target = train_target[idx]
# train_target = train_target[train_target[:] != 0].reshape(
# train_target.shape[0], train_target.shape[1]-1)

if device.type == 'cuda':
    model = model.cuda(0)
    # train_input = train_input.cuda(0)
    # train_target = train_target.cuda(0)
    # train_len = train_len.cuda(0)
    # test_input = test_input.cuda(0)
    # test_target = test_target.cuda(0)
    # test_len = test_len.cuda(0)


# %%
# epoch_loss = 0

data_batch = train_input.size(0)
for i in range(10000):

    randidx = torch.randperm(train_input.shape[0])
    train_input = train_input[randidx]
    train_len = train_len[randidx]
    train_target = train_target[randidx]

    num_batch_iteration = int(data_batch/args.batch_size)
    loss_result = []
    acc_result = []

    for j in range(num_batch_iteration):

        temp_j = (j+1)*batch_size

        sample_train_input = train_input[(j * batch_size):(temp_j)]
        sample_train_target = train_target[(j * batch_size):(temp_j)]

        sample_train_len = train_len[(j * batch_size):(temp_j)]

        sample_train_len, idx = torch.sort(sample_train_len, descending=True)

        sample_train_input = sample_train_input[idx][:,
                                                     0:sample_train_len[0], :]
        sample_train_target = sample_train_target[idx]

        sample_train_input = sample_train_input.permute(1, 0, 2).to(device)
        sample_train_target = sample_train_target.permute(1, 0).to(device

        output=model(sample_train_input,
                       sample_train_target, sample_train_len)

        # output = [trg len,batch size, output dim]

        output_dim=output.shape[-1]
        output=output[1:].reshape(-1, output_dim)
        trg=sample_train_target[1:].reshape(-1)

        # output = output.view((output.size(0),output.size(2),output.size(1)))
        # train_target = train_target.T

        loss=criterion(output, trg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc=((output.argmax(1) == trg).sum()).item()/trg.size(0)
        acc_result.append(acc)
        loss_result.append(loss.item())

    ACC=sum(acc_result)/len(acc_result) * 100
    Loss=sum(loss_result)/len(loss_result)

    # acc = float(acc) / (train_target.size(0) * (train_target.size(1)-1)) * 100

    # epoch_loss += loss.item()
    print("iteration : {} A-loss : {:.5f} accuracy : {:.4f}".format(
        i+1, Loss, ACC))
    model_summary_writer.add_scalar('loss', Loss, i)
    model_summary_writer.add_scalar('acc', ACC, i)

# %%
# PATH = "C:/Users/iziz56/Desktop/DeepMapMatching/pytorch_model/trained_model_2.pth"
# torch.save(model.state_dict(), PATH)
# validation
test_len, idx=torch.sort(test_len, descending=True)

test_input=test_input[idx][:,
                             0:test_len[0], :].permute(1, 0, 2)
test_target=test_target[idx].permute(1, 0)

test_input=test_input.to(device)
test_target=test_target.to(device)


output=model(test_input, test_target, test_len)
output_dim=output.shape[-1]
output=output[1:].reshape(-1, output_dim)
trg=test_target[1:].reshape(-1)

acc=((output.argmax(1) == trg).sum()).item()/trg.size(0) * 100


# acc = torch.sum(torch.argmax(output, 1) == test_target_1)
# acc = float(acc) / (test_target.size(0) * (test_target.size(1)-1)) * 100
# print(acc)

# %%
