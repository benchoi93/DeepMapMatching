# %%
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
import sys
# %%
sys.argv = ['']
# temporary code for debugging

torch.cuda.empty_cache()

model_summary_writer = SummaryWriter(
    'log/encoder_decoder_{}'.format(time.time()/60))

parser = argparse.ArgumentParser()
# parser.add_argument('--gps-data', default="data/GPS/GPS_0.npy", type=str)
parser.add_argument(
    '--gps_data', default="data/training_data/GPSmax7_new_6.npy", type=str)
parser.add_argument(
    '--label_data', default="data/training_data/Label_smax7_new_6.npy", type=str)
# parser.add_argument('--label_data', default="Label_1.npy", type=str)
parser.add_argument('--train-ratio', default=0.5, type=float)
parser.add_argument('--learning-rate', default=0.0001, type=float)
# parser.add_argument('--training_num', default=100, type=int)
parser.add_argument('--batch_size', default=1, type=int)


args = parser.parse_args()


# initialize dataset
# raw_input = datacombination("data/GPS/*.npy")
# raw_target = datacombination("data/Label/*.npy")
raw_input = np.load(args.gps_data)[0:4]
raw_target = np.load(args.label_data)[0:4]
raw_target[raw_target < 0] = 0
# raw_target = raw_target[:, 1

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
# initialize deep networks
src_pad_idx = -1
trg_pad_idx = 0
src_vocab_size = 2
trg_vocab_size = 231

model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(
    device
)
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
data_batch = train_input.size(0)
PATH = "pytorch_model/transformer_2.pth"
loss_1 = 3
for i in range(1000):

    randidx = torch.randperm(train_input.shape[0])
    train_input = train_input[randidx]
    train_len = train_len[randidx]
    train_target = train_target[randidx]

    num_batch_iteration = int(data_batch/args.batch_size)
    loss_result = []
    acc_result = []
    start_time = time.time()
    for j in range(num_batch_iteration):

        temp_j = (j+1)*args.batch_size

        sample_train_input = train_input[(j * args.batch_size):(temp_j)]

        # sample_train_len = train_len[(j * args.batch_size):(temp_j)]
        sample_train_target = train_target[(
            j * args.batch_size):(temp_j)]
        trg = sample_train_target[:, 1:].reshape(-1).to(device)
        sample_train_target[sample_train_target == 230] = 0
        sample_train_target = sample_train_target[:, :-1]
        sample_train_target_mask = make_trg_mask(sample_train_target, device)
        sample_train_input_mask = sample_train_input[:, :, 0]
        sample_train_input_mask = make_input_mask(
            sample_train_input_mask, -1, device)
        # sample_train_len, idx = torch.sort(sample_train_len, descending=True)

        # sample_train_input = sample_train_input[idx][:,
        #                                              0:sample_train_len[0], :]
        # sample_train_target = sample_train_target[idx]

        sample_train_input = sample_train_input.to(device)
        sample_train_target = sample_train_target.to(device)

        output = model(sample_train_input,
                       sample_train_target, sample_train_input_mask, sample_train_target_mask)

        # output = [trg len,batch size, output dim]

        output_dim = output.shape[-1]
        output = output.reshape(-1, output_dim)

        # mask = sample_train_target[1:] != 0
        # padding_count = (mask == 0).sum().item()
        # output_2 = output.argmax(2).masked_fill(mask == 0, 0)

        # output = output.view((output.size(0),output.size(2),output.size(1)))
        # train_target = train_target.T

        loss = criterion(output, trg)
        if loss < loss_1:
            torch.save(model, PATH)
            loss_1 = loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # acc = ((output.argmax(1) == trg).sum()).item()/trg.size(0)
        # acc_result.append(acc)
        loss_result.append(loss.item())

        torch.cuda.empty_cache()

    # ACC = sum(acc_result)/len(acc_result) * 100
    ACC = 0
    Loss = sum(loss_result)/len(loss_result)

    # acc = float(acc) / (train_target.size(0) * (train_target.size(1)-1)) * 100
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    # epoch_loss += loss.item()
    print("iteration : {} loss : {:.5f} accuracy : {:.4f}% Time: {:1f}m, {:2f}s ".format(
        i+1, Loss, ACC*100, epoch_mins, epoch_secs))

    # epoch_loss += loss.item()

    model_summary_writer.add_scalar('loss', Loss, i)
    model_summary_writer.add_scalar('acc', ACC, i)


# %%

# model = torch.load(PATH)

encoder = model.encoder
decoder = model.decoder

test_input = train_input[0:1].to(device)
test_target = train_target[0:1].to(device)
# test_target_1 = torch.empty(2, 10)
# test_target_1.fill_(0)
# test_target_1 = test_target_1.type(torch.LongTensor).to('cuda')

trg_indexes = [229]
trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
test_target_mask = make_trg_mask(trg_tensor, device)
test_input_mask = test_input[:, :, 0]

test_ipnut_mask = make_input_mask(
    test_input_mask, -1, device)
enc_src = encoder(test_input, test_input_mask)
output = decoder(trg_tensor, enc_src, test_input_mask, test_target_mask)

# %%

test_target_mask = make_trg_mask(test_target, device)
test_input_mask = test_input[:, :, 0]
test_input_mask = make_input_mask(
    test_input_mask, -1, device)
# sample_train_len, idx = torch.sort(sample_train_len, descending=True)

# sample_train_input = sample_train_input[idx][:,
#                                              0:sample_train_len[0], :]
# sample_train_target = sample_train_target[idx]


output_1 = model(test_input,
                 test_target, test_input_mask, test_target_mask)

output_1.argmax(2)

# output = [trg len,batch size, output dim]
# %%
output_dim = output.shape[-1]
output = output.reshape(-1, output_dim)
trg = test_target.reshape(-1)
test_output = output.argmax(1).reshape(300, 4)
test_1 = test_output[:, 0:3]
test_2 = test_target[:, 0:3]

acc = (test_1 == test_2).sum().item()/(test_1.size(0)*test_2.size(1))*100


# train_input_mask = None
# train_target_mask = make_trg_mask(train_target, device)
# for i in range(10000):
#     train_len, idx = torch.sort(train_len, descending=True)

#     train_input = train_input[idx][:,0:train_len[0], :]
#     train_target = train_target[idx]
#     train_input_mask = train_input[:, :, 0]
#     train_input_mask = make_input_mask(train_input_mask,-1,device)

#     output = model(train_input, train_target,
#                    train_input_mask, train_target_mask)
#     output_dim = output.shape[-1]
#     output_1 = output.reshape(-1, output_dim)
#     train_target_1 = train_target.reshape(-1)
#     loss = criterion(output_1, train_target_1)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
#     print(loss)

# test_len, idx = torch.sort(test_len, descending=True)

# test_input = test_input[idx][:,0:test_len[0], :]
# test_target = test_target[idx]
# test_input_mask = test_input[:, :, 0]
# test_input_mask = make_input_mask(test_input_mask,-1,device)
# test_target_mask = make_trg_mask(test_target, device)

# output = model(test_input,test_target,test_input_mask,test_target_mask)
# output_dim = output.shape[-1]
# output_1 = output.reshape(-1, output_dim)
# test_target.reshape(-1)
# %%


def datacombination(path, PAD):
    filelist = os.listdir(path)
    df_len = []
    total_df = []
    for file in filelist:
        df = pd.read_csv(path+file)
        df = df.sort_values(by=['time'])
        df_len.append(len(df))
        test_df = df.values[:, 1:3]
        total_df.append(test_df)

    max_len = max(df_len)

    for i, length in enumerate(df_len):
        temp = total_df[i]
        added_row = np.tile([PAD, PAD], ((max_len-length), 1))
        temp = np.append(temp, added_row, axis=0)
        total_df[i] = np.expand_dims(temp, axis=0)
    total_df = np.concatenate(total_df)
    df_len = np.array(df_len)

    return (total_df, df_len, filelist)


file_path = '../validation_data_2/'
test_input, test_len, filelist = datacombination(file_path, -1)
num = 8
test_input = test_input[num:num+1]
# test_input = test_input[num:num+1][0][100:175]
# test_input = np.expand_dims(test_input, axis=0)
# print(test_input)
test_len = test_len[num:num+1]
# test_len = np.array([75])
filelist = filelist[num]
# %%

# PATH = "pytorch_model/transformer_2.pth"
# model = torch.load(PATH)

max_len = 20
device = "cuda"
test_input = normalization(test_input)
test_input = torch.Tensor(test_input)
# test_input = test_input.unsqueeze(0)

test_len = torch.LongTensor(test_len)
test_target = torch.empty(len(test_len), max_len)
test_target.fill_(0)
test_target = test_target.long()
# %%
test_target_mask = make_trg_mask(test_target, device)
test_input_mask = test_input[:, :, 0]
test_input_mask = make_input_mask(
    test_input_mask, -1, device)
test_input = test_input.to(device)
test_target = test_target.to(device)

output = model(test_input,
               test_target, test_input_mask, test_target_mask)

# %%

# %%
n = 8
print((output_1[0][n]*1000).long())
print((output_1[0][n]*1000).long().topk(2))
