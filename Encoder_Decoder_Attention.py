
# %%
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
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
import pickle
from models.models_attention import *

from src.util import *
import argparse

import time
# temporary code for debugging
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
    '--gps-data', default="data/training_data/GPSmax3_new_1.npy", type=str)
parser.add_argument(
    '--label_data', default="data/training_data/Label_smax3_new_1.npy", type=str)

# parser.add_argument('--gps_data', default="test_data/total_GPS.npy", type=str)
# parser.add_argument('--label_data', default="test_data/total_Label.npy", type=str)
parser.add_argument('--train-ratio', default=0.7, type=float)
parser.add_argument('--learning-rate', default=0.007, type=float)
# parser.add_argument('--training_num', default=100, type=int)
# parser.add_argument('--batch_size', default=10, type=int)


args = parser.parse_args()


# initialize dataset

# raw_input = datacombination("data/GPS/*.npy")
# raw_target = datacombination("data/Label/*.npy")
# raw_target = np.load(args.label_data)


# raw_target = shortencode(raw_target)
raw_input = np.load(args.gps_data)
raw_target = np.load(args.label_data)
# raw_input = raw_input
# raw_target = raw_target[0:100]
raw_target[raw_target < 0] = 0
# %%
# collect unique value from target labeling data
# raw_target = shortencode(raw_target)

# add start and end token in the target data
# raw_input = addEOS(raw_input, EOS=230)


# delete padding data
# raw_target = raw_target[:, 0:raw_target.shape[1]-1]

# raw_target = raw_target+2

# raw_target = addSOSEOS(raw_target, EOS=229, SOS=230)
# raw_target[raw_target < 0] = 0

# raw_target = addpadding(raw_target, padding=0)
# raw_target = raw_target[:, 0:raw_target.shape[1]-1]


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

# train_target is longtensor for embedding
train_target = torch.LongTensor(train_target)
train_target = train_target.squeeze()
train_len = torch.LongTensor(train_len)

test_input = torch.Tensor(test_input)
test_target = torch.LongTensor(test_target)
test_target = test_target.squeeze()
test_len = torch.LongTensor(test_len)

# %%

# initialize deep networks
Encoder_in_feature = 2
Encoder_embbeding_size = 256
Encoder_hidden_size = 512
Encoder_num_layers = 1
Encoder_dropout = 0

Decoder_embbeding_size = 231
Decoder_hidden_size = 512
Decoder_output_size = 231
Decoder_num_layers = 1
Decoder_dropout = 0
Decoder_sequence_length = train_input.size()[1]

Model_input_pad_index = -1

encoder = EncoderRNN(Encoder_in_feature, Encoder_embbeding_size,
                     Encoder_hidden_size, Encoder_num_layers, Encoder_dropout)
decoder = DecoderRNN(Decoder_embbeding_size, Decoder_hidden_size,
                     Decoder_output_size, Decoder_num_layers, Decoder_dropout)
model = Seq2Seq(encoder, decoder, Model_input_pad_index, device)

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
clip = 1
data_batch = train_input.size(0)
batch_size = 700
# PATH = "pytorch_model/max3_2.pth"

# loss_1 = 3
for i in range(100):

    randidx = torch.randperm(train_input.shape[0])
    train_input = train_input[randidx]
    train_len = train_len[randidx]
    train_target = train_target[randidx]

    num_batch_iteration = int(data_batch/batch_size)
    loss_result = []
    acc_result = []

    start_time = time.time()

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
        sample_train_target = sample_train_target.permute(1, 0).to(device)

        # encoder_states,hidden,cell = encoder(sample_train_input, sample_train_len)

        output = model(sample_train_input, sample_train_len,
                       sample_train_target)[0]

        output_dim = output.shape[-1]
        output_1 = output[1:].reshape(-1, output_dim)
        trg = sample_train_target[1:].reshape(-1)
        loss = criterion(output_1, trg)

        mask = sample_train_target[1:] != 0

        padding_count = (mask == 0).sum().item()

        output_2 = output[1:].argmax(2).masked_fill(mask == 0, 0)

        acc_1 = (output_2 == sample_train_target[1:]).sum()-padding_count
        total = output_2.size(0)*output_2.size(1)-padding_count
        if loss < loss_1:
            torch.save(model, PATH)
            loss_1 = loss

        # del sample_train_input
        # del sample_train_len
        # del sample_train_target

        # output = output[1:].reshape(
        #     (output[1:].size(1), output[1:].size(0), output[1:].size(2)))

        # output = output.view((output.size(0),output.size(2),output.size(1)))
        # train_target = train_target.T

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        acc = acc_1.item()/total

        acc_result.append(acc)
        loss_result.append(loss.item())
        torch.cuda.empty_cache()

    ACC = sum(acc_result)/len(acc_result)
    Loss = sum(loss_result)/len(loss_result)
    # acc = float(acc) / (train_target.size(0) * (train_target.size(1)-1)) * 100
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    # epoch_loss += loss.item()
    print("iteration : {} loss : {:.5f} accuracy : {:.4f}% Time: {:1f}m, {:2f}s ".format(
        i+1, Loss, ACC*100, epoch_mins, epoch_secs))
    model_summary_writer.add_scalar('loss', Loss, i)
    model_summary_writer.add_scalar('acc', ACC*100, i)
PATH = "pytorch_model/max3_2.pth"
# torch.save(model,/PAT# %%
# PATH = "C:/Users/iziz56/Desktop/DeepMapMatching/pytorch_model/trained_model_2.pth"
# torch.save(model.state_dict(), PATH)
# torch.cuda.empty_cache()

# device = "cuda"
# test_len, idx = torch.sort(test_len, descending=True)
# model = model.to(device)
# test_input = test_input[idx][:,
#                              0:test_len[0], :].permute(1, 0, 2)
# test_target = test_target[idx].permute(1, 0)

# test_input = test_input.to(device)
# test_target = test_target.to(device)


# output = model(test_input, test_len, test_target)

# mask = test_target[1:] != 0

# padding_count = (mask == 0).sum().item()

# output = output.to(device)
# output_2 = output[1:].argmax(2).masked_fill(mask == 0, 0)

# acc_1 = (output_2 == test_target[1:]).sum()-padding_count
# total = output_2.size(0)*output_2.size(1)-padding_count
# print(acc_1.item()/total*100)


# output_dim = output.shape[-1]
# output = output[1:].reshape(-1, output_dim)
# trg = test_target[1:].reshape(-1)

# acc = ((output.argmax(1) == trg).sum()).item()/trg.size(0) * 100

# %%

"""
test for real data
"""

# df = pd.read_csv("test/180111606_1.csv")
# df = pd.read_csv("test/180720868_0.csv")

# df = df.sort_values(by=['time'])
# test_input = df.values[:, 1:3]
# test_input = np.expand_dims(test_input, axis=0)


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
num = 0
test_input = test_input[num:num+1]
#test_input = test_input[num:num+1][0][0:100]
#test_input = np.expand_dims(test_input, axis=0)
# print(test_input)
test_len = test_len[num:num+1]
#test_len = np.array([100])
filelist = filelist[num]


# %%
def normalization(raw_input):
    # x_min = 127.01691
    # x_max = 127.07

    # y_min = 37.48184
    # y_max = 37.532416
    x_min = 127.015
    x_max = 127.095

    y_min = 37.47
    y_max = 37.55

    def apply_normalize_x(x): return (x-x_min)/(x_max-x_min) if x != -1 else -1
    def apply_normalize_y(y): return (y-y_min)/(y_max-y_min) if y != -1 else -1

    raw_input[:, :, 0] = np.vectorize(apply_normalize_x)(raw_input[:, :, 0])
    raw_input[:, :, 1] = np.vectorize(apply_normalize_y)(raw_input[:, :, 1])
    return raw_input


max_len = 10
device = "cuda"
test_input = normalization(test_input)
test_input = torch.Tensor(test_input)
# test_input = test_input.unsqueeze(0)

test_len = torch.LongTensor(test_len)
test_target = torch.empty(len(test_len), max_len)
test_target.fill_(229)
test_target = test_target.long()
# %%
test_len, idx = torch.sort(test_len, descending=True)

test_input = test_input[idx][:, 0:test_len[0], :]
test_target = test_target[idx]
#filelist = [filelist[idx.item()] for idx in idx]
# test_input_1 = [test_input_1[idx.item()] for idx in idx]
test_input = test_input.permute(1, 0, 2).to(device)
test_target = test_target.permute(1, 0).to(device)

# %%
PATH = "pytorch_model/max3_2.pth"


model = torch.load(PATH)

# /if device == 'cuda':
# model = model.cuda(0)
# train_input = train_input.cuda(0)
# train_target = train_target.cuda(0)
# train_len = train_len.cuda(0)
test_input = test_input.cuda(0)
test_target = test_target.cuda(0)
# test_len = test_len.cuda(0)
# model = torch.load('pytorch_model/best_1.pth')
# test_len = torch.LongTensor([21])
output = model(test_input, test_len, test_target)[0]
attentions = model(test_input, test_len, test_target)[1]

result = output[1:].argmax(2).permute(1, 0)
#result = torch.flip(result, [0])
attentions = attentions[1:].permute(1, 0, 2)
#attentions_1 = torch.flip(attentions,[0])
# with open(file_path+"filelist.txt","wb") as fp:
#     pickle.dump(filelist,fp)

#result_1 = result.to('cpu').numpy()
# np.savetxt('result_1.csv', result_1,delimiter = ',')
output_1 = nn.Softmax(dim=2)(output[1:].permute(1, 0, 2))
# %%
#n = 8
# print((output_1[0][n]*1000).long())
# print((output_1[0][n]*1000).long().topk(2))
# %%# visualization

#attentions = attentions[1:].permute(1, 0, 2)
#atfig = plt.figure(figsize=(1,1))
number = 0
plt.matshow(torch.log(attentions[number]).detach().numpy(), aspect="auto")
print(result[number])
print(attentions[number].topk(3))

print(filelist)
# %%


# %%
"""
PATH = 'generated_data/'
raw_input = np.load(args.gps_data)
for i in range(300):
    temp_data = raw_input[i]
    temp_data = temp_data[temp_data > 0].reshape(-1, 2)
    file_name = PATH + str(i)+'.csv'
    fmt = '%1.11f', '%1.11f'
    np.savetxt(file_name, temp_data, delimiter=',',
               header='X,Y', fmt=fmt, comments="")
""""
# %%
