from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import torch
import os
import pandas as pd


def addSOSEOS(train, SOS, EOS):
    # SOS = 229
    # EOS = 230
    padding = -1
    temp_input = np.array([])
    for i in range(len(train)):  # len(train_input)):
        count = sum(sum(train[i] == 0))
        temp_length = train[i].shape[0]
        temp_repeat = train[i].shape[1]
        temp = np.insert(train[i], temp_length*temp_repeat -
                         count, np.repeat(EOS, temp_repeat))
        temp = np.insert(temp, 0, np.repeat(SOS, temp_repeat))
        temp_input = np.append(temp_input, temp)
    temp_input = temp_input.reshape(len(train), temp_length+2, temp_repeat)
    return(temp_input)


def addEOS(train, EOS):
    # SOS = 229
    # EOS = 230
    padding = -1
    temp_input = np.array([])
    for i in range(len(train)):  # len(train_input)):
        count = sum(sum(train[i] == -1))
        temp_length = train[i].shape[0]
        temp_repeat = train[i].shape[1]
        temp = np.insert(train[i], temp_length*temp_repeat -
                         count, np.repeat(EOS, temp_repeat))
        temp_input = np.append(temp_input, temp)
        print(i)
    temp_input = temp_input.reshape(len(train), temp_length+1, temp_repeat)
    return(temp_input)

# "..data/*.npy"


def datacombine(path):
    Link = []
    for f in glob.glob(path, recursive=True):
        temp = np.load(f)
        Link.append(temp)
    Link = np.concatenate(Link)
    return Link


def shortencode(raw_target):
    raw_target = raw_target.reshape(raw_target.shape[0], raw_target.shape[1])

    padding = -1
    temp_input = np.array([])
    length_batch = raw_target.shape[0]
    length_len = raw_target.shape[1]
    for i in range(length_batch):
        temp_data = raw_target[i]
        _, idx = np.unique(temp_data, return_index=True)
        temp = temp_data[np.sort(idx)]
        if sum(temp == -1) == 0:
            temp = np.insert(temp, len(temp), padding)
        temp_input = np.append(temp_input, temp)
    temp_input = temp_input.reshape(length_batch, len(temp), 1)
    temp_input = temp_input.astype(int)

    return temp_input


def addpadding(train, padding):

    temp_input = np.array([])
    for i in range(len(train)):  # len(train_input)):
        temp_length = train[i].shape[0]
        temp_repeat = train[i].shape[1]
        temp = np.insert(train[i], temp_length,
                         np.repeat(padding, temp_repeat))
        temp_input = np.append(temp_input, temp)
    temp_input = temp_input.reshape(len(train), temp_length+1, temp_repeat)
    return(temp_input)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def normalization(raw_input, x_min, x_max, y_min, y_max):
    # x_min = 127.015
    # x_max = 127.095

    # y_min = 37.47d
    # y_max = 37.55

    def apply_normalize_x(x): return (x-x_min)/(x_max-x_min) if x != -1 else -1
    def apply_normalize_y(y): return (y-y_min)/(y_max-y_min) if y != -1 else -1

    raw_input[:, :, 0] = np.vectorize(apply_normalize_x)(raw_input[:, :, 0])
    raw_input[:, :, 1] = np.vectorize(apply_normalize_y)(raw_input[:, :, 1])
    return raw_input


def make_input_mask(input, input_pad, device):
    input_mask = (input != input_pad).unsqueeze(1).unsqueeze(2)
    # (N, 1, 1, input_len)
    return input_mask.to(device)


def make_target_mask(trg, device):
    N, trg_len = trg.shape
    trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
        N, 1, trg_len, trg_len
    )

    return trg_mask.to(device)


def datacombination(path, PAD):
    filelist = os.listdir(path)
    df_len = []
    total_df = []
    for file in filelist:
        df = pd.read_csv(path+file)
        df = df.sort_values(by=['time'])
        df_len.append(len(df))
        test_df = df.values[:, 0:2]
        total_df.append(test_df)

    max_len = max(df_len)+5

    for i, length in enumerate(df_len):
        temp = total_df[i]
        added_row = np.tile([PAD, PAD], ((max_len-length), 1))
        temp = np.append(temp, added_row, axis=0)
        total_df[i] = np.expand_dims(temp, axis=0)
    total_df = np.concatenate(total_df)
    df_len = np.array(df_len)

    return (total_df, df_len, filelist)


def roundup(x, value):
    return x if x % value == 0 else x+value-x % value


class sequence_data(Dataset):
    def __init__(self, gps, link, link_long, len):
        self.gps = gps
        self.link = link
        self.link_long = link_long
        self.len = len

        self.data_size = gps.size(0)

    def __getitem__(self, index):
        return self.gps[index], self.link[index], self.link_long[index], self.len[index]

    def __len__(self):
        return self.data_size


def numpy2csv(input_path):
    # os.mkdir("augmentation_data")

    raw_input = np.load(input_path)
    for i in range(10):
        temp_data = raw_input.copy()[i]
        temp_data_1 = (temp_data[temp_data != -1]).reshape(-1, 2)
        np.savetxt("augmentation_data/aug"+str(i)+".csv",
                   temp_data_1, header="x,y", delimiter=",", comments="")
