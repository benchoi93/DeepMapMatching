import numpy as np
import glob
import torch


def addSOSEOS(train, SOS, EOS):
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


def datacombination(path):
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


def normalization(raw_input):
    x_min = 127.015
    x_max = 127.095

    y_min = 37.47
    y_max = 37.55

    def apply_normalize_x(x): return (x-x_min)/(x_max-x_min) if x != -1 else -1
    def apply_normalize_y(y): return (y-y_min)/(y_max-y_min) if y != -1 else -1

    raw_input[:, :, 0] = np.vectorize(apply_normalize_x)(raw_input[:, :, 0])
    raw_input[:, :, 1] = np.vectorize(apply_normalize_y)(raw_input[:, :, 1])
    return raw_input


def make_input_mask(input, input_pad, device):
    input_mask = (input != input_pad).unsqueeze(1).unsqueeze(2)
    # (N, 1, 1, input_len)
    return input_mask.to(device)


def make_trg_mask(trg, device):
    N, trg_len = trg.shape
    trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
        N, 1, trg_len, trg_len
    )

    return trg_mask.to(device)
