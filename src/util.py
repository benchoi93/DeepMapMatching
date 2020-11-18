import numpy as np
import glob


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
