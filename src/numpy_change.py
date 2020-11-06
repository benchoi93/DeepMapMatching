# %%
import numpy as np
raw_target = np.load("../data/Label_0.npy")

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

np.save("../data/Label_1.npy", temp_input)

# %%


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
