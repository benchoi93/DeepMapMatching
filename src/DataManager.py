from numpy import testing
import torch
import numpy as np

from src.util import normalization, sequence_data
from torch.utils.data import Dataset, DataLoader


class DataManager(object):
    def __init__(self, input_path, label_path, label_path_long, boundaries, train_ratio, batch_size):
        self.raw_input = np.load(input_path)
        self.boundaries = boundaries
        self.raw_input = normalization(
            self.raw_input, boundaries[0], boundaries[1], boundaries[2], boundaries[3])

        self.raw_label = np.load(label_path)
        self.raw_label[self.raw_label < 0] = 0

        self.raw_label_long = np.load(label_path_long)
        self.raw_label_long[self.raw_label_long < 0] = 0

        self.train_ratio = train_ratio
        self.test_ratio = 1-train_ratio

        n_data = self.raw_label.shape[0]
        n_train = int(n_data*train_ratio)
        n_test = n_data-n_train

        randidx = np.random.permutation(n_data)
        train_idx = randidx[:n_train]
        test_idx = randidx[n_train:(n_train+n_test)]

        self.train_input = torch.FloatTensor(self.raw_input[train_idx])
        self.train_label = torch.LongTensor(self.raw_label[train_idx])
        self.train_label_long = torch.LongTensor(
            self.raw_label_long[train_idx])

        self.train_len = self.train_input[:, :, 0] != -1
        self.train_len = self.train_len.sum(axis=1)

        self.test_input = torch.FloatTensor(self.raw_input[test_idx])
        self.test_label = torch.LongTensor(self.raw_label[test_idx])
        self.test_label_long = torch.LongTensor(self.raw_label_long[test_idx])

        self.test_len = self.test_input[:, :, 0] != -1
        self.test_len = self.test_len.sum(axis=1)

        self.train_data = sequence_data(
            self.train_input, self.train_label, self.train_label_long, self.train_len)
        self.test_data = sequence_data(
            self.test_input, self.test_label, self.test_label_long, self.test_len)

        self.train_loader = DataLoader(dataset=self.train_data,
                                       batch_size=batch_size,
                                       shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_data,
                                      batch_size=batch_size,
                                      shuffle=True)
