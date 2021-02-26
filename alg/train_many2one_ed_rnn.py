
import torch
import torch.nn as nn
import torch.optim as optim

import time

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter
from src.DataManager import DataManager
from models.many2one import RnnModel


class RNNEDTrain(object):
    def __init__(self, params, device, data_manager: DataManager):
        self.params = params
        self.device = device

        self.model = RnnModel(params["in_feature"],
                              params["out_feature"],
                              params["hidden"],
                              params["num_layer"]).to(device)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=params["learning_rate"], eps=1e-10)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        self.model_summary_writer = SummaryWriter(
            'log/encoder_decoder_{}'.format(time.time()/60))
        self.data_manager = data_manager

    def train(self):

        acc_result = []
        loss_result = []

        for input, label, length in self.data_manager.train_loader:
            length, idx = torch.sort(length, descending=True)
            input = input[idx]
            label = label[idx]

            length = length.to(self.device)
            input = input.to(self.device)
            label = label.to(self.device)

            # input = input.permute(1, 0, 2)
            output = self.model(input, length)

            trg = label[:, 1:2].reshape(-1)
            loss = self.criterion(output, trg)
            acc = self.get_accuracy(output, label)

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            acc_result.append(acc)
            loss_result.append(loss.item())

        ACC = sum(acc_result)/len(acc_result)
        LOSS = sum(loss_result)/len(loss_result)
        return ACC, LOSS

    def test(self):

        acc_result = []
        loss_result = []

        with torch.no_grad():
            for input, label, length in self.data_manager.test_loader:
                length, idx = torch.sort(length, descending=True)
                input = input[idx]
                label = label[idx]

                length = length.to(self.device)
                input = input.to(self.device)
                label = label.to(self.device)

                #input = input.permute(1, 0, 2)

                output = self.model(input, length)

                trg = label[:, 1:2].reshape(-1)
                loss = self.criterion(output, trg)
                acc = self.get_accuracy(output, label)

                acc_result.append(acc)
                loss_result.append(loss.item())

            ACC = sum(acc_result)/len(acc_result)
            LOSS = sum(loss_result)/len(loss_result)
        return ACC, LOSS

    def get_accuracy(self, output, label):

        #mask = label[1:] != self.params['Model_target_pad_index']
        # padding_count = (
        #     mask == self.params['Model_target_pad_index']).sum().item()
        output_1 = output.argmax(1)

        correct = (output_1 == label[:, 1]).sum()
        total = output_1.size(0)

        acc = correct.item()/total

        return acc
