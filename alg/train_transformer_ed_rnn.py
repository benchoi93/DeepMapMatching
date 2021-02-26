
import torch
import torch.nn as nn
import torch.optim as optim

import time

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter
from src.DataManager import DataManager
from models.Transformer import Encoder, Decoder, Transformer

from src.util import make_input_mask, make_target_mask


class TransformerEDTrain(object):
    def __init__(self, params, device, data_manager: DataManager):
        self.params = params
        self.device = device

        self.model = Transformer(params["Encoder_in_feature"],
                                 params["Decoder_in_feature"],
                                 params["Encoder_pad"],
                                 params["Decoder_pad"],
                                 params["Embedding_size"],
                                 params["Num_layers"],
                                 params["Forward_expansion"],
                                 params["Heads"],
                                 params["Dropout"],
                                 params["Device"],
                                 params["Max_length"]).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=params["learning_rate"], eps=1e-10)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        self.model_summary_writer = SummaryWriter(
            'log/encoder_decoder_{}'.format(time.time()/60))
        self.data_manager = data_manager

    def train(self):
        self.model.train()

        acc_result = []
        loss_result = []

        for input, label, _, _ in self.data_manager.train_loader:

            label_1 = label[:, 1:].reshape(-1).to(self.device)
            label_2 = label[:, :-1]
            label_mask = make_target_mask(label_2, self.device)
            input_mask = input[:, :, 0]
            input_mask = make_input_mask(
                input_mask, self.params["Encoder_pad"], self.device)

            input = input.to(self.device)
            label_2 = label_2.to(self.device)

            output, _, _, _ = self.model(
                input, label_2, input_mask, label_mask)

            output_dim = output.shape[-1]
            output_1 = output.reshape(-1, output_dim)
            loss = self.criterion(output_1, label_1)

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
        self.model.eval()

        acc_result = []
        loss_result = []

        with torch.no_grad():

            for input, label, _, _ in self.data_manager.test_loader:

                label_1 = label[:, 1:].reshape(-1).to(self.device)
                label_2 = label[:, :-1]
                label_mask = make_target_mask(label_2, self.device)
                input_mask = input[:, :, 0]
                input_mask = make_input_mask(
                    input_mask, self.params["Encoder_pad"], self.device)

                input = input.to(self.device)
                label_2 = label_2.to(self.device)

                output, _, _, _ = self.model(
                    input, label_2, input_mask, label_mask)

                output_dim = output.shape[-1]
                output_1 = output.reshape(-1, output_dim)
                loss = self.criterion(output_1, label_1)

                acc = self.get_accuracy(output, label)

                acc_result.append(acc)
                loss_result.append(loss.item())

            ACC = sum(acc_result)/len(acc_result)
            LOSS = sum(loss_result)/len(loss_result)
        return ACC, LOSS

    def get_accuracy(self, output, label):

        mask = (label[:, 1:] != self.params["Decoder_pad"]).to(self.device)
        padding_count = (mask == self.params["Decoder_pad"]).sum().item()
        output_1 = output.argmax(2).masked_fill(
            mask == self.params["Decoder_pad"], 0)

        correct = (output_1 == label[:, 1:].to(
            self.device)).sum()-padding_count
        total = output_1.size(0)*output_1.size(1)-padding_count

        acc = correct.item()/total

        return acc

# %%
