# %%
from src.util import addSOSEOS
import torch
import torch.nn as nn
import torch.optim as optim

import time

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter
from src.DataManager import DataManager
from models.Transformer_full import Transformer

from src.util import make_input_mask, make_target_mask, addSOSEOS


class TransformerfullEDTrain(object):
    def __init__(self, params, device, data_manager: DataManager):
        self.params = params
        self.device = device

        self.model = Transformer(params["Embedding_size"],
                                 params["Encoder_in_feature"],
                                 params["Decoder_in_feature"],
                                 params["Encoder_pad"],
                                 params["Heads"],
                                 params["Num_layers"],
                                 params["Num_layers"],
                                 params["Forward_expansion"],
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

        for input, _, label_long, _ in self.data_manager.train_loader:

            label_long_1 = torch.LongTensor(label_long).squeeze(2)

            label_1 = label_long_1[:, 1:].clone().reshape(-1).to(self.device)

            # input = input[0:1]
            # label = label[0:1]

            input = input.permute(1, 0, 2)
            label = label_long_1.permute(1, 0)

            input = input.to(self.device)
            label = label.to(self.device)

            # src = input
            # trg = label[:-1,:]

            # self = self.model
            output, _, _, _ = self.model(input, label[:-1, :])
#            plt.matshow(dec_attn_weight[0].to("cpu").detach().numpy())

            output = output.permute(1, 0, 2)
            label = label.permute(1, 0)

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

            for input, label, label_long, _ in self.data_manager.test_loader:
                label_long_1 = torch.LongTensor(label_long).squeeze(2)

                label_1 = label_long_1[:, 1:].clone(
                ).reshape(-1).to(self.device)

                # input = input[0:1]
                # label = label[0:1]

                input = input.permute(1, 0, 2)
                label = label_long_1.permute(1, 0)

                input = input.to(self.device)
                label = label.to(self.device)

                output, enc_attn, dec_attn, _ = self.model(
                    input, label[:-1, :])
                # plt.matshow(dec_attn[0][2][1:].to("cpu").detach().numpy())
                output = output.permute(1, 0, 2)
                label = label.permute(1, 0)

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

    def get_accuracy_2(self, output, label):

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
# w = 10
# h = 10
# fig = plt.figure(figsize=(40, 8))
# columns = 1
# rows = 1

# fig = plt.figure()
# for i in range(len(dec_attn)):
#     img = dec_attn[i][30][1:, :].to("cpu").detach().numpy()
#     ax = fig.add_subplot(6, 1, i+1)
#     # ax.set_ylim([0,3])
#     plt.imshow(img)
# plt.show()
# '

# %%
# %%
