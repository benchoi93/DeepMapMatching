
# %%
import json
from src.util import normalization, make_input_mask, make_target_mask, datacombination, roundup
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import csv
import sys
import argparse

from models.models_attention import * 
# %%
sys.argv = ['']
parser = argparse.ArgumentParser()
parser.add_argument(
    '--gps-data', default="data/GPSmax3_test_mix_1.npy", type=str)
parser.add_argument(
    '--label_data', default="data/Label_smax3_test_mix_1.npy", type=str)
parser.add_argument(
    '--label_data_long', default="data/Labelmax3_test_mix_1.npy", type=str)
parser.add_argument('--train-ratio', default=0.7, type=float)
parser.add_argument('--batch-size', default=300, type=int)
parser.add_argument('--iteration', default=1000, type=int)
parser.add_argument('--learning-rate', default=0.007, type=float)
parser.add_argument("--clip", default=1.0, type=float)
parser.add_argument('--cuda', default=True, action="store_true")
args = parser.parse_args()

device = torch.device('cuda' if (
    torch.cuda.is_available() & args.cuda) else 'cpu')

boundaries = [127.015, 127.095, 37.47, 37.55]

params = {"Encoder_in_feature": 2,
          "Encoder_embbeding_size": 256,
          "Encoder_hidden_size": 512,
          "Encoder_num_layers": 1,
          "Encoder_dropout": 0,
          "Decoder_embbeding_size": 231,
          "Decoder_hidden_size": 512,
          "Decoder_output_size": 231,
          "Decoder_num_layers": 1,
          "Decoder_dropout": 0,
          "Model_input_pad_index": -1,
          "Model_target_pad_index": 0,
          "clip": args.clip}

encoder = EncoderRNN(params["Encoder_in_feature"],
                            params["Encoder_embbeding_size"],
                            params["Encoder_hidden_size"],
                            params["Encoder_num_layers"],
                            params["Encoder_dropout"]).to(device)

decoder = DecoderRNN(params["Decoder_embbeding_size"],
                            params["Decoder_hidden_size"],
                            params["Decoder_output_size"],
                            params["Decoder_num_layers"],
                            params["Decoder_dropout"]).to(device)

model = Seq2Seq(encoder,decoder,
                        params["Model_input_pad_index"],
                        device).to(device)

path = 'pytorch_model/attention/attn_long.pth'
model.load_state_dict(torch.load(path))

#%%
file_path = 'validation/validation_short/'
padding = -1 
data_input, data_len, data_filelist = datacombination(file_path, padding)


max_len = 30
device = "cuda"
SOT = 229
#%%

test_input = data_input[0:1]
test_len = data_len[0:1]
data_filelist = data_filelist[0:1]
test_input = normalization(
    test_input, boundaries[0], boundaries[1], boundaries[2], boundaries[3])
test_input = torch.Tensor(test_input).to(device)
test_len = torch.LongTensor(test_len)
test_target = torch.empty(len(test_len), max_len)
test_target.fill_(SOT)
test_target = test_target.long()

test_input = test_input.permute(1, 0, 2).to(device)
test_target = test_target.permute(1, 0).to(device)
output, attentions = model(test_input, test_len, test_target)
result = output[1:].argmax(2).permute(1, 0)

#%%
def attention_test(num, data_input, temp_len, boundaries,
                   window, iteration, max_len, SOT, device):
    final_result = []
    final_attention = []
    for i in range(iteration):
        temp_total_input = data_input.copy()
        if window*(i+1) >= temp_len:

            test_input = temp_total_input[num:num+1][:,
                                                     int(temp_len)-window:int(temp_len), :]
        else:

            test_input = temp_total_input[num:num +
                                          1][:, window*i:window*(i+1), :]
            # print("yes")
        test_len = np.array([window])
        test_input = normalization(
            test_input, boundaries[0], boundaries[1], boundaries[2], boundaries[3])
        test_input = torch.Tensor(test_input).to(device)
        test_len = torch.LongTensor(test_len)
        test_target = torch.empty(len(test_len), max_len)
        test_target.fill_(SOT)
        test_target = test_target.long()

        test_input = test_input.permute(1, 0, 2).to(device)
        test_target = test_target.permute(1, 0).to(device)

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

        # %%# visualization
        number = 0
        #atfig = plt.figure(figsize=(1, 1))
        final_result.append(result)
        # print(final_result)
        final_attention.append(attentions)
        #plt.matshow(torch.log(attentions[number]).detach().numpy(), aspect="auto")

    return final_result, final_attention


# %%
with open("C:/Users/iziz56/Downloads/Telegram Desktop/newtraj/new/filtered_Lin.csv", 'w', newline='') as csvfile:
    fieldnames = ['filename', 'sequence']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    total = []
    for num in range(len(data_filelist)):
        print(num)
        temp_len = data_len[num]
        if window >= temp_len:
            window = temp_len
        filelist = data_filelist[num]
        iteration = math.ceil(temp_len/window)
        result = attention_test(num, data_input, temp_len, boundaries,
                                window, iteration, max_len, SOT, device)[0]
        c = np.array([])
        for i in range(len(result)):
            tempa = result[i].to('cpu').clone().numpy()
            temp_index = np.where(tempa == 230)[1]

            if temp_index.tolist() == []: 
                pass
            else: 
                index = temp_index[0]
                tempa[index:] = 230
            c = np.append(c, tempa)
            c = c[c != 230]
            indexes = np.unique(c, return_index=True)[1]
            k = [int(c[index]) for index in sorted(indexes)]
        writer.writerow({'filename': filelist, 'sequence': k})
        total.append(k)


csvfile.close()

 # %%
