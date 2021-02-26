# %%
import sys
import argparse
import time

from numpy.lib.function_base import average

import tensorboardX
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.Transformer import *
from src.DataManager import DataManager
from src.util import *
from alg.train_transformer_ed_rnn import TransformerEDTrain
from src.util import make_input_mask, make_target_mask
import csv

sys.argv = ['']
parser = argparse.ArgumentParser()
# parser.add_argument(
#     '--gps-data', default="data/GPSmax3_test_mix_1.npy", type=str)
# parser.add_argument(
#     '--label_data', default="data/Label_smax3_test_mix_1.npy", type=str)
# parser.add_argument(
#     '--label_data_long', default="data/Labelmax3_test_mix_1.npy", type=str)
parser.add_argument('--train-ratio', default=0.7, type=float)
parser.add_argument('--batch-size', default=500, type=int)
parser.add_argument('--iteration', default=200, type=int)
parser.add_argument('--learning-rate', default=0.00015, type=float)
parser.add_argument('--cuda', default=True, action="store_true")
args = parser.parse_args()

device = torch.device('cuda' if (
    torch.cuda.is_available() & args.cuda) else 'cpu')

boundaries = [127.015, 127.095, 37.47, 37.55]

params = {"Encoder_in_feature": 2,
          "Decoder_in_feature": 231,
          "Encoder_pad": -1,
          "Decoder_pad": 0,
          "Embedding_size": 256,
          "Num_layers": 6,
          "Forward_expansion": 4,
          "Heads": 8,
          "Dropout": 0,
          "Device": device,
          "Max_length": 100,
          "learning_rate": args.learning_rate}


model = Transformer(params["Encoder_in_feature"],
                    params["Decoder_in_feature"],
                    params["Encoder_pad"],
                    params["Decoder_pad"],
                    params["Embedding_size"],
                    params["Num_layers"],
                    params["Forward_expansion"],
                    params["Heads"],
                    params["Dropout"],
                    params["Device"],
                    params["Max_length"])

path = 'pytorch_model/transformer/test_model_1.pth'
model.load_state_dict(torch.load(path))
# %%


def datacombination(path, PAD):
    os.chdir("validation")
    filelist = os.listdir(path)
    #filelist = list(pd.read_csv(filename)['filename'])
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
    os.chdir("..")

    return (total_df, df_len, filelist)


file_path = "validation_short/"
#filename = "filtered_Lin_100.csv"
data_input, data_len, data_filelist = datacombination(file_path, -1)
# %%
with open("result.csv", 'w', newline='') as csvfile:
    fieldnames = ['filename', 'sequence']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for num in range(len(data_filelist)):
        test_input = data_input.copy()[num:num+1][:, :, :]
        # test_input = test_input[num][90:180]
        # print(test_input)
        # test_len = np.array([75])
        filelist = data_filelist[num]

    # path = 'pytorch_model/transformer/best_model3_full.pth'

        #model = torch.load(PATH)
        # model.load_state_dict(torch.load(path))

        max_len = 7
        device = "cuda"

        test_input = normalization(
            test_input, boundaries[0], boundaries[1], boundaries[2], boundaries[3])
        test_input = torch.Tensor(test_input).to(device)
        # test_input = test_input.unsqueeze(0)

        # test_target_1 = torch.empty(2, 10)
        # test_target_1.fill_(0)
        # test_target_1 = test_target_1.type(torch.LongTensor).to('cuda')
        trg_indexes = []
        #trg_indexes = [[229]]
        for _ in range(1):
            trg_indexes.append(229)

        model = model.to(device)
        input_mask = test_input[:, :, 0]
        input_mask = make_input_mask(input_mask, -1, device)

        for i in range(20):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
            label_mask = make_target_mask(trg_tensor, device)

            with torch.no_grad():
                output, enc_attn, dec_attn_1, dec_attn_2 = model(
                    test_input, trg_tensor, input_mask, label_mask)

                pred_token = output.argmax(2)[:, -1]
                trg_indexes.append(pred_token.item())

            if pred_token.item() == 230:
                break
        writer.writerow({'filename': filelist, 'sequence': trg_indexes})
# %%
