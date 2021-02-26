# %%
from bertviz.bertviz import head_view
import argparse
import sys

import matplotlib.pyplot as plt
from numpy.lib.function_base import average
import tensorboardX
import torch

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.Transformer import *
from src.util import *
from models.Transformer_full import Transformer
from src.DataManager import *
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
parser.add_argument('--batch-size', default=400, type=int)
parser.add_argument('--iteration', default=200, type=int)
parser.add_argument('--learning-rate', default=0.00015, type=float)
parser.add_argument('--cuda', default=True, action="store_true")
args = parser.parse_args()


device = torch.device('cuda' if (
    torch.cuda.is_available()) else 'cpu')

boundaries = [127.015, 127.095, 37.47, 37.55]

params = {"Embedding_size": 256,
          "Encoder_in_feature": 2,
          "Decoder_in_feature": 231,
          "Encoder_pad": -1,
          "Decoder_pad": 0,
          "Num_layers": 6,
          "Forward_expansion": 1024,
          "Heads": 8,
          "Dropout": 0,
          "Device": device,
          "Max_length": 200}


model = Transformer(params["Embedding_size"],
                    params["Encoder_in_feature"],
                    params["Decoder_in_feature"],
                    params["Encoder_pad"],
                    params["Heads"],
                    params["Num_layers"],
                    params["Num_layers"],
                    params["Forward_expansion"],
                    params["Dropout"],
                    params["Device"],
                    params["Max_length"]).to(device)

path = 'pytorch_model/transformer/best_model3_full.pth'
#model = torch.load(PATH)
model.load_state_dict(torch.load(path))

dm = DataManager(input_path=args.gps_data,
                 label_path=args.label_data,
                 label_path_long=args.label_data_long,
                 boundaries=boundaries,
                 train_ratio=args.train_ratio,
                 batch_size=args.batch_size)

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
for num in range(len(data_filelist)):
    test_input = data_input.copy()[num:num+1][:, :, :]
    # test_input = test_input[num][90:180]
    # print(test_input)
    # test_len = np.array([75])
    filelist = data_filelist[num:num+1]

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
    test_input = test_input.permute(1, 0, 2)

    for i in range(100):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_tensor = trg_tensor.permute(1, 0)

        with torch.no_grad():
            output, enc_attn, dec_attn, memory = model(test_input, trg_tensor)

            pred_token = output.argmax(2)[-1, :]
            trg_indexes.append(pred_token.item())

        if pred_token.item() == 230:
            break

    print(trg_indexes)

    out = np.array(trg_indexes)


# %%


fig = plt.figure()
for i in range(4):
    print(i)
    attn_temp = torch.cat([x[0, i, :].unsqueeze(0) for x in dec_attn]).log10()
    ax = fig.add_subplot(4, 1, i+1)
    # ax.set_ylim([0,3])
    plt.imshow(attn_temp.cpu().numpy()/np.log(10))
    plt.clim(-6, 0)
    plt.colorbar()
plt.show()
# %%

fig = plt.figure()

for i in range(len(dec_attn)):
    img = dec_attn[i][0][:, :].to("cpu").log10().detach().numpy()
    ax = fig.add_subplot(6, 1, i+1)
    # ax.set_ylim([0,3])
    plt.imshow(img)
    plt.clim(-6, 0)
    plt.colorbar()
plt.show()
# %%%

# %%


tokens = list(map(str, range(0, 27)))
test_attn = enc_attn[0].unsqueeze(0)
head_view(enc_attn, tokens)

# %%

#attn_output, attn_output_weights.sum(dim=1) / num_heads
# %%
self = dm
for input, label, label_long, _ in self.test_loader:
    label_1 = label[:, 1:].clone().reshape(-1).to(device)

    input = input.permute(1, 0, 2)
    label = label.permute(1, 0)

    input = input.to(device)
    label = label.to(device)

    output, enc_attn, dec_attn, _ = model(input, label[:-1, :])
    # plt.matshow(dec_attn[0][2][1:].to("cpu").detach().numpy())


test_enc = [enc_attn[i][399:400] for i in range(len(enc_attn))]
test_dec = [dec_attn[i][399:400] for i in range(len(dec_attn))]

test_output = output[-1]
test_output.argmax(1)


# %%
tokens = list(map(str, range(0, test_enc[0].shape[-1])))
head_view(test_enc, tokens)
