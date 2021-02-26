
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
from alg.train_transformer_full_ed_rnn import TransformerfullEDTrain
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
parser.add_argument('--batch-size', default=200, type=int)
parser.add_argument('--iteration', default=200, type=int)
parser.add_argument('--learning-rate', default=0.00015, type=float)
parser.add_argument('--cuda', default=True, action="store_true")
args = parser.parse_args()

device = torch.device('cuda' if (
    torch.cuda.is_available() & args.cuda) else 'cpu')

boundaries = [127.015, 127.095, 37.47, 37.55]
dm = DataManager(input_path=args.gps_data,
                 label_path=args.label_data,
                 label_path_long=args.label_data_long,
                 boundaries=boundaries,
                 train_ratio=args.train_ratio,
                 batch_size=args.batch_size)
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
          "Max_length": 200,
          "learning_rate": args.learning_rate}
# %%
trainer = TransformerfullEDTrain(params, device, data_manager=dm)
self = trainer
best_test_loss = float('inf')
for epoch in range(args.iteration):
    start_time = time.time()
    train_acc, train_loss = trainer.train()
    test_acc, test_loss = trainer.test()
    trainer.model_summary_writer.add_scalar('loss/train', train_loss, epoch)
    trainer.model_summary_writer.add_scalar('acc/train', train_acc*100, epoch)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(trainer.model.state_dict(),
                   'pytorch_model/transformer/transformer_long_2.pth')

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train ACC: {train_acc*100:.4f}')
    print(f'\t Test Loss: {test_loss:.3f} |  Test  ACC: {test_acc*100:.4f}')
# %%