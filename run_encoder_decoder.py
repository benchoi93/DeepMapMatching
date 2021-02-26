# %%
import sys
import argparse
import time

import tensorboardX
import torch

from src.DataManager import DataManager
from src.util import *
from alg.train_encdec_ed_rnn import EncDecEDTrain


sys.argv = ['']
parser = argparse.ArgumentParser()
parser.add_argument(
    '--gps-data', default="data/GPSmax1_test_mix_1.npy", type=str)
parser.add_argument(
    '--label_data', default="data/Label_smax1_test_1.npy", type=str)
parser.add_argument('--train-ratio', default=0.7, type=float)
parser.add_argument('--batch-size', default=5000, type=int)
parser.add_argument('--iteration', default=1000, type=int)
parser.add_argument('--learning-rate', default=0.007, type=float)
parser.add_argument('--cuda', default=True, action="store_true")
args = parser.parse_args()

device = torch.device('cuda' if (
    torch.cuda.is_available() & args.cuda) else 'cpu')

boundaries = [127.015, 127.095, 37.47, 37.55]
dm = DataManager(input_path=args.gps_data,
                 label_path=args.label_data,
                 boundaries=boundaries,
                 train_ratio=args.train_ratio,
                 batch_size=args.batch_size)

params = {"Encoder_in_feature": 2,
          "Encoder_embbeding_size": 256,
          "Encoder_hidden_size": 512,
          "Encoder_num_layers": 1,
          "Encoder_dropout": 0,
          "Decoder_embbeding_size": 256,
          "Decoder_hidden_size": 512,
          "Decoder_output_size": 231,
          "Decoder_num_layers": 1,
          "Decoder_dropout": 0,
          "Model_input_pad_index": -1,
          "Model_target_pad_index": 0,
          "learning_rate": args.learning_rate}

trainer = EncDecEDTrain(params, device, data_manager=dm)
# %%
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
                   'pytorch_model/encdec/best_model1.pth')

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train ACC: {train_acc*100:.4f}')
    print(f'\t Test Loss: {test_loss:.3f} |  Test ACC: {test_acc*100:.4f}')


# %%
