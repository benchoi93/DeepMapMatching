import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.models import * 
from src.util import * 
import argparse

### temporary code for debugging
import sys
sys.argv = ['']
### temporary code for debugging


parser = argparse.ArgumentParser()
parser.add_argument('--gps-data' , default = "data/GPS_0.npy", type = str)
parser.add_argument('--label-data' , default = "data/Label_0.npy", type = str)
parser.add_argument('--train-ratio', default=0.7, type=float)
parser.add_argument('--learning-rate', default=0.007, type=float)

args = parser.parse_args()

Encoder_in_feature = 2 
Encoder_hidden_size_1=256
Encoder_hidden_size_2 =512 
Encoder_num_layer=1 

Decoder_emb_size = 256 
Decoder_hidden_size=512 
Decoder_output = 231 
Decoder_num_layer=1


### initialize dataset
raw_input = np.load(args.gps_data) 
raw_target = np.load(args.label_data) 

# normalization
x_min = np.min(raw_input[:,:,0][np.where(raw_input[:,:,0] != -1)])
x_max = np.max(raw_input[:,:,0][np.where(raw_input[:,:,0] != -1)])

y_min = np.min(raw_input[:,:,1][np.where(raw_input[:,:,1] != -1)])
y_max = np.max(raw_input[:,:,1][np.where(raw_input[:,:,1] != -1)])

apply_normalize_x = lambda x : (x-x_min)/(x_max-x_min) if x != -1 else -1
apply_normalize_y = lambda y : (y-y_min)/(y_max-y_min) if y != -1 else -1

raw_input[:,:,0] = np.vectorize(apply_normalize_x)(raw_input[:,:,0])
raw_input[:,:,1] = np.vectorize(apply_normalize_y)(raw_input[:,:,1])


### device setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#final training data    
raw_target = addSOSEOS(raw_target,SOS = 229,EOS = 230)
raw_target[raw_target==-1]=0

train_ratio = args.train_ratio
test_ratio = 1-train_ratio
n_data = raw_target.shape[0]
n_train = int(n_data*train_ratio)
n_test = n_data-n_train

randidx=  np.sort(np.random.permutation(n_data))
train_idx = randidx[:n_train]
test_idx = randidx[n_train:(n_train+n_test)]

train_input = raw_input[train_idx]
train_target = raw_target[train_idx]

test_input = raw_input[test_idx]
test_target = raw_target[test_idx]

# change to tensor 
train_input = torch.Tensor(train_input)
train_target = torch.LongTensor(train_target)
train_target = train_target.squeeze()

test_input = torch.Tensor(train_input)
test_target = torch.LongTensor(train_target)
test_target = train_target.squeeze()

# initialize deep networks
encoder = EncoderRNN(Encoder_in_feature,Encoder_hidden_size_1,Encoder_hidden_size_2,Encoder_num_layer)
decoder = DecoderRNN(Decoder_emb_size,Decoder_hidden_size,Decoder_output,Decoder_num_layer)
model = Seq2Seq(encoder,decoder,device)

# set opimizer adn criterioin
optimizer = optim.Adam(model.parameters(),lr = args.learning_rate,eps =1e-10)
criterion = nn.CrossEntropyLoss(ignore_index = 0)

if device.type =='cuda':
    model = model.cuda(0)
    train_input = train_input.cuda(0)
    train_target = train_target.cuda(0)
    test_input = test_input.cuda(0)
    test_target = test_target.cuda(0)


# def train(model, train_input,train_target,optimizer,criterion): 
epoch_loss = 0 
for i in range(10000): 
    output= model(train_input,train_target)

    output = output.view((output.size(0),output.size(2),output.size(1)))
    # train_target = train_target.T

    loss = criterion(output, train_target.T)
    
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()

    acc = torch.sum(torch.argmax(output,1) == train_target.T) 
    acc = float(acc) / (train_target.size(0) * train_target.size(1)) * 100

    epoch_loss +=loss.item()
    print("iteration : {} loss : {:.5f} accuracy : {:.2f}".format(i,loss.item(),acc))
    


    # return epoch_loss






# %%
