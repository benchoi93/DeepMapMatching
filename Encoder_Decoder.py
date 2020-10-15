#%%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

raw_input = np.load('GPS_0.npy') 
raw_target = np.load('Label_0.npy') 

train_ratio = 0.7
test_ratio = 1-train_ratio

n_data = raw_target.shape[0]
n_train = int(n_data*train_ratio)
n_test = n_data-n_train
# normalization

x_min = np.min(raw_input[:,:,0][np.where(raw_input[:,:,0] != -1)])
x_max = np.max(raw_input[:,:,0][np.where(raw_input[:,:,0] != -1)])

y_min = np.min(raw_input[:,:,1][np.where(raw_input[:,:,1] != -1)])
y_max = np.max(raw_input[:,:,1][np.where(raw_input[:,:,1] != -1)])

apply_normalize_x = lambda x : (x-x_min)/(x_max-x_min) if x != -1 else -1
apply_normalize_y = lambda y : (y-y_min)/(y_max-y_min) if y != -1 else -1

raw_input[:,:,0] = np.vectorize(apply_normalize_x)(raw_input[:,:,0])
raw_input[:,:,1] = np.vectorize(apply_normalize_y)(raw_input[:,:,1])


#%%
#print(train_input.shape) # 1280 * 30 * 2 = 58880
#print(train_target.shape) # 1280 * 30 * 2 = 58880
SOS = 229
EOS = 230
#add SOS and EOS in the exist data 
def addSOSEOS(train):
    SOS = 229
    EOS = 230
    padding = -1
    temp_input = np.array([])
    for i in range(len(train)):#len(train_input)): 
        count = sum(sum(train[i]==-1))
        temp_length = train[i].shape[0]
        temp_repeat = train[i].shape[1]
        temp = np.insert(train[i],temp_length*temp_repeat-count,np.repeat(EOS,temp_repeat))
        temp = np.insert(temp,0,np.repeat(SOS,temp_repeat))
        temp_input = np.append(temp_input,temp)
    temp_input = temp_input.reshape(len(train),temp_length+2,temp_repeat)
    return(temp_input)
#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#final training data    
#raw_input = addSOSEOS(raw_input)
raw_target = addSOSEOS(raw_target)
raw_target[raw_target==-1]=0
randidx=  np.sort(np.random.permutation(n_data))
train_idx = randidx[:n_train]
test_idx = randidx[n_train:(n_train+n_test)]

train_input = raw_input[train_idx]
train_target = raw_target[train_idx]

test_input = raw_input[test_idx]
test_target = raw_target[test_idx]

#rain_input[train_input==-1]=padding

#train_len = (train_input[:,:,0]!=padding).sum(axis=1)
# change to tensor 
train_input = torch.Tensor(train_input)
train_target = torch.LongTensor(train_target)
train_target = train_target.squeeze()

test_input = torch.Tensor(train_input)
test_target = torch.LongTensor(train_target)
test_target = train_target.squeeze()
#train_len = torch.LongTensor(train_len)
#train_len,idx = torch.sort(train_len,descending=True )
#train_input = train_input[idx]
#train_target = train_target[idx]

#%%

class EncoderRNN(nn.Module):
    def __init__(self, in_feature, hidden_size_1, hidden_size_2, num_layer):
        super(EncoderRNN, self).__init__()
        self.in_feature = in_feature

        self.hidden_size_1 = hidden_size_1
        
        self.hidden_size_2 = hidden_size_2

        self.embed_fc = torch.nn.Linear(in_feature, hidden_size_1)
        self.rnn = torch.nn.LSTM(hidden_size_1, hidden_size_2, num_layer, batch_first=True)

    def forward(self, input):

        x = self.embed_fc(input)
        #packed_x = pack_padded_sequence(x,input_len,batch_first= True)
        _,(hidden,cell) = self.rnn(x)
        #x,_ = pad_packed_sequence(packed_x,batch_first=True)
        return hidden, cell

#%%

class DecoderRNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size,num_layer):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, emb_size,padding_idx = 0)
        self.rnn = nn.LSTM(emb_size,hidden_size,num_layer)
        self.out_1 = nn.Linear(hidden_size, hidden_size)
        self.output_size = output_size
        self.out_2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.activation = torch.nn.ReLU()

    def forward(self, x, hidden,cell):
        x = x.unsqueeze(0)
        x = self.embedding(x)
        output, (hidden,cell) = self.rnn(x, (hidden,cell))
        output = output.squeeze(0)
        output = self.out_1(output)
        output = self.activation(output)
        output = self.out_2(output)
        output = self.activation(output)
        #output = self.softmax(output)
        
        return output, hidden, cell

#%%
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder,device):
        super().__init__()
    
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        

    def forward(self,train,target):
        # train = [batch,len,2]
        # target = [batch,len]
        batch_size = target.shape[0]
        target_len = target.shape[1]
        target_output_size = self.decoder.output_size

        #tensor to store decoder output
        outputs = torch.zeros(target_len,batch_size,target_output_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder 

        hidden, cell = self.encoder(train)

        # first input to the decoder  is the  <SOS> tokens

        input = target[:,0]

        for t in range (1,target_len):
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output,hidden,cell = self.decoder(input,hidden,cell)

            #place outputs in a tensor holding it for each token 

            outputs[t] = output 

            # decide if we are going to use teacher forcing or not 
            # teacher_force = random.random()<teacher_force_ratio

            # get the highest predictd token from output 
            input = output.argmax(1)

        return outputs

# %%
Encoder_in_feature = 2 
Encoder_hidden_size_1=256
Encoder_hidden_size_2 =512 
Encoder_num_layer=1 

Decoder_emb_size = 256 
Decoder_hidden_size=512 
Decoder_output = 231 
Decoder_num_layer=1

encoder = EncoderRNN(Encoder_in_feature,Encoder_hidden_size_1,Encoder_hidden_size_2,Encoder_num_layer)
decoder = DecoderRNN(Decoder_emb_size,Decoder_hidden_size,Decoder_output,Decoder_num_layer)
model = Seq2Seq(encoder,decoder,device)

# set opimizer adn criterioin
optimizer = optim.Adam(model.parameters(),lr = 0.007,eps =1e-10)
criterion = nn.CrossEntropyLoss(ignore_index = 0)

if device.type =='cuda':
    model = model.cuda(0)
    train_input = train_input.cuda(0)
    train_target = train_target.cuda(0)
    #train_len = train_len.cuda(0)

if device.type =='cuda':
    model = model.cuda(0)
    test_input = test_input.cuda(0)
    test_target = test_target.cuda(0)
# %%


def train(model, train_input,train_target,optimizer,criterion): 
    epoch_loss = 0 
    encoder
    for i in range(1): 
        optimizer.zero_grad() 

        #output = [len, batch_size,output_di,]
        output= model(train_input,train_target)

        output_dim = output.shape[-1]

        #output = [(trg len - 1) * batch size, output dim]
        output = output[1:].view(-1,output_dim)
        train_target_T = torch.transpose(train_target,0,1)
        trg = train_target_T[1:].reshape(-1)

        loss = criterion(output,trg)

        loss.backward()

        optimizer.step()

        epoch_loss +=loss.item()

    return epoch_loss






# %%
