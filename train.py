
# %%
# import numpy as np
import pandas as pd
import torch
from models.rnn import RNNMODEL
import argparse
import os
import glob
import sys
import numpy as np
# %%
# TODO :: load --> variable store --> torch dataloader
sys.argv = ['']
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', default=0.007, type=float)
parser.add_argument('--train_ratio', default=0.7, type=float)
args = parser.parse_args()
learning_rate = args.learning_rate
train_ratio = args.train_ratio
# learning_rate = 0.001
# train_ratio = 0.7
test_ratio = 1-train_ratio
num_iterations = 100
cuda = True
batch_size = 1000
if torch.cuda.is_available() & cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# def datacombination():
#     Link = []
#     for f in glob.glob("**/*.npy", recursive=True):
#         temp = np.load(f)
#         Link.append(temp)
#     Link = np.concatenate(Link)
#     return Link


# Link4_GPS = datacombination()[0:12000]
# os.chdir("../Label")
# Link4_Label = datacombination()[0:12000]
Link4_GPS = np.load('GPSmax6_1.npy')
Link4_Label = np.load('Labelmax6_1.npy')
n_data = Link4_GPS.shape[0]


# %%
x_min = 127.01691
x_max = 127.07

y_min = 37.48184
y_max = 37.532416


def apply_normalize_x(x): return (x-x_min)/(x_max-x_min) if x != -1 else -1
def apply_normalize_y(y): return (y-y_min)/(y_max-y_min) if y != -1 else -1

#apply_unnormalize_x = lambda p : p * (x_max-x_min) + x_min if p != -1 else -1
#apply_unnormalize_y = lambda p : p * (y_max-y_min) + y_min if p != -1 else -1


Link4_GPS[:, :, 0] = np.vectorize(apply_normalize_x)(Link4_GPS[:, :, 0])
Link4_GPS[:, :, 1] = np.vectorize(apply_normalize_y)(Link4_GPS[:, :, 1])

n_train = int(n_data * train_ratio)
n_test = n_data - n_train

randidx = np.sort(np.random.permutation(n_data))
train_idx = randidx[:n_train]
test_idx = randidx[n_train:(n_train+n_test)]

train_input = Link4_GPS[train_idx]
train_label = Link4_Label[train_idx]
train_len = train_input[:, :, 0] != -1
train_len = train_len.sum(axis=1)

test_input = Link4_GPS[test_idx]
test_label = Link4_Label[test_idx]
test_len = test_input[:, :, 0] != -1
test_len = test_len.sum(axis=1)
# %%
# unq_labels= np.unique(train_label)
# {i:unq_labels[i] for i in range(len(unq_labels))}


model = RNNMODEL(in_feature=2,
                 out_feature=229,
                 hidden=128,
                 num_layer=3)

if device.type == "cuda":
    model = model.cuda(0)

criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-10)

train_input = torch.Tensor(train_input)
train_label = torch.LongTensor(train_label)
train_len = torch.LongTensor(train_len)
total_train = sum(train_len).item()

# train_len , idx = torch.sort(train_len,descending=True)
# train_input = train_input[idx]
# train_label = train_label[idx]
train_label.squeeze_()
# cuda = torch.device('cuda')
# if device.type == "cuda":
#     train_input = train_input.cuda(0)
#     train_label = train_label.cuda(0)
# %%
total_loss = []
training_num = 10000
for i in range(training_num):
    # model.train()
    correct = 0
    randidx = torch.randperm(train_input.shape[0]).to(device)
    # print(randidx)

    train_input = train_input[randidx]
    train_len = train_len[randidx]
    train_label = train_label[randidx]

    data_length = train_input.shape[0]
    num_batch_iteration = int(data_length/batch_size)

    loss_result = []
    accuracy = []
    for j in range(num_batch_iteration+1):
        temp_j = (j+1)*batch_size
        if temp_j > len(train_input)-1:
            temp_j = len(train_input)-1
            # print(temp_j)
        sample_train_input = train_input[(j * batch_size):(temp_j)]

        sample_train_len = train_len[(j * batch_size):(temp_j)]
        sample_train_label = train_label[(j * batch_size):(temp_j)]

        sample_train_len, idx = torch.sort(sample_train_len, descending=True)
        sample_train_input = sample_train_input[idx]
        sample_train_label = sample_train_label[idx]
        sample_train_label = sample_train_label[:, ~torch.all(
            sample_train_label == -1, dim=0)]

        sample_train_input = sample_train_input.to(device)
        sample_train_label = sample_train_label.to(device)
        sample_train_lebel = sample_train_label.to(device)

        optimizer.zero_grad()
        train_pred = model.forward(sample_train_input, sample_train_len)
        loss = criterion(train_pred.view(sample_train_label.size(0) * sample_train_label.size(1), train_pred.size(2)),
                         sample_train_label.view(sample_train_label.size(
                             0) * sample_train_label.size(1))

                         )
# CrossEntropyLoss(input shape is (minibatch,class), target is (minitatch,) here target is label )
# train_pred = model.forward(train_input, train_len)
# loss = criterion(   train_pred.view(train_label.size(0) * train_label.size(1), train_pred.size(2)) ,
#                     train_label.view(train_label.size(0) * train_label.size(1)))

        loss.backward()
        optimizer.step()

        loss_result.append(loss.item())

        prob = torch.softmax(train_pred, dim=2)

        prob_temp = prob.view((prob.size(0)*prob.size(1), prob.size(2)))

        sample_train_label_temp = sample_train_label.view(
            (sample_train_label.size(0) * sample_train_label.size(1)))

        prob_temp = prob_temp[torch.where(sample_train_label_temp != -1)[0], :]

        prob_temp_1 = prob_temp.topk(1, dim=1)[1]

        sample_train_label_temp = sample_train_label_temp[torch.where(
            sample_train_label_temp != -1)[0]]
    # temp_correct = sum(prob_temp_1.squeeze_() == sample_train_label_temp).item()
        # accuracy.append(temp_correct)
        prob_temp = prob_temp[torch.arange(
            prob_temp.size(0)), sample_train_label_temp]
        accuracy_2_result = prob_temp.mean()

        # del sample_train_input
        # del sample_train_label
        # del sample_train_label_temp, sample_train_len

    # accuracy = sum(accuracy)/total_train*100
    accuracy = 100


# if i%10 == 0:
    print("Iter :  {} .. Training Loss : {:.5f} ..Training Accuracy : {:.5f} .. Acc2 : {:.5f}%".format(
        i, np.mean(loss_result), accuracy, accuracy_2_result.item() * 100))
# TODO :: tensorboard

# %%
# torch.cuda.empty_cache()
device = 'cuda'
test_input = torch.Tensor(test_input).to(device)
test_label = torch.LongTensor(test_label).to(device)
test_len = torch.LongTensor(test_len).to(device)
# test_label.squeeze_()
# if device.type == "cuda":
#         model = model.cpu()
# if device.type == "cuda":
#     test_input = test_input.cpu()
#     test_label = test_label.cpu()
# %%
randidx = torch.randperm(test_input.shape[0]).to(device)
sample_test_len = test_len[:256]
sample_test_input = test_input[:256, :, :]
sample_test_label = test_label[:256, :]

sample_test_len, idx = torch.sort(sample_test_len, descending=True)
sample_test_input = sample_test_input[idx]
sample_test_label = sample_test_label[idx]
sample_test_label.squeeze_()


test_pred = model.forward(sample_test_input, sample_test_len)
# %%
i = 7
print(test_pred.argmax(2)[i, :sample_test_len[i]])
print(sample_test_label[i, :sample_test_len[i]])
print(sample_test_label[i, :sample_test_len[i]] ==
      test_pred.argmax(2)[i, :sample_test_len[i]])
# %%
# test_len , idx = torch.sort(test_len,descending=True)
# test_pred = model.forward(test_input, test_len)

# sample_test_label = test_label[:,~torch.all(test_label == -1 , dim =0)]
# prob_1 = torch.softmax(test_pred,dim=2)
# prob_temp_1 = prob_1.view((prob_1.size(0)*prob_1.size(1) , prob_1.size(2)))
# sample_test_label_temp = sample_test_label.view((sample_test_label.size(0) * sample_test_label.size(1)))
# prob_temp_1 = prob_temp_1.cpu()
# sample_test_label_temp = sample_test_label_temp.cpu()
# prob_temp_2 = prob_temp_1[torch.where(sample_test_label_temp!= -1)[0],:]
# prob_temp_2 = prob_temp_2.topk(1,dim=1)[1]
# sample_test_label_temp_1 = sample_test_label_temp[torch.where(sample_test_label_temp!= -1)[0]]
# temp_correct = sum(prob_temp_2.squeeze_() == sample_test_label_temp_1).item()
# accuracy = temp_correct / sample_test_label_temp_1.size()[0]
# print(accuracy)
# %%
correct = 0
train_input = test_input
train_len = test_len
train_label = test_label
data_length = train_input.shape[0]

# %% torhc save model
PATH = os.getcwd()
torch.save(model, PATH + 'model.pt')  # 전체 모델 저장
# 모델 객체의 state_dict 저장
torch.save(model.state_dict(), PATH + 'model_state_dict.pt')
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict()
}, PATH + 'all.tar')

# %%

GPS_1 = pd.read_csv('GPS_edit.csv')
GPS_1_edit = GPS_1.to_numpy()

GPS = torch.Tensor(GPS_1_edit).to(device)
GPS = GPS.unsqueeze(0)
GPS_len = GPS[:, :, 0] != -1
GPS_len = GPS_len.sum(axis=1)
pred = model.forward(GPS, GPS_len).topk(1)[1].reshape(196, 1)

# %%
