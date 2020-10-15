import numpy as np


def addSOSEOS(train, SOS, EOS):
    # SOS = 229
    # EOS = 230
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