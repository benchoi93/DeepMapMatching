# %%
import os

import numpy as np
import pandas as pd

# %%
df = pd.read_csv("180002578_4.csv")


df = df.sort_values(by=['time'])

test_df = df.values[:, 1:3]

def lnglatdistance(lat1, lon1, lat2,lon2):
    R = 6371.0
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c * 1000
    return distance

def error_generation(list, number_of_error, var):
    l = []
    count = 0
    while count < number_of_error:
        noise_1 = float(np.random.normal(0, var, 1))/np.sqrt(2)
        #noise_2 = float(np.random.normal(0,var,1))/np.sqrt(2)
        count += 1
        l.append([list[0]+noise_1, list[1]+noise_1])
    return l[0]


total_GPS = []
total_Label = []
for j in range(1000):
    l = []
    temp_label = [229, 16, 60, 68, 230, 0]
    for i in range(len(test_df)):
        temp_data = error_generation(test_df[i], 1,0.00018)
        l.append(temp_data)
    total_GPS.append(l)
    total_Label.append(temp_label)


total_GPS = np.array(total_GPS)
total_Label = np.array(total_Label)

np.save("total_GPS.npy", total_GPS)
np.save("total_Label.npy", total_Label)


# %%
def test():
    l = []
    list = [127,56]
    for i in range(100): 
        a = error_generation(list,1,0.00001)
        b = lnglatdistance(list[0],list[1],a[0],a[1])
        l.append(b)
    return(l)
from matplotlib import pyplot as plt 

plt.hist(test())

# %%
PATH = 'C:/Users/iziz56/Desktop/test_1/DeepMapMatching/data_csv/'
for i in range(100):
    temp_data = raw_input[i]
    temp_data = temp_data[temp_data > 0].reshape(-1, 2)
    file_name = PATH + str(i)+'.csv'
    fmt = '%1.11f', '%1.11f'
    np.savetxt(file_name, temp_data, delimiter=',',
               header='X,Y', fmt=fmt, comments="")


# %%


"""
filtering real data 

"""

import math 
def lnglatdistance(lat1, lon1, lat2,lon2):
    R = 6371.0
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c * 1000
    return distance
os.chdir("D:/Seongjin_Trajectory_filtering/")
outdir_csv = "filtered_csv_1"
files = os.listdir(outdir_csv)
#files = np.sort(files)
remove_list = []
for i in range(len(files)):
    print(i)
    temp_data = pd.read_csv(outdir_csv+"/"+files[i])
    temp_data = temp_data.sort_values(by=['time'])
    for j in range(len(temp_data)):
        if j+2>len(temp_data):
            break
        else:
            x1 = temp_data[j:j+1]['x_coord']
            y1 = temp_data[j:j+1]['y_coord']
            x2 = temp_data[j+1:j+2]['x_coord']
            y2 = temp_data[j+1:j+2]['y_coord']
            dist = lnglatdistance(x1,y1,x2,y2)
            if dist>=200:
                remove_list.append(files[i])
                break 
os.chdir("D:/Seongjin_Trajectory_filtering/filtered_csv_1")

for i in range(len(remove_list)):
    print (i)
    os.remove(remove_list[i])
files = os.listdir()
files.sort(key=lambda f: os.stat(f).st_size,reverse=True)
print(files)
