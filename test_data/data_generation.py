# %%
import os

import numpy as np
import pandas as pd

# %%
df = pd.read_csv("180002578_4.csv")


df = df.sort_values(by=['time'])

test_df = df.values[:, 1:3]


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
        temp_data = error_generation(test_df[i], 1, 0.00018)

        l.append(temp_data)
    total_GPS.append(l)
    total_Label.append(temp_label)


total_GPS = np.array(total_GPS)
total_Label = np.array(total_Label)

np.save("total_GPS.npy", total_GPS)
np.save("total_Label.npy", total_Label)


# %%
