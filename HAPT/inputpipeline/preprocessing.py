import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import csv
import matplotlib.pyplot as plt
import PIL.Image as Image
import os

gyro_name = "gyro_exp59_user29.txt"
acc_name = "acc_exp05_user03.txt"

def preprocessing(root_path):
    usr_dict = {}
    for file_name in os.listdir(root_path):
        if file_name != "labels.txt" and file_name != '.DS_Store':
            usr_index = int(file_name[-6:-4])##!!!! if -1 -> does not include last one!!!!
            exp_index = int(file_name[-13:-11])
            if usr_index in usr_dict.keys():
                if not exp_index in usr_dict[usr_index].keys():
                    usr_dict[usr_index][exp_index] = [file_name[3:]]
            else:
                usr_dict[usr_index] = {}
                usr_dict[usr_index][exp_index] =[file_name[3:]]
            # with open(root_path + file_name,'r') as file:
            #     file_array = file.readlines()[251:-250]

    # for usr in usr_dict:
    #     for exp_index in usr:
    #         exp_index.append()

    with open(root_path+'labels.txt','r') as label_file:
        dict_reader = label_file.readlines()
        for row in dict_reader:
            row = row.split()
            usr_index = int(row[1])
            exp_index = int(row[0])
            starting_time = int(row[-2])
            ending_time = int(row[-1])
            label = int(row[2])
            usr_dict[usr_index][exp_index].append({'starting_time':starting_time,
                                                   'ending_time':ending_time,
                                                   'label':label})
    return usr_dict
# np.save('label_dict.npy',dict)
# print(usr_dict[10][20])