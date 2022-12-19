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
from HAPT.inputpipeline.preprocessing import usr_dict


data_root_path = "/Users/hlj/Documents/NoSync.nosync/DL_Lab/HAPT Data Set/RawData/"


class creat_Dataset(Dataset):
    def __init__(self,mode='train',Window_shift=125,Window_length=250):
        super(creat_Dataset,self).__init__()
        if mode == 'train':
            self.usr_index = range(1,22)
        elif mode == 'test':
            self.usr_index = range(22, 28)
        else:
            self.usr_index = range(28, 31)
        # self.label_dict = np.load('./label_dict.npy',allow_pickle=True).item()
        self.label_dict = usr_dict
        self.shift = Window_shift
        self.win_len = Window_length
        samples=[]
        labels=[]
        for usr_idx in self.usr_index:
            for exp_idx in self.label_dict[usr_idx]:###!!!!!!迭代的是key!!!!!
                filename = self.label_dict[usr_idx][exp_idx][0].strip('o')
                acc_file_path = 'acc'+ filename
                gyro_file_path = 'gyro' + filename
                with open(data_root_path+acc_file_path) as acc:
                    file_acc = acc.readlines()
                with open(data_root_path+gyro_file_path) as gyro:
                    file_gyro = gyro.readlines()
                for sequence in self.label_dict[usr_idx][exp_idx][1:]:
                    start = max(sequence['starting_time'],250)
                    end = min(sequence['ending_time'],len(file_acc)-250)
                    num_shift = int((end-start)/self.shift) -2
                    for num_shift in range(num_shift):
                        window=[]
                        for win in range(self.win_len):
                            window.append(list(map(float,(file_acc[start+num_shift*self.shift+win].split())))+
                                          list(map(float,(file_gyro[start + num_shift * self.shift + win].split())))
                                          )
                        samples.append(window)
                        labels.append(sequence['label'])
        self.samples = torch.Tensor(samples)#####！！必须具有单一维度！！
        print('in total %d sequences' % self.samples.shape[0],
              'each sequence has %d samples' % self.samples.shape[1])
        self.labels = torch.Tensor(labels)
    def __getitem__(self,index):
        sample = self.samples[index]
        label = self.labels[index]
        return sample,label
    def __len__(self):
        return len(self.samples)

# train_dataset = creat_Dataset(mode='train',Window_shift=125,Window_length=250)
# test_dataset = creat_Dataset(mode='test',Window_shift=125,Window_length=250)
# validation_dataset = creat_Dataset(mode='validation',Window_shift=125,Window_length=250)
#
# train_loader = DataLoader(train_dataset,batch_size=3,shuffle=True)
# test_loader = DataLoader(test_dataset,batch_size=3,shuffle=True)
# validation_loader = DataLoader(validation_dataset,batch_size=3,shuffle=True)

def get_dataloader(mode='train',Window_shift=125,Window_length=250,batch_size=3,shuffle=True):
    dataset = creat_Dataset(mode=mode,Window_shift=125,Window_length=250)
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)




# train_loader = get_dataloader(mode='train',Window_shift=125,Window_length=250,batch_size=3,shuffle=True)
# for step,data in enumerate(train_loader):##!!!!!!
#     if step <= 0:
#         print(data[0].shape,data[1])








