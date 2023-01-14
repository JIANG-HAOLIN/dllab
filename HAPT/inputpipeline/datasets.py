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
from inputpipeline.preprocessing import preprocessing


# data_root_path = "/Users/hlj/Documents/NoSync.nosync/DL_Lab/HAPT Data Set/RawData/"


##training dataset共2550, validation dataset共423
class creat_Dataset(Dataset):
    def __init__(self,mode='train',Window_shift=125,Window_length=250,batchsize=32,data_root_path = './RawData/'):
        super(creat_Dataset,self).__init__()
        if mode == 'train':
            self.usr_index = range(1,22)
        elif mode == 'test':
            self.usr_index = range(22, 28)
        else:
            self.usr_index = range(28, 31)
        # self.label_dict = np.load('./label_dict.npy',allow_pickle=True).item()
        self.label_dict = preprocessing(data_root_path)
        self.shift = Window_shift
        self.win_len = Window_length
        self.samples=[]
        self.labels=[]
        self.file=[]
        self.interval=[]
        for usr_idx in self.usr_index:
            for exp_idx in self.label_dict[usr_idx]:###!!!!!!迭代的是key!!!!!
                filename = self.label_dict[usr_idx][exp_idx][0].strip('o')
                acc_file_path = 'acc'+ filename
                gyro_file_path = 'gyro' + filename
                acc_gyro = self.string_list_to_tensor(data_root_path,acc_file_path,gyro_file_path)
                for sequence in self.label_dict[usr_idx][exp_idx][1:]:
                    start = max(sequence['starting_time'],250)
                    end = min(sequence['ending_time'],acc_gyro.shape[0]-250)
                    sequence_length = end-start
                    num_windows = sequence_length/self.win_len
                    if num_windows >= 1:
                        num_shift = int((sequence_length-self.win_len)/self.shift)
                        for step in range(0,num_shift+1):
                            window_start = start + step * self.shift
                            window = acc_gyro[window_start:window_start+250].view(1,self.win_len,6)
                            # print(f'sequence {window_start}--{window_start+self.win_len} from file exp{exp_idx} start with {window[0]}!')
                            self.samples.append(window)
                            self.labels.append(sequence['label']-1)
                            self.file.append((usr_idx,exp_idx))
                            self.interval.append((window_start,window_start+self.win_len))
                    else :
                        print(f'sequence {start}--{end}(sequence length{sequence_length}) from file exp{exp_idx} can\'t form a window length of {self.win_len}!')
        print('\n\nin total %d sequences' % len(self.samples),'window length  %d samples' % len(self.samples[0]))
        self.samples = torch.cat(self.samples,dim=0)#####！！若由list变tensor必须具有单一维度！！


    def string_list_to_tensor(self,data_root_path,acc_file_path=None,gyro_file_path=None):
        with open(data_root_path + acc_file_path) as acc:
            file_acc = acc.readlines()
        with open(data_root_path + gyro_file_path) as gyro:
            file_gyro = gyro.readlines()
        float_list = []
        for line_index in range(len(file_acc)):
            float_list.append( list(map(float,(file_acc[line_index].split()))) + list(map(float,(file_gyro[line_index].split()))) )
        # data_array = np.array(float_list)
        data_tensor = torch.Tensor(float_list).view(-1,1,6)
        data_tensor = self.normalize(data_tensor)
        return data_tensor

    def normalize(self,tensor):
        mean = torch.mean(tensor,dim=(0,1))
        std = torch.std(tensor,dim=(0,1))
        tensor = (tensor-mean)/std
        return tensor

    def __getitem__(self,index):
        sample = self.samples[index]
        label = self.labels[index]
        file = self.file[index]
        interval = self.interval[index]
        return sample,label,file,interval
    def __len__(self):
        return len(self.samples)

# train_dataset = creat_Dataset(mode='train',Window_shift=125,Window_length=250)
# test_dataset = creat_Dataset(mode='test',Window_shift=125,Window_length=250)
# validation_dataset = creat_Dataset(mode='validation',Window_shift=125,Window_length=250)
#
# train_loader = DataLoader(train_dataset,batch_size=3,shuffle=True)
# test_loader = DataLoader(test_dataset,batch_size=3,shuffle=True)
# validation_loader = DataLoader(validation_dataset,batch_size=3,shuffle=True)

def get_dataloader(mode='train',Window_shift=125,Window_length=250,batch_size=3,shuffle=True,root_path='./RawData/'):
    dataset = creat_Dataset(mode=mode,Window_shift=125,Window_length=250,batchsize=batch_size,data_root_path = root_path)
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,drop_last=True)



##测试dataset
# train_loader = get_dataloader(mode='train',Window_shift=125,Window_length=250,batch_size=1,shuffle=True)
# for step,data in enumerate(train_loader):##!!!!!!
#     if step <= 0:
#         print('\n\nsample batch shape:',data[0].shape,'\n',data[0],'\n\n label:',data[1],'\n\n usr index =',int(data[2][0]),'\n\nfile exp index =', int(data[2][1]),\
#                 '\n\n time interval = ',int(data[3][0]),'-->',int(data[3][1]))








