import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import csv
import matplotlib.pyplot as plt
import PIL.Image as Image
import os
from inputpipeline.preprocessing import preprocessing


root_path ='/Users/hlj/Documents/NoSync.nosync/DL_Lab/dl-lab-22w-team15/HAPT/realworld2016_dataset/'

class dataset_HAR(Dataset):
    def __init__(self,mode='train',position='chest',root_path=root_path,win_len=250,shift=125):
        super(dataset_HAR,self).__init__()
        self.set_dict = {'train':[1,2,5,8,11,12,13,15],'validation':[3],'test':[9,10]}
        self.label_dict = {'climbingdown':1,'climbingup':2,'jumping':3,'lying':4,
                           'running':5,'sitting':6,'standing':7,'walking':8}
        self.subjects = self.set_dict[mode]
        self.win_len = win_len
        self.shift = shift
        self.samples = []
        self.labels = []
        self.file = []
        self.interval = []

        for subject in self.subjects:
            subject_root_path = root_path+'/proband'+str(subject)+'/data/'
            for act,label in self.label_dict.items():### !!!!!


                acc_folder_path = 'acc_' + act + '_csv/'
                gyr_folder_path = 'gyr_' + act + '_csv/'
                acc_gyro = self.csv2tensor(root_path=subject_root_path,
                                           acc_folder_path=acc_folder_path,
                                           gyr_folder_path=gyr_folder_path,
                                           position=position)

                start = 250
                end = acc_gyro.shape[0]-250
                sequence_length = end - start
                # print(sequence_length)
                num_windows = sequence_length / self.win_len
                if num_windows >= 1:
                    num_shift = int((sequence_length - self.win_len) / self.shift)
                    for step in range(0, num_shift + 1):
                        window_start = start + step * self.shift
                        window = acc_gyro[window_start:window_start + 250].view(1, self.win_len, 6)
                        # print(f'sequence {window_start}--{window_start+self.win_len} from file exp{exp_idx} start with {window[0]}!')
                        self.samples.append(window)
                        self.labels.append(label - 1)
                        self.file.append((subject,act))
                        self.interval.append((window_start, window_start + self.win_len))
                else:
                    print('error')
            # print('\n\nin total %d sequences' % len(self.samples),
            #   'window length  %d samples' % self.samples[0].shape[1])
        self.samples = torch.cat(self.samples, dim=0)  #####！！若由list变tensor必须具有单一维度！！


    def csv2tensor(self,root_path,acc_folder_path,gyr_folder_path,position):
        acc_file_path = acc_folder_path[:-4]+position+'.csv'
        Gyroscope_file_path='Gyroscope_'+ gyr_folder_path[4:-4]+position+'.csv'
        # print(acc_file_path,Gyroscope_file_path)
        with open(root_path + acc_folder_path + acc_file_path) as acc:
            acc = acc.readlines()[1:]
        with open(root_path + gyr_folder_path + Gyroscope_file_path) as gyro:
            gyro = gyro.readlines()[1:]
        length = min(len(acc),len(gyro))
        float_list = []
        for line_index in range(length):
            float_list.append(
                list(map(float, (acc[line_index].split(','))))[2:] + list(map(float, (gyro[line_index].split(','))))[2:])
        # data_array = np.array(float_list)
        data_tensor = torch.Tensor(float_list).view(-1, 1, 6)
        # data_tensor = self.normalize(data_tensor)
        return data_tensor

    def normalize(self,tensor):
        # if self.mode == 'train':
        #     mean = torch.mean(tensor,dim=(0,1))
        #     std = torch.std(tensor,dim=(0,1))
        #     tensor = (tensor-mean)/std
        # else:
        #     tensor = tensor
        mean = torch.mean(tensor, dim=(0, 1))
        std = torch.std(tensor, dim=(0, 1))
        tensor = (tensor - mean) / std
        return tensor

    def __getitem__(self,index):
        sample = self.samples[index]
        label = self.labels[index]
        file = self.file[index]
        interval = self.interval[index]
        return sample,label,file,interval
    def __len__(self):
        return len(self.samples)





def get_dataloader_HAR(mode='train',Window_shift=125,Window_length=250,batch_size=3,shuffle=True,root_path='/Users/hlj/Documents/NoSync.nosync/DL_Lab/dl-lab-22w-team15/HAPT/realworld2016_dataset/'):
    dataset =dataset_HAR(mode=mode,shift=125,win_len=250,root_path = root_path)
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,drop_last=True)

##测试dataset
# train_loader = get_dataloader_HAR(mode='validation',Window_shift=125,Window_length=250,batch_size=1,shuffle=True)
# for step,data in enumerate(train_loader):##!!!!!!
#     if step <= 1:
#         print('\n\nsample batch shape:',data[0].shape,'\n',data[0],'\n\n label:',int(data[1]),'\n\n usr index =',int(data[2][0]),'\n\nfile exp index =', data[2][1],
#                 '\n\n time interval = ',int(data[3][0]),'-->',int(data[3][1]))



