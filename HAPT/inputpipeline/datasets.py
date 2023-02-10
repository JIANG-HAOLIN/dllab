import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import csv
import matplotlib.pyplot as plt
import PIL.Image as Image
import os
from inputpipeline.preprocessing import preprocessing


##training dataset共2550, validation dataset共423
class creat_Dataset(Dataset):
    def __init__(self, mode='train', Window_shift=125, Window_length=250, batchsize=32, data_root_path='./RawData/'):
        print(f'{mode} ' + 'dataset:HAPT','mode:s2l')
        super(creat_Dataset, self).__init__()
        self.mode = mode
        if mode == 'train':
            self.usr_index = range(1, 22)
        elif mode == 'test':
            self.usr_index = range(22, 28)
        elif mode == 'val':
            self.usr_index = range(28, 31)
        # self.label_dict = np.load('./label_dict.npy',allow_pickle=True).item()
        self.label_dict = preprocessing(data_root_path)
        self.shift = Window_shift
        self.win_len = Window_length
        self.samples = []
        self.labels = []
        self.file = []
        self.interval = []
        for usr_idx in self.usr_index:
            for exp_idx in self.label_dict[usr_idx]:  ###!!!!!!迭代的是key!!!!!
                filename = self.label_dict[usr_idx][exp_idx][0].strip('o')
                acc_file_path = 'acc' + filename
                gyro_file_path = 'gyro' + filename
                acc_gyro = self.string_list_to_tensor(data_root_path, acc_file_path, gyro_file_path)
                for sequence in self.label_dict[usr_idx][exp_idx][1:]:
                    start = max(sequence['starting_time'], 250)
                    end = min(sequence['ending_time'], acc_gyro.shape[0] - 250)
                    sequence_length = end - start
                    num_windows = sequence_length / self.win_len
                    if num_windows >= 1:
                        num_shift = int((sequence_length - self.win_len) / self.shift)
                        for step in range(0, num_shift + 1):
                            window_start = start + step * self.shift
                            window = acc_gyro[window_start:window_start + 250].view(1, self.win_len, 6)
                            # print(f'sequence {window_start}--{window_start+self.win_len} from file exp{exp_idx} start with {window[0]}!')
                            self.samples.append(window)
                            self.labels.append(sequence['label'] - 1)
                            self.file.append((usr_idx, exp_idx))
                            self.interval.append((window_start, window_start + self.win_len))
                    else:
                        # print(f'sequence {start}--{end}(sequence length{sequence_length}) from file exp{exp_idx} can\'t form a window length of {self.win_len}!')
                        pass
        # print('\n\nin total %d sequences' % len(self.samples),'window length  %d samples' % self.samples[0].shape[1])
        self.samples = torch.cat(self.samples, dim=0)  #####！！若由list变tensor必须具有单一维度！！

    def string_list_to_tensor(self, data_root_path, acc_file_path=None, gyro_file_path=None):
        with open(data_root_path + acc_file_path) as acc:
            file_acc = acc.readlines()
        with open(data_root_path + gyro_file_path) as gyro:
            file_gyro = gyro.readlines()
        float_list = []
        for line_index in range(len(file_acc)):
            float_list.append(
                list(map(float, (file_acc[line_index].split()))) + list(map(float, (file_gyro[line_index].split()))))
        # data_array = np.array(float_list)
        data_tensor = torch.Tensor(float_list).view(-1, 1, 6)
        data_tensor = self.normalize(data_tensor)
        return data_tensor

    def normalize(self, tensor):
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

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        file = self.file[index]
        interval = self.interval[index]
        return sample, label, file, interval

    def __len__(self):
        return len(self.samples)


class creat_Dataset_s2s(Dataset):
    def __init__(self, mode='train', Window_shift=125, Window_length=250, batchsize=32, data_root_path='./RawData/'):
        print(f'{mode} ' + 'dataset:HAPT','mode:s2s')
        super(creat_Dataset_s2s, self).__init__()
        self.mode = mode
        if mode == 'train':
            self.usr_index = range(1, 22)
        elif mode == 'test':
            self.usr_index = range(22, 28)
        elif mode == 'val':
            self.usr_index = range(28, 31)
        # self.label_dict = np.load('./label_dict.npy',allow_pickle=True).item()
        self.label_dict = preprocessing(data_root_path)
        self.shift = Window_shift
        self.win_len = Window_length
        self.samples = []
        self.labels = []
        self.source = []
        for usr_idx in self.usr_index:
            for exp_idx in self.label_dict[usr_idx]:  ###!!!!!!迭代的是key!!!!!
                filename = self.label_dict[usr_idx][exp_idx][0].strip('o')
                acc_file_path = 'acc' + filename
                gyro_file_path = 'gyro' + filename
                acc_gyro = self.string_list_to_tensor(self.label_dict, usr_idx, exp_idx, data_root_path, acc_file_path,
                                                      gyro_file_path)
                total_sample = len(acc_gyro)
                total_step = int((total_sample - self.win_len) / self.shift)
                for step_idx in range(total_step + 1):
                    sample_sequence = []
                    label_sequence = []
                    source_sequence = []
                    window_start = self.shift * step_idx
                    windows_end = self.shift * step_idx + self.win_len
                    for data_idx in range(window_start, windows_end):
                        sample_sequence.append(acc_gyro[data_idx][-1])
                        label_sequence.append(acc_gyro[data_idx][-2])
                        source_sequence.append(acc_gyro[data_idx][0:3])
                    self.samples.append(torch.cat(sample_sequence, dim=0))
                    self.labels.append(torch.Tensor(label_sequence))
                    self.source.append(torch.Tensor(source_sequence))

        # print('\n\nin total %d sequences' % len(self.samples),'window length  %d samples' % self.samples[0].shape[1])
        # self.samples = torch.cat(self.samples,dim=0)#####！！若由list变tensor必须具有单一维度！！

    def string_list_to_tensor(self, label_dict, usr_idx, exp_idx, data_root_path, acc_file_path=None,
                              gyro_file_path=None):

        with open(data_root_path + acc_file_path) as acc:
            file_acc = acc.readlines()
        with open(data_root_path + gyro_file_path) as gyro:
            file_gyro = gyro.readlines()
        float_list = []
        for line_index in range(len(file_acc)):
            float_list.append(
                list(map(float, (file_acc[line_index].split()))) + list(map(float, (file_gyro[line_index].split()))))
        data_tensor = torch.Tensor(float_list).view(-1, 1, 6)
        data_tensor = self.normalize(data_tensor)
        sorted_data = []
        for sequence in label_dict[usr_idx][exp_idx][1:]:
            start = sequence['starting_time']
            end = sequence['ending_time']
            label = sequence['label'] - 1
            for i in range(start, (end + 1)):
                sorted_data.append([usr_idx, exp_idx, i, label, data_tensor[i-1, :, :]])
        return sorted_data

    def normalize(self, tensor):
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

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        source = self.source[index]
        return sample, label,source

    def __len__(self):
        return len(self.samples)


# train_dataset = creat_Dataset(mode='train',Window_shift=125,Window_length=250)
# test_dataset = creat_Dataset(mode='test',Window_shift=125,Window_length=250)
# validation_dataset = creat_Dataset(mode='validation',Window_shift=125,Window_length=250)
#
# train_loader = DataLoader(train_dataset,batch_size=3,shuffle=True)
# test_loader = DataLoader(test_dataset,batch_size=3,shuffle=True)
# validation_loader = DataLoader(validation_dataset,batch_size=3,shuffle=True)

def get_dataloader(how='s2l', mode='train', Window_shift=125, Window_length=250, batch_size=3, shuffle=True,
                   root_path='./RawData/'):
    if how == 's2l':
        dataset = creat_Dataset(mode=mode, Window_shift=125, Window_length=250, batchsize=batch_size,
                                data_root_path=root_path)
    elif how == 's2s':
        dataset = creat_Dataset_s2s(mode=mode, Window_shift=125, Window_length=250, batchsize=batch_size,
                                    data_root_path=root_path)
    # total_samples = dataset.__len__()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)




##测试dataset
# data_root_path = "/Users/hlj/Documents/NoSync.nosync/DL_Lab/HAPT Data Set/RawData/"
# train_loader = get_dataloader(how = 's2l',mode='train',Window_shift=125,Window_length=250,batch_size=1,shuffle=True,root_path=data_root_path)
#
# for step,data in enumerate(train_loader):##!!!!!!
#     if step <= 0:
#         print('\n\nsample batch shape:',data[0].shape,'\n',data[0],'\n\n label:',data[1],'\n\n usr index =',int(data[2][0]),'\n\nfile exp index =', int(data[2][1]),\
#                 '\n\n time interval = ',int(data[3][0]),'-->',int(data[3][1]))



# data_root_path = "/Users/hlj/Documents/NoSync.nosync/DL_Lab/HAPT Data Set/RawData/"
# train_loader = get_dataloader(how='s2s', mode='train', Window_shift=125, Window_length=250, batch_size=2, shuffle=True,
#                               root_path=data_root_path)
# for step, (sample,label,source) in enumerate(train_loader):  ##!!!!!!
#     sample = sample[0][100]
#     label = label[0][100]
#     source = source[0][100]
#     if step <= 0:
#         print('\n\n''\n','sample:',sample, '\n\n label:', label, '\n\n usr index =',
#               source[0], '\n\nfile exp index =', source[1], \
#               '\n\n time instance = ', source[2])

######