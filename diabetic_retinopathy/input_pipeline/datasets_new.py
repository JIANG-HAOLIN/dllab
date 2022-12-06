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
# from find_mean_and_std import get_mean_std
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

###!!!!注意有两个config!!!!

# path = "/Users/hlj/Documents/NoSync.nosync/DL_Lab/IDRID_dataset/images/resized_test/IDRiD_001.jpg"

# root_path = "/no_backups/s1434/lab/labdata/"
# root_path = os.path.join(r"D:\labdata")
# root_path = os.path.join('/no_backups/s1434/lab/dl-lab-22w-team15/diabetic_retinopathy/')



class Lab_Dataset(Dataset):

    def __init__(self, root='../',which='IDRID_dataset', train=True, transform=None,wanted_size=None, reg=False):

        super(Lab_Dataset, self).__init__()

        self.train = train
        self. transform = transform
        self.reg = reg
        if which == 'IDRID_dataset':
            if self.train:
                file_annotation = root + "/labels/train.csv"
                img_folder = root + which+"_preprocessed_train_size%sx%s/" %(wanted_size,wanted_size)
            else:
                file_annotation = root + "/labels/test.csv"
                img_folder = root + which+"_preprocessed_test_size%sx%s/" %(wanted_size,wanted_size)
        else:
            if self.train :
                file_annotation = root + "/labels/" +which+"_test.csv"
                img_folder = root + which+"_preprocessed_train_size%sx%s/" %(wanted_size,wanted_size)
            else:
                file_annotation = root + "/labels/" +which+"_test.csv"
                img_folder = root + which+"_preprocessed_test_size%sx%s/" %(wanted_size,wanted_size)


        label_dict = {
                        "Image name": [],
                        "Retinopathy grade": [],
                        # "Risk of macular edema ": []
                      }

        with open(file_annotation, 'r') as file:
            data_DictReader = csv.DictReader(file)
            for row in data_DictReader:
                label_dict["Image name"].append(row["Image name"])
                label_dict["Retinopathy grade"].append(row["Retinopathy grade"])
                # label_dict["Risk of macular edema "].append(row["Risk of macular edema "])

        assert len(label_dict["Image name"])==len(label_dict["Retinopathy grade"])
        num_label = len(label_dict["Image name"])

        self.filenames = []
        self.labels = []
        self.img_folder = img_folder
        for i in range(num_label):
            self.filenames.append(label_dict["Image name"][i])
            # self.labels.append(0. if int(label_dict["Retinopathy grade"][i]) <= 1 else 1.)
            # self.labels[2].append(label_dict["Risk of macular edema "][i])
            if not self.reg:
                self.labels.append(0. if int(label_dict["Retinopathy grade"][i]) <= 1 else 1.)
            else:
                self.labels.append(int(label_dict["Retinopathy grade"][i]))

    def __getitem__(self, index):
        img_name = self.img_folder + self.filenames[index]
        label = self.labels[index]
        img = Image.open(img_name + ".jpg")
        # print(img)
        if self.transform is not None:
            # img = np.array(img)
            # img = self.transform(image=img)["image"]
            # img = img.permute(2, 0, 1)
            img = self.transform(img)
            # print(img.size())
        return img, label

    def __len__(self):
        return len(self.filenames)


train_transforms = transforms.Compose(
    [


        # transforms.RandomResizedCrop(size=int(wanted_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        # transforms.RandomRotation(degrees=45),
        # transforms.RandomApply([
        #                         transforms.RandomRotation(degrees=90),
        #                         transforms.RandomRotation(degrees=30)
        #                         ],
        #                         p=0.2),
        transforms.RandomGrayscale(p=0.3),
        transforms.RandomApply([transforms.RandomAffine(shear=30,degrees=0)],p=0.2),



        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5424, 0.2638, 0.0875],
            std=[0.4982, 0.4407, 0.2826],
        ),

    ]
)

val_transforms = transforms.Compose(
    [

        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5424, 0.2638, 0.0875],
            std=[0.4982, 0.4407, 0.2826],
        ),

    ]
)


class Lab_kaggle_Dataset(Dataset):

    def __init__(self, root='../',which='IDRID_dataset', train=True, transform=None,wanted_size=None, reg=False):

        super(Lab_kaggle_Dataset, self).__init__()

        self.train = train
        self. transform = transform
        self.reg = reg
        if which == 'IDRID_dataset' :
            if self.train :
                file_annotation = root + "/labels/train.csv"
                img_folder = root + which+"_preprocessed_train_size%sx%s/" %(wanted_size,wanted_size)
            else:
                file_annotation = root + "/labels/test.csv"
                img_folder = root + which+"_preprocessed_train_size%sx%s/" %(wanted_size,wanted_size)
        else :
            if self.train :
                file_annotation = root + "/labels/" +which+"_train.csv"
                img_folder = root + which+"_preprocessed_train_size%sx%s/" %(wanted_size,wanted_size)
            else:
                file_annotation = root + "/labels/" +which+"_test.csv"
                img_folder = root + which+"_preprocessed_train_size%sx%s/" %(wanted_size,wanted_size)



        with open(file_annotation, 'r') as file:
            data_DictReader = csv.DictReader(file)
            # for row in data_DictReader:
            #     label_dict[row['image']] = row['level']####!!字典添加元素可以直接通过对字典中不存在的元素进行赋值!!
            label_dict={row['image']:int(row['level']) for row in data_DictReader}
            ####每个rou为一个字典！！字典key为'image' 和 'label'！！


        self.filenames = []
        self.labels = []
        self.img_folder = img_folder
        for files in os.listdir(img_folder):
            if files != '.DS_Store':
                filename = files.strip('.jpeg')
                self.filenames.append(filename)
                if not self.reg:
                    self.labels.append(0. if int(label_dict[filename]) <= 1 else 1.)
                else:
                    self.labels.append(int(label_dict[filename]))

    def __getitem__(self, index):
        img_name = self.img_folder + self.filenames[index]
        label = self.labels[index]
        img = Image.open(img_name + ".jpeg")
        if self.transform is not None:
            # img = np.array(img)
            # img = self.transform(image=img)["image"]
            # img = img.permute(2, 0, 1)
            img = self.transform(img)
            # print(img.size())
        return img, label

    def __len__(self):
        return len(self.filenames)



def get_both_loader(root_path='../',which='IDRID_dataset',batch_size=12,wanted_size=728,reg=False):
    train_dataset = Lab_Dataset(root=root_path, which=which,
                                train=True, transform=train_transforms, wanted_size=wanted_size,reg=reg)
    test_dataset = Lab_Dataset(root=root_path, which=which,
                               train=False, transform=val_transforms, wanted_size=wanted_size,reg=reg)
    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def get_kaggle_loader(train=False, root_path='../',which='kaggle',batch_size=12,wanted_size=728,reg=False):

    kaggle_test_dataset = Lab_kaggle_Dataset(root=root_path, which=which,
                               train=train, transform=val_transforms, wanted_size=wanted_size,reg=reg)

    test_loader = DataLoader(dataset=kaggle_test_dataset,batch_size=batch_size, shuffle=True)

    return test_loader




