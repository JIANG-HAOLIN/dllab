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
import pandas as pd
# path = "/Users/hlj/Documents/NoSync.nosync/DL_Lab/IDRID_dataset/images/resized_test/IDRiD_001.jpg"

# root_path = "/no_backups/s1434/lab/labdata/"
root_path = os.path.join(r"D:\labdata")



class Lab_Dataset(Dataset):

    def __init__(self, root, train=True, transform=None, reg=False, weighting=False):

        super(Lab_Dataset, self).__init__()
        self.reg = reg
        self.train = train
        self.transform = transform
        self.weighting = weighting

        if self.train :
            file_annotation = root + "/IDRID_dataset/labels/train.csv"
            img_folder = root + "/IDRID_dataset/images/resized_train/"
        else:
            file_annotation = root + "/IDRID_dataset/labels/test.csv"
            img_folder = root + "/IDRID_dataset/images/resized_test/"

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

        assert len(label_dict["Image name"]) == len(label_dict["Retinopathy grade"])
        num_label = len(label_dict["Image name"])

        self.filenames = []
        self.labels = []
        self.img_folder = img_folder
        for i in range(num_label):
            self.filenames.append(label_dict["Image name"][i])
            if self.weighting:
                if int(label_dict["Retinopathy grade"][i]) == 1:
                    self.filenames.append(label_dict["Image name"][i])

            if not self.reg:
                self.labels.append(0. if int(label_dict["Retinopathy grade"][i]) <= 1 else 1.)
                if self.weighting:
                    if int(label_dict["Retinopathy grade"][i]) == 1:
                        self.labels.append(0. if int(label_dict["Retinopathy grade"][i]) <= 1 else 1.)

            else:
                self.labels.append(int(label_dict["Retinopathy grade"][i]))

    def __getitem__(self, index):
        img_name = self.img_folder + self.filenames[index]
        label = self.labels[index]
        img = Image.open(img_name + ".jpg")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.filenames)


train_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5424, 0.2638, 0.0875],
            std=[0.4982, 0.4407, 0.2826]),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5)
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

kaggle_transforms_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize([512, 512]),
        transforms.Normalize(
            mean=[0.4408, 0.3105, 0.2273],
            std=[0.4965, 0.4627, 0.4191]),
    ]
)

kaggle_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize([512, 512]),
        transforms.Normalize(
            mean=[0.4408, 0.3105, 0.2273],
            std=[0.4965, 0.4627, 0.4191]),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5)
    ]
)

train_dataset = Lab_Dataset(root=root_path, train=True, transform=train_transforms, weighting=False)
train_dataset_reg = Lab_Dataset(root=root_path, train=True, transform=train_transforms, reg=True)
test_dataset = Lab_Dataset(root=root_path, train=False, transform=val_transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)
train_loader_reg = DataLoader(dataset=train_dataset_reg, batch_size=2, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False)

class Kaggle_Dataset(Dataset):

    def __init__(self, root, transform=None):

        super(Kaggle_Dataset, self).__init__()
        self.kaggle_path = root + "/kaggle_big"
        self.kaggle_file_list = os.listdir(self.kaggle_path)
        self.transform = transform

    def __getitem__(self, index):
        img_name = self.kaggle_file_list[index]
        img = Image.open(os.path.join(self.kaggle_path, img_name))
        label = -1
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.kaggle_file_list)

class Kaggle_Dataset_test(Dataset):

    def __init__(self, root, transform=None):

        super(Kaggle_Dataset_test, self).__init__()
        self.kaggle_path = root + "/kaggle_dataset"
        self.kaggle_file_list = os.listdir(self.kaggle_path)
        self.transform = transform
        self.labels = pd.read_csv(os.path.join(root, "kaggle_test.csv"), index_col=0)

    def __getitem__(self, index):
        img_name = self.kaggle_file_list[index]
        img = Image.open(os.path.join(self.kaggle_path, img_name))
        img_name = img_name.split(".")[0]
        label = 0 if int(self.labels.loc[img_name]) <= 1 else 1
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.kaggle_file_list)
        # return self.labels.shape[0]



kaggle_data = Kaggle_Dataset(root_path, transform=kaggle_transforms)
kaggle_loader = DataLoader(dataset=kaggle_data, batch_size=2, shuffle=True)

kaggle_test_data = Kaggle_Dataset_test(root_path, transform=kaggle_transforms_test)
kaggle_test_loader = DataLoader(dataset=kaggle_test_data, batch_size=2, shuffle=False)
#







