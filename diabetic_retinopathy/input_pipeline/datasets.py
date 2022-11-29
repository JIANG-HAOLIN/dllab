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

# path = "/Users/hlj/Documents/NoSync.nosync/DL_Lab/IDRID_dataset/images/resized_test/IDRiD_001.jpg"

# root_path = "/no_backups/s1434/lab/labdata/"
root_path = os.path.join(r"D:\labdata")



class Lab_Dataset(Dataset):

    def __init__(self, root, train=True, transform=None, reg=False):

        super(Lab_Dataset, self).__init__()
        self.reg = reg
        self.train = train
        self.transform = transform

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

        assert len(label_dict["Image name"])==len(label_dict["Retinopathy grade"])
        num_label = len(label_dict["Image name"])

        self.filenames = []
        self.labels = []
        self.img_folder = img_folder
        for i in range(num_label):
            self.filenames.append(label_dict["Image name"][i])
            if not self.reg:
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

train_dataset = Lab_Dataset(root=root_path, train=True, transform=train_transforms)
train_dataset_reg = Lab_Dataset(root=root_path, train=True, transform=train_transforms, reg=True)
test_dataset = Lab_Dataset(root=root_path, train=False, transform=val_transforms)


train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
train_loader_reg = DataLoader(dataset=train_dataset_reg, batch_size=4, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False)

# mean, std = get_mean_std(train_loader)

#tensor([0.5424, 0.2638, 0.0875])
#tensor([0.4982, 0.4407, 0.2826])


