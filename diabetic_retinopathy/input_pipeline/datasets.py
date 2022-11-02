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

# 把图像转换为tensor，和对应的标签包装到dataset里

# path = "/Users/hlj/Documents/NoSync.nosync/DL_Lab/IDRID_dataset/images/resized_test/IDRiD_001.jpg"

# root_path = "/no_backups/s1434/lab/labdata/"
root_path = os.path.join(r"D:\labdata")



class Lab_Dataset(Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):

        super(Lab_Dataset, self).__init__()

        self.train = train
        self.transform = transform
        self.target_transform = target_transform

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
            self.labels.append(label_dict["Retinopathy grade"][i])
            # self.labels[2].append(label_dict["Risk of macular edema "][i])

    def __getitem__(self, index):
        img_name = self.img_folder + self.filenames[index]
        label = self.labels[index]
        img = Image.open(img_name + ".jpg")
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.filenames)


train_dataset = Lab_Dataset(root=root_path, train=True, transform=transforms.ToTensor())
test_dataset = Lab_Dataset(root=root_path, train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=True)

