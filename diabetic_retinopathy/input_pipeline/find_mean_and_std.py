import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# train_dataset = datasets.CIFAR10(root="./CIFAR", train=True, transform=transforms.ToTensor(), download=True)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

def get_mean_std(loader):
#Var = E(X^2) - E(X)^2
    channels_sum, channels_square_sum, num_batches = 0, 0, 0
    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_square_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum/num_batches
    std = (channels_sum/num_batches -mean**2)**0.5
    return mean, std


