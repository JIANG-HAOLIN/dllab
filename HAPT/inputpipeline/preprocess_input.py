import torch

def preprocess_input(input,label,device):
    input = input.to(device)
    batch_size, sequence_length, feature_channels = input.shape
    label = label.long()
    label = label.to(device)
    return input, label
