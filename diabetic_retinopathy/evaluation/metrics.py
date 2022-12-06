import torch
import torch.nn as nn
import numpy as np


output = np.array([[0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9]])
label = np.array([1, 1, 0, 1])


def compute_matrix(output, label):
    total = output.shape[0]
    output[output >= 0.5] = 1
    output[output < 0.5] = 0
    correct = output == label
    incorrect = output != label
    TP = np.count_nonzero(correct[output == 1] == True)/total
    TN = np.count_nonzero(correct[output == 0] == True)/total
    FP = np.count_nonzero((incorrect)[output == 1] == True)/total
    FN = np.count_nonzero((incorrect)[output == 0] == True) / total

    return [TP, TN, FP, FN], TP+TN


def compute_matrix_CE(output, label):
    total = output.shape[0]
    output = np.argmax(output, axis=1)
    correct = output == label
    incorrect = output != label

    TP = np.count_nonzero(correct[output == 1] == True)/total
    TN = np.count_nonzero(correct[output == 0] == True)/total
    FP = np.count_nonzero((incorrect)[output == 1] == True)/total
    FN = np.count_nonzero((incorrect)[output == 0] == True) / total

    return [TP, TN, FP, FN], TP+TN

