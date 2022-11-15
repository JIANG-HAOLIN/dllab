import torch
import torch.nn as nn
import numpy as np


output = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.9, 0.3, 0.1])
label = np.array([1, 0, 0, 1, 1, 0., 1, 0])



def compute_matrix(ouput, label):
    total = ouput.shape[0]
    ouput[ouput >= 0.5] = 1
    ouput[ouput < 0.5] = 0
    correct = ouput == label
    incorrect = ouput != label
    TP = np.count_nonzero(correct[ouput == 1] == True)/total
    TN = np.count_nonzero(correct[ouput == 0] == True)/total
    FP = np.count_nonzero((incorrect)[ouput == 1] == True)/total
    FN = np.count_nonzero((incorrect)[ouput == 0] == True) / total

    return [TP, TN, FP, FN]



print(compute_matrix(output, label))



