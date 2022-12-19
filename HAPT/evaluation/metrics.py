import numpy as np


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