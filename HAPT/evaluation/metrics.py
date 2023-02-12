import numpy as np


def compute_accuracy(output, label):
    total = output.shape[0]
    output = np.argmax(output, axis=1)
    correct = output == label ####[False False False]
    incorrect = output != label

    accuracy = sum(correct)/total
    return accuracy

