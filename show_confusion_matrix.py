import numpy as np
from evaluation.confusion_matrix import generate_confusion_Matrix

def show_confusion_matrix(result_dict,true_dict,how):
    label_dict = {0:'walking',1:'walking_upstairs',2:'walking_downstairs',3:'sitting',4:'standing',5:'laying',
            6:'stand_to_sit',7:'sit_to_stand',8:'sit_to_lie',9:'lie_to_sit',10:'stand_to_lie',11:'lie_to_stand'}
    result_dict = np.load(result_dict,allow_pickle=True).item()
    true_dict = np.load(true_dict,allow_pickle=True).item()
    result_label = []
    true_label =[]
    ## confusion Matrix for whole test set
    results = []
    y_true = []
    for usr, files in result_dict.items():
        for file, datas in files.items():
            for data in datas:
                results.append(int(data[1]))
    for usr, files in true_dict.items():
        for file, datas in files.items():
            for data in datas:
                y_true.append(int(data[1]))
    results = np.array(results)
    y_true = np.array(y_true)
    label_list = []
    all_possible_labels = np.concatenate((results, y_true))
    for i in range(12):
        if i in all_possible_labels:
            label_list.append(label_dict[i])
    generate_confusion_Matrix(y_true, results, label_list=label_list)

show_confusion_matrix(result_dict='./result_dict_s2s.npy',true_dict='./true_dict_s2s.npy',how=None)