import numpy as np
import torch
from inputpipeline.datasets import get_dataloader
from inputpipeline.HAR_Dataset import get_dataloader_HAR

import config
from tqdm import tqdm
from models.multi_models import *
from models.conv_lstm import *
from models.transformer import Encoder

from inputpipeline.preprocess_input import preprocess_input
from evaluation.metrics import compute_accuracy
import matplotlib
import matplotlib.pyplot as plt

from evaluation.confusion_matrix import generate_confusion_Matrix



opt = config.read_arguments()

batch_size = 1
root_path =opt.root_path
shift_length = opt.shift_length
window_size = opt.window_size
device = opt.device
hidden_size = opt.hidden_size
num_layers = opt.num_layers
lr = opt.lr
bidirectional = opt.bidirectional
out_name = opt.out_name+'b'+str(batch_size)+'hidden'+str(hidden_size)+'layer'+str(num_layers)


# test_loader = get_dataloader_HAR(mode='test', Window_shift=125, Window_length=250, batch_size=batch_size,
#                                        shuffle=True, root_path='./realworld2016_dataset/')

# how = 's2l'
# file_name = 'bi4b64hidden12layer2best_epoch.pth'

how = 's2s'
test_loader = get_dataloader_HAR(mode='test',Window_shift=125,Window_length=250,
                                   batch_size=batch_size,shuffle=False,root_path='./realworld2016_dataset/')
# file_name = 'pthHAPT_lstm_s2s_lstm_cnn_b64hidden24layer3_88_1540_.pth'
file_name = 'pthHAR_transformer_transformerb64hidden12layer2_90_2300_.pth'

# file_name = 'HAPT_transformer_validation_accuracyb32hidden24layer2best_epoch.pth'
# mdl = model_HAPT(batchsize=batch_size,device=device,hidden_size =24,
#             num_layers=3,bidirectional=True,window_size=250,how=how).to(device)
# mdl = model_gru_HAPT(batchsize=batch_size,device=device,hidden_size =12,
#             num_layers=2,bidirectional=True,window_size=250).to(device)
# mdl = Conv_lstm(device=device, num_lstm_layers=num_layers, hidden_size=96, batch_size=batch_size, how=how).to(device)
# mdl = model_transformer_HAPT(batchsize=batch_size, num_classes=12, seq_len=window_size, how=how).to(device)
# mdl = model_transformer_HAPT(batchsize=batch_size)
# mdl = model_gru_HAR(batchsize=batch_size, device=device, hidden_size=12,
#                         num_layers=2, bidirectional=True, window_size=window_size,).to(device)
# mdl = Conv_lstm_realworld(device=device,num_lstm_layers=2,hidden_size=96,batch_size=batch_size)
mdl = model_transformer_HAR(batchsize=batch_size,num_classes=8,seq_len=window_size)

mdl.load_state_dict(torch.load(file_name,map_location=torch.device('cpu')))
mdl.eval()
loss_computer = torch.nn.CrossEntropyLoss()
loss_val = 0
acu = 0
accu=0
true_dict = {9:{s: [] for s in range(1,9)},10:{s: [] for s in range(1,9)}}
result_dict = {9:{s: [] for s in range(1,9)},10:{s: [] for s in range(1,9)}}
label_dict1 = {'climbingdown': 1, 'climbingup': 2, 'jumping': 3, 'lying': 4,
                   'running': 5, 'sitting': 6, 'standing': 7, 'walking': 8}
for step_test, test_data in tqdm(enumerate(test_loader)):
    test_input = test_data[0]
    test_label = test_data[1]
    file = test_data[2]
    interval = test_data[3]
    with torch.no_grad():
        test_input, test_label = preprocess_input(test_input, test_label, device)
        test_output = mdl(test_input)
        for i in range(batch_size):
            usr = int(file[0][i])
            exp = label_dict1[(file[1][i])]
            start = int(interval[0][i])
            end = int(interval[1][i])
            true_dict[usr][exp].append([[start,end],test_label])
            result_dict[usr][exp].append([[start,end],np.argmax(test_output, axis=1)])
        loss_val = (loss_val * step_test + loss_computer(test_output, test_label)) / (step_test + 1)  ##??
        # print(val_output,val_label)
        accuracy = compute_accuracy(test_output.detach().cpu().numpy(), test_label.detach().cpu().numpy())  ##??
        # print(accuracy)
        accu = (accu * step_test + accuracy) / (step_test + 1)  ##validation set的平均accuracy
print('accuracy =',accu)
np.save('HAR_result_dict_s2l.npy',result_dict)
np.save('HAR_true_dict_s2l.npy',true_dict)



def show_confusion_matrix(result_dict,true_dict,how):

    label_dict = {0:'climbingdown', 1:'climbingup', 2:'jumping', 3:'lying',
                       4:'running', 5:'sitting', 6:'standing', 7:'walking'}

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
    label_list1 = np.unique(np.concatenate((results, y_true)))
    label_list = []
    for i in label_list1:
        label_list.append(label_dict[i])
    # all_possible_labels = np.concatenate((results, y_true))
    # for i in range(8):
    #     if i in all_possible_labels:
    #         label_list.append(label_dict[i])
    generate_confusion_Matrix(y_true, results, label_list=label_list)

show_confusion_matrix(result_dict='./HAR_result_dict_s2l.npy',true_dict='./HAR_true_dict_s2l.npy',how=None)

