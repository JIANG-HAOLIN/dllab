import numpy as np
import torch
from inputpipeline.datasets import get_dataloader
from inputpipeline.HAR_Dataset import get_dataloader_HAR

import config
from tqdm import tqdm
from models.multi_models import model_HAPT,model_HAR,model_gru_HAPT,model_transformer_HAPT

from inputpipeline.preprocess_input import preprocess_input
from evaluation.metrics import compute_accuracy
import matplotlib
import matplotlib.pyplot as plt

from evaluation.confusion_matrix import pp_matrix_from_data



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
test_loader = get_dataloader(mode='test',Window_shift=125,Window_length=250,
                                   batch_size=batch_size,shuffle=False,root_path='./RawData/',how=how)
file_name = 'HAPT_lstm_s2sb64hidden12layer2best_epoch.pth'
# file_name = 'HAPT_transformer_validation_accuracyb32hidden24layer2best_epoch.pth'
mdl = model_HAPT(batchsize=batch_size,device=device,hidden_size =12,
            num_layers=2,bidirectional=True,window_size=250,how=how).to(device)
# mdl = model_transformer_HAPT(batchsize=batch_size)





mdl.load_state_dict(torch.load(file_name,map_location=torch.device('cpu')))
mdl.eval()
loss_computer = torch.nn.CrossEntropyLoss()
loss_val = 0
acu = 0
accu=0
true_dict = {f'{s}': {f'{e}':[] for e in (2*s,2*s+1)} for s in range(22,28)}
result_dict = {f'{s}': {f'{e}':[] for e in (2*s,2*s+1)} for s in range(22,28)}
if how == 's2l':
    for step_test, test_data in tqdm(enumerate(test_loader)):
        test_input = test_data[0]
        test_label = test_data[1]
        file = test_data[2]
        interval = test_data[3]
        with torch.no_grad():
            test_input, test_label = preprocess_input(test_input, test_label, device)
            test_output = mdl(test_input)
            for i in range(batch_size):
                usr = str(int(file[0][i]))
                exp = str(int(file[1][i]))
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
    np.save('result_dict_s2l.npy',result_dict)
    np.save('true_dict_s2l.npy',true_dict)


elif how == 's2s':
    for step_test, test_data in tqdm(enumerate(test_loader)):
        test_input = test_data[0]
        test_label = test_data[1].view(-1)
        source = test_data[2].view(-1,3)
        with torch.no_grad():
            test_input, test_label = preprocess_input(test_input, test_label, device)
            test_output = mdl(test_input).view(-1,12)
            for i in range(batch_size*window_size):
                usr = str(int(source[i][0]))
                exp = str(int(source[i][1]))
                time_instance = int(source[i][2])
                true_dict[usr][exp].append([time_instance,test_label[i]])
                result_dict[usr][exp].append([time_instance,np.argmax(test_output, axis=1)[i]])
            accuracy = compute_accuracy(test_output.detach().cpu().numpy(), test_label.detach().cpu().numpy())  ##??
            # print(accuracy)
            accu = (accu * step_test + accuracy) / (step_test + 1)  ##validation set的平均accuracy
    print('accuracy =',accu)
    np.save('result_dict_s2s.npy',result_dict)
    np.save('true_dict_s2s.npy',true_dict)




# label_dict = {0: 'walking', 1: 'walking_upstairs', 2: 'walking_downstairs', 3: 'sitting', 4: 'standing', 5: 'laying',
#               6: 'stand_to_sit', 7: 'sit_to_stand', 8: 'sit_to_lie', 9: 'lie_to_sit', 10: 'stand_to_lie',
#               11: 'lie_to_stand'}
# results =[]
# y_true = []
# for usr,files in result_dict.items():
#     for file,data in files.items():
#         results.append(int(data[1]))
# for usr,files in true_dict.items():
#     for file,data in files.items():
#         y_true.append(int(data[1]))
# results = np.array(results)
# y_true = np.array(y_true)
# label_list = []
# all_possible_labels = np.concatenate((results,y_true))
# for i in range(12):
#         if i in all_possible_labels:
#                 label_list.append(label_dict[i])
# pp_matrix_from_data(y_true, results,label_list=label_list)



# print(result_dict['23']['47'])




