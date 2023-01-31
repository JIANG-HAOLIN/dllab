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

test_loader = get_dataloader(mode='test',Window_shift=125,Window_length=250,
                                   batch_size=batch_size,shuffle=False,root_path='./RawData/')
# test_loader = get_dataloader_HAR(mode='test', Window_shift=125, Window_length=250, batch_size=batch_size,
#                                        shuffle=True, root_path='./realworld2016_dataset/')

file_name = 'bi4b64hidden12layer2best_epoch.pth'
# file_name = 'HAPT_transformer_validation_accuracyb32hidden24layer2best_epoch.pth'
mdl = model_HAPT(batchsize=batch_size,device=device,hidden_size =12,
            num_layers=2,bidirectional=True,window_size=250).to(device)
# mdl = model_transformer_HAPT(batchsize=batch_size)
mdl.load_state_dict(torch.load(file_name,map_location=torch.device('cpu')))
mdl.eval()

loss_computer = torch.nn.CrossEntropyLoss()
loss_val = 0
acu = 0


##测试test dataset
# for step,(sample,label,_,_) in tqdm(enumerate(validation_loader)):
#     if step <=0 :
#         print(sample,label)
accu=0
true_dict = {f'{s}': {f'{e}':[] for e in (2*s,2*s+1)} for s in range(22,28)}
result_dict = {f'{s}': {f'{e}':[] for e in (2*s,2*s+1)} for s in range(22,28)}

for step_test, (test_input, test_label, file, interval) in tqdm(enumerate(test_loader)):
    with torch.no_grad():
        test_input, test_label = preprocess_input(test_input, test_label, device)
        for i in range(batch_size):
            usr = str(int(file[0][i]))
            exp = str(int(file[1][i]))
            start = int(interval[0][i])
            end = int(interval[1][i])
            true_dict[usr][exp].append([[start,end],test_label])
            test_output = mdl(test_input)
            result_dict[usr][exp].append([[start,end],np.argmax(test_output, axis=1)])
        loss_val = (loss_val * step_test + loss_computer(test_output, test_label)) / (step_test + 1)  ##??
        # print(val_output,val_label)
        accuracy = compute_accuracy(test_output.detach().cpu().numpy(), test_label.detach().cpu().numpy())  ##??
        # print(accuracy)
        accu = (accu * step_test + accuracy) / (step_test + 1)  ##validation set的平均accuracy
np.save('result_dict.npy',result_dict)
np.save('true_dict.npy',true_dict)


print(accu)
print(result_dict['23']['47'])




