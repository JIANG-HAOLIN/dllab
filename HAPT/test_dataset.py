import torch
from inputpipeline.datasets import get_dataloader
import config
from tqdm import tqdm
from models.simple_model import model
from inputpipeline.preprocess_input import preprocess_input
from evaluation.metrics import compute_accuracy


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
                                   batch_size=32,shuffle=True,root_path='./RawData/')


mdl = model(batchsize=32,device=device,hidden_size =12,
            num_layers=2,bidirectional=True,window_size=250).to(device)
mdl.load_state_dict(torch.load('bi4b64hidden12layer2best_epoch.pth',map_location=torch.device('cpu')))
mdl.eval()

loss_computer = torch.nn.CrossEntropyLoss()
loss_val = 0
acu = 0


##测试test dataset
# for step,(sample,label,_,_) in tqdm(enumerate(validation_loader)):
#     if step <=0 :
#         print(sample,label)
accu=0
for step_test, (test_input, test_label, _, _) in tqdm(enumerate(test_loader)):
    with torch.no_grad():
        test_input, test_label = preprocess_input(test_input, test_label, device)
        test_output = mdl(test_input)
        loss_val = (loss_val * step_test + loss_computer(test_output, test_label)) / (step_test + 1)  ##??
        # print(val_output,val_label)
        accuracy = compute_accuracy(test_output.detach().cpu().numpy(), test_label.detach().cpu().numpy())  ##??
        # print(accuracy)
        accu = (accu * step_test + accuracy) / (step_test + 1)  ##validation set的平均accuracy
print(accu)