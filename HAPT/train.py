import torch
from models.multi_models import model_HAPT,model_HAR,model_gru_HAPT,model_transformer_HAPT
from inputpipeline.datasets import get_dataloader
from inputpipeline.preprocess_input import preprocess_input
from inputpipeline.HAR_Dataset import get_dataloader_HAR


import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from evaluation.metrics import compute_accuracy
import config


opt = config.read_arguments()

batch_size = opt.batch_size
root_path =opt.root_path
shift_length = opt.shift_length
window_size = opt.window_size
device = opt.device
hidden_size = opt.hidden_size
num_layers = opt.num_layers
lr = opt.lr
bidirectional = opt.bidirectional
dataset = opt.dataset
structure = opt.structure
out_name = opt.dataset+'_'+opt.structure+'_'+opt.out_name+'b'+str(batch_size)+'hidden'+str(hidden_size)+'layer'+str(num_layers)


loss_computer = torch.nn.CrossEntropyLoss()
if dataset == 'HAPT':
    train_loader = get_dataloader(mode='train',Window_shift=125,Window_length=250,
                                  batch_size=batch_size,shuffle=True,root_path='./RawData/')
    validation_loader = get_dataloader(mode='validation',Window_shift=125,Window_length=250,
                                       batch_size=batch_size,shuffle=True,root_path='./RawData/')
    if structure == 'lstm':
        mdl = model_HAPT(batchsize=batch_size, device=device, hidden_size=hidden_size,
                         num_layers=num_layers, bidirectional=bidirectional, window_size=window_size).to(device)
    elif structure == 'gru':
        mdl = model_gru_HAPT(batchsize=batch_size, device=device, hidden_size=hidden_size,
                         num_layers=num_layers, bidirectional=bidirectional, window_size=window_size).to(device)
    elif structure == 'transformer':
        mdl = model_transformer_HAPT(batchsize=batch_size)
elif dataset == 'HAR':
    train_loader = get_dataloader_HAR(mode='train',Window_shift=125,Window_length=250,batch_size=batch_size,
                                      shuffle=True,root_path='./realworld2016_dataset/')
    validation_loader = get_dataloader_HAR(mode='validation',Window_shift=125,Window_length=250,batch_size=batch_size,
                                           shuffle=True,root_path='./realworld2016_dataset/')
    if structure == 'lstm':
        mdl = model_HAR(batchsize=batch_size, device=device, hidden_size=hidden_size,
                         num_layers=num_layers, bidirectional=bidirectional, window_size=window_size).to(device)
    elif structure == 'transformer':
        mdl = model_transformer_HAPT(batchsize=batch_size)

writer = SummaryWriter("logs")
opt = torch.optim.Adam(mdl.parameters(),lr=lr,weight_decay=5e-3)

loss_list=[]
accu_list=[]
cur = []
epoch = 1000
best_accu = 0


for epoch in range(epoch):
    for step,(input,label,_,_) in enumerate(train_loader):##!!!!!!
        cur_iter = epoch*(int(2550/batch_size)) + step + 1
        #input.shape [3,250,6]
        input,label = preprocess_input(input,label,device)
        output = mdl(input)
        loss = loss_computer(output,label)
        opt.zero_grad()
        loss.backward()##not backwards....
        opt.step()

        if cur_iter % 10 == 0:
            writer.add_scalar("train loss", loss.item(), cur_iter)
            cur.append(cur_iter)
            mdl.eval()
            accu = 0
            loss_val = 0

            for step_val,(val_input,val_label,_,_) in enumerate(validation_loader):
                with torch.no_grad():
                    val_input, val_label = preprocess_input(val_input, val_label, device)
                    val_output = mdl(val_input)
                    loss_val = (loss_val * step_val + loss_computer(val_output, val_label)) / (step_val + 1)##??
                    # print(val_output,val_label)
                    accuracy = compute_accuracy(val_output.detach().cpu().numpy(), val_label.detach().cpu().numpy())##??
                    # print(accuracy)
                    accu = (accu * step_val + accuracy) / (step_val + 1)##validation set的平均accuracy

            mdl.train()
            loss_list.append(loss_val.item())
            accu_list.append(accu)##??
            writer.add_scalar("validation loss", loss_val.item(), cur_iter)
            writer.add_scalar("validation accuracy", accu, cur_iter)

            if accu > best_accu:
                best_accu = accu
                print(f'at epoch{epoch} has best validation accuracy:',best_accu)
                torch.save(mdl.state_dict(), out_name+"best_epoch.pth")

            fig = plt.figure()
            plt.plot(cur, loss_list, label="val_loss")  # plot example
            plt.legend()
            fig.savefig(f'{out_name}_val_loss.png')

            fig2 = plt.figure()
            plt.plot(cur, accu_list, label="accuracy")  # plot example
            plt.legend()
            fig2.savefig(f'{out_name}_val_accuracy.png')

    writer.close()


