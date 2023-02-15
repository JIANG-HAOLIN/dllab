import wandb
import math
import torch
from models.multi_models import model_HAPT,model_HAR,model_gru_HAPT,model_transformer_HAPT
from inputpipeline.datasets import get_dataloader
from inputpipeline.preprocess_input import preprocess_input
from inputpipeline.HAR_Dataset import get_dataloader_HAR


import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from evaluation.metrics import compute_accuracy


def Trainer(mdl = None,
           loss_computer = None,
            optimizer = None,
            epoch = None,
            learning_rate = 2e-4,
            train_loader = None,
            validation_loader = None,
            out_name = None,
            device = None,
            batch_size = None,
            how = 's2s',
            wandb_control = False,
            label_size =12
                        ):
    writer = SummaryWriter("logs")
    loss_list=[]
    accu_list=[]
    cur = []
    best_accu = 0
    cur_iter = 0
    for epoch in range(epoch):
        for step,train_data in enumerate(train_loader):##!!!!!!
            # cur_iter = epoch*(int(2550/batch_size)) + step + 1
            input = train_data[0]
            label = train_data[1].view(-1)
            cur_iter += 1
            #input.shape [3,250,6]
            input,label = preprocess_input(input,label,device)
            output = mdl(input).view(-1,label_size)
            loss = loss_computer(output,label)

    ##### wandb
            # wandb.log({"loss": loss})

            # Optional
            # wandb.watch(mdl)
    ##### wandb

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if cur_iter % 10 == 0:
                writer.add_scalar("train loss", loss.item(), cur_iter)
                cur.append(cur_iter)
                mdl.eval()
                accu = 0
                loss_val = 0

                for step_val,val_data in enumerate(validation_loader):
                    with torch.no_grad():
                        val_input = val_data[0]
                        val_label = val_data[1].view(-1)
                        val_input, val_label = preprocess_input(val_input, val_label, device)
                        val_output = mdl(val_input).view(-1,label_size)
                        loss_val = (loss_val * step_val + loss_computer(val_output, val_label)) / (step_val + 1)##??
                        # print(val_output,val_label)
                        accuracy = compute_accuracy(val_output.detach().cpu().numpy(), val_label.detach().cpu().numpy())##??
                        # print(accuracy)
                        accu = (accu * step_val + accuracy) / (step_val + 1)##validation set的平均accuracy

                mdl.train()
                loss_list.append(loss_val.item())
                accu_list.append(accu)
                writer.add_scalar("validation loss", loss_val.item(), cur_iter)
                writer.add_scalar("validation accuracy", accu, cur_iter)

                if wandb_control:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': loss,
                        'val_acc': accu,
                        'val_loss': loss_val
                    })

                if accu > best_accu:
                    best_accu = accu
                    print(f'at epoch{epoch} has best validation accuracy:',best_accu)
                    torch.save(mdl.state_dict(), out_name+"best_epoch.pth")
                    if best_accu >= 0.8:
                        torch.save(mdl.state_dict(),'pth'+ out_name + f'_{int(100*best_accu)}_'+f'{cur_iter}_'+'.pth')


                fig = plt.figure()
                plt.plot(cur, loss_list, label="val_loss")  # plot example
                plt.legend()
                fig.savefig(f'{out_name}_val_loss.png')

                fig2 = plt.figure()
                plt.plot(cur, accu_list, label="accuracy")  # plot example
                plt.title(f'best accuracy:{best_accu} at iter{cur_iter}')
                plt.legend()
                fig2.savefig(f'{out_name}_val_accuracy.png')

                # plt.close(fig)
                # plt.close(fig2)
                plt.close('all')

        writer.close()
        print('epoch:',f'{epoch}','\n','current iteration=',f'{cur_iter}',)
