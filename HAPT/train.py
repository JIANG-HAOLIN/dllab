import torch
from HAPT.models.simple_model import model
from HAPT.inputpipeline.datasets import get_dataloader
from HAPT.inputpipeline.preprocess_input import preprocess_input

import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from HAPT.evaluation.metrics import compute_matrix_CE


batch_size = 3
device = 'cpu'
loss_computer = torch.nn.CrossEntropyLoss()
train_loader = get_dataloader(mode='train',Window_shift=125,Window_length=250,batch_size=batch_size,shuffle=True)
validation_loader = get_dataloader(mode='validation',Window_shift=125,Window_length=250,batch_size=batch_size,shuffle=True)
mdl = model().to(device)
writer = SummaryWriter("logs")
opt = torch.optim.Adam(mdl.parameters(),lr=1e-4,weight_decay=5e-3)

loss_list=[]
accu_list=[]
cur = []
epoch = 1000
best_accu = 0


for epoch in range(epoch):
    for step,(input,label) in tqdm(enumerate(train_loader)):##!!!!!!
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

            for step_val,(val_input,val_label) in tqdm(enumerate(validation_loader)):
                with torch.no_grad():
                    val_input, val_label = preprocess_input(val_input, val_label, device)
                    val_output = mdl(val_input)
                    loss_val = (loss_val * step_val + loss_computer(val_output, val_label)) / (step_val + 1)##??
                    _, accuracy = compute_matrix_CE(output.detach().cpu().numpy(), label.detach().cpu().numpy())##??
                    accu = (accu * step_val + accuracy) / (step_val + 1)

            mdl.train()
            loss_list.append(loss_val.item())
            accu_list.append(accu)##??
            writer.add_scalar("validation loss", loss_val.item(), cur_iter)
            writer.add_scalar("validation accuracy", accu, cur_iter)

            if accu > best_accu:
                best_acu = accu
                torch.save(mdl.state_dict(), "./best_epoch.pth")

            fig = plt.figure()
            plt.plot(cur, loss_list, label="val_loss")  # plot example
            plt.legend()
            fig.savefig('val_loss.png')

            fig2 = plt.figure()
            plt.plot(cur, accu_list, label="accuracy")  # plot example
            plt.legend()
            fig2.savefig('val_accuracy.png')

    writer.close()


